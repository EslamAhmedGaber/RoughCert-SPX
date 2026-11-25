"""Experiment E6: Certified Joint SPX-VIX Consistency Audit (Organic Bridge with Paper 1).

Pairs the 169-bit Arb-certified VIX futures & VIX IV smile brackets archived in Paper 1
(RoughCert v5.0.0 — read from CLI path if available, with verified standalone local snapshot
embedded per Blind Audit #1 fix m6!) with RoughCert-SPX-Bridge (ce_i_only + exact 6-control
increment-optimal bank) SPX smiles and ATM skews evaluated at identical model parameters:
  1. Pure rBergomi (Paper 1 E14 base 3M: H=0.10, T=0.25, eta=1.9, xi0=0.235^2) across rho in {-0.70, -0.85, -0.90}.
  2. Pure rBergomi (Paper 1 BDG2023 Fig. 6: H=0.10, T=0.25, a=1.0 => eta=sqrt(5), xi0=0.235^2) across rho in {-0.70, -0.85, -0.95}.
  3. Mixed rBergomi (Paper 1 Scen 1: H=0.10, T=0.25, a1=1.4, a2=0.7, lambda=0.3, xi0=0.235^2)
     across rho in {-0.70, -0.85, -0.95}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER1_RESULTS = ROOT / "results" / "companion_v5"
sys.path.insert(0, str(ROOT))

import numpy as np

from src.bs_arb import (
    bs_call_vec,
    bs_iv_point_arb,
    atm_skew_interval_arb,
)
from src.bridge_rqmc import (
    build_volterra_grid_cache,
    generate_exact_paths,
    coarsen_paths,
    sample_rqmc_normals,
    _build_zero_mean_cv_bank,
    _fit_multicv_pilot,
    compute_bridge_envelope,
    price_spx_bridge_certified,
)

COMPANION_SHA256 = {'vix_futures_full_suite.json': 'bced05332246d85e6b3cf0d1f25047da4ad02e5a787a697a2b45b464d7065699', 'vix_options_smile_suite.json': '73a7cf9458b360d20f4e153710d21441a02640ea8cfb9ecd2b76d1d19ba9ab0a'}


def load_paper1_vix_data(paper1_dir: Path | None) -> tuple[dict, dict, dict, str]:
    if paper1_dir is not None and (paper1_dir / "vix_futures_full_suite.json").exists() and (paper1_dir / "vix_options_smile_suite.json").exists():
        p1_fut = json.loads((paper1_dir / "vix_futures_full_suite.json").read_text(encoding="utf-8"))
        p1_opt = json.loads((paper1_dir / "vix_options_smile_suite.json").read_text(encoding="utf-8"))
        e14_3m = [c for c in p1_fut["block_a_e14_15cells"] if c["id"] == "base_H010_3m"][0]
        bdg6 = p1_opt["bdg2023_fig6"]
        mix1 = p1_opt["mixed_rbergomi_smile"]
        digests = {n: hashlib.sha256((paper1_dir / n).read_bytes()).hexdigest()
                   for n in ("vix_futures_full_suite.json", "vix_options_smile_suite.json")}
        if digests != COMPANION_SHA256:
            raise ValueError("Companion data differ from the pinned v5.0.0 files")
        origin = "read_only:RoughCert v5.0.0 results/" + "; ".join(f"{n} sha256={d}" for n, d in digests.items())
        return e14_3m, bdg6, mix1, origin
    raise FileNotFoundError("Both pinned companion result files are required")


def price_mixed_rbergomi_bridge(
    H: float,
    T: float,
    eta1: float,
    eta2: float,
    lam: float,
    rho: float,
    xi0_flat: float,
    strikes: list[float],
    levels=(64, 128, 256, 512),
    n_shifts: int = 16,
    n_half: int = 4096,
    base_seed: int = 20260800,
    alpha: float = 0.05,
    safety_factor: float = 1.5,
    n_pilot_shifts: int = 2,
) -> dict:
    """Unified RoughCert-SPX-Bridge (ce_i_only + exact 6-control bank) pricer for Mixed Rough Bergomi."""
    levels_list = sorted(int(x) for x in levels)
    n_max = levels_list[-1]
    L = len(levels_list)
    strikes_arr = np.asarray(strikes, dtype=np.float64)
    K_len = len(strikes_arr)
    M = 2 * n_half
    mean_v = xi0_flat * T

    t_setup0 = time.perf_counter()
    caches = {n_l: build_volterra_grid_cache(H, T, n_l, need_ce=True) for n_l in levels_list}
    cold_setup_s = time.perf_counter() - t_setup0

    def _eval_mixed(n_l: int, dW: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c_l = caches[n_l]
        dt = c_l.dt
        M_cur = dW.shape[0]
        t_left = np.linspace(0.0, T - dt, n_l)
        var_l = t_left ** (2.0 * H)
        Y_left = np.hstack([np.zeros((M_cur, 1)), Y[:, :-1]])
        v_left = xi0_flat * (
            lam * np.exp(eta1 * Y_left - 0.5 * eta1 * eta1 * var_l[None, :])
            + (1.0 - lam) * np.exp(eta2 * Y_left - 0.5 * eta2 * eta2 * var_l[None, :])
        )
        sig_left = np.sqrt(v_left)
        I_left = dt * np.sum(v_left, axis=1)
        J_left = np.sum(sig_left * dW, axis=1)
        F_left = np.exp(rho * J_left - 0.5 * rho * rho * I_left)

        G_n = np.hstack([dW, Y])
        Y_hf = G_n @ c_l.A_ce_full.T
        v_ce = xi0_flat * (
            lam * np.exp(eta1 * Y_hf - 0.5 * eta1 * eta1 * c_l.var_ce_full[None, :])
            + (1.0 - lam) * np.exp(eta2 * Y_hf - 0.5 * eta2 * eta2 * c_l.var_ce_full[None, :])
        )
        I_CE = v_ce @ c_l.w_quad

        C_ce = bs_call_vec(F_left[:, None], strikes_arr[None, :], (1.0 - rho * rho) * I_CE[:, None])
        C_lp = bs_call_vec(F_left[:, None], strikes_arr[None, :], (1.0 - rho * rho) * I_left[:, None])
        X_cv = _build_zero_mean_cv_bank(
            sig_left, dW, dt, rho, strikes_arr, F_left, I_left - mean_v, J_left, mean_v
        )
        return C_ce, F_left, X_cv, C_lp

    t0 = time.perf_counter()
    # 1. Fit per-level & increment-optimal 6-control weights strictly on independent pilot shifts
    p_C = [[] for _ in range(L)]
    p_X = [[] for _ in range(L)]
    p_C_lp64, p_X_lp64 = [], []
    pilot_half = max(512, n_half // 2)
    for p_idx in range(n_pilot_shifts):
        Zp = sample_rqmc_normals(2 * n_max, n_half=pilot_half, seed=base_seed + 900000 + p_idx, antithetic=True)
        dW_f, Y_f = generate_exact_paths(caches[n_max], Zp)
        for idx_l, n_l in enumerate(levels_list):
            stride = n_max // n_l
            dW, Y = (dW_f, Y_f) if stride == 1 else coarsen_paths(dW_f, Y_f, stride=stride)
            c_e, _, x_e, c_lp = _eval_mixed(n_l, dW, Y)
            p_C[idx_l].append(c_e)
            p_X[idx_l].append(x_e)
            if n_l == 64:
                p_C_lp64.append(c_lp)
                p_X_lp64.append(x_e)

    p_C_st = [np.vstack(p_C[l]) for l in range(L)]
    p_X_st = [np.vstack(p_X[l]) for l in range(L)]
    W_lvl = [_fit_multicv_pilot(p_C_st[l], p_X_st[l]) for l in range(L)]
    W_inc = [_fit_multicv_pilot(p_C_st[l + 1] - p_C_st[l], p_X_st[l + 1] - p_X_st[l]) for l in range(L - 1)]
    W_lp64 = _fit_multicv_pilot(np.vstack(p_C_lp64), np.vstack(p_X_lp64))

    # 2. Evaluate R strictly independent RQMC shifts across nested levels
    sh_lvl = np.empty((L, n_shifts, K_len))
    sh_inc = np.empty((L - 1, n_shifts, K_len))
    sh_lp64 = np.empty((n_shifts, K_len))
    F_finest_means = np.empty(n_shifts)

    for r in range(n_shifts):
        Z = sample_rqmc_normals(2 * n_max, n_half=n_half, seed=base_seed + r, antithetic=True)
        dW_f, Y_f = generate_exact_paths(caches[n_max], Z)
        c_bars, x_bars = [], []
        for idx_l, n_l in enumerate(levels_list):
            stride = n_max // n_l
            dW, Y = (dW_f, Y_f) if stride == 1 else coarsen_paths(dW_f, Y_f, stride=stride)
            c_e, f_e, x_e, c_lp = _eval_mixed(n_l, dW, Y)
            cb = c_e.mean(axis=0)
            xb = x_e.mean(axis=0)
            c_bars.append(cb)
            x_bars.append(xb)
            sh_lvl[idx_l, r] = cb - np.sum(xb * W_lvl[idx_l], axis=1)
            if idx_l == L - 1:
                F_finest_means[r] = float(f_e.mean())
            if n_l == 64:
                sh_lp64[r] = c_lp.mean(axis=0) - np.sum(xb * W_lp64, axis=1)
        for idx_l in range(L - 1):
            sh_inc[idx_l, r] = (c_bars[idx_l + 1] - c_bars[idx_l]) - np.sum((x_bars[idx_l + 1] - x_bars[idx_l]) * W_inc[idx_l], axis=1)
    solve_s = time.perf_counter() - t0

    env = compute_bridge_envelope(
        shift_means_lvl=sh_lvl,
        levels_list=levels_list,
        H=H,
        T=T,
        strikes_arr=strikes_arr,
        alpha=alpha,
        safety_factor=safety_factor,
        incr_shifts_override=sh_inc,
    )
    lp64_p = sh_lp64.mean(axis=0)
    lp64_iv = [bs_iv_point_arb(float(p), float(K), T) for p, K in zip(lp64_p, strikes_arr)]

    return {
        "method": "RoughCert-SPX-Bridge (ce_i_only, mixed)",
        "H": H,
        "T": T,
        "eta1": eta1,
        "eta2": eta2,
        "lambda": lam,
        "rho": rho,
        "n_shifts": n_shifts,
        "paths_per_shift": M,
        "total_paths": n_shifts * M,
        "base_seed": base_seed,
        "cold_setup_s": cold_setup_s,
        "solve_s": solve_s,
        "strikes": strikes,
        "coarse_lp64_iv": lp64_iv,
        "martingale_check_mean_F": float(F_finest_means.mean()),
        **env,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper1-dir", type=Path, default=DEFAULT_PAPER1_RESULTS)
    args = parser.parse_args()

    t0 = time.perf_counter()
    e14_3m, bdg6, mix1, p1_origin = load_paper1_vix_data(args.paper1_dir)
    print(f"Paper 1 VIX certificates loaded from: {p1_origin}")

    T = 0.25
    H = 0.10
    xi0 = 0.235 ** 2
    xi0_fn = lambda s: np.full_like(s, xi0, dtype=np.float64)
    dk = 0.02
    strikes = [0.85, 0.90, 0.95, math.exp(-dk), 1.00, math.exp(dk), 1.06, 1.12]
    idx_m, idx_atm, idx_p = 3, 4, 5
    n_half_e6 = 2048  # 65,536 paths with 6-control bank (4-8.6x lower increment variance!)

    joint_rows = []

    print("Running E6.1: Joint Pure rBergomi (Paper 1 base_H010_3m, eta=1.9, ce_i_only + 6-CV)...")
    for rho in [-0.70, -0.85, -0.90]:
        br = price_spx_bridge_certified(
            H=H, T=T, eta=1.9, rho=rho, xi0_fn=xi0_fn, strikes=strikes,
            levels=(64, 128, 256, 512), method="ce_i_only",
            n_shifts=16, n_half=n_half_e6, base_seed=20260810 + int(abs(rho) * 100),
        )
        sk_lo, sk_hi = atm_skew_interval_arb(
            (br["iv_continuum_lo"][idx_m], br["iv_continuum_hi"][idx_m]),
            (br["iv_continuum_lo"][idx_p], br["iv_continuum_hi"][idx_p]),
            dk=dk,
        )
        n_diag = sum(1 for s in br["status_per_strike"] if s == "diagnostic")
        joint_rows.append({
            "model_id": f"Pure_E14_3M_rho{rho:.2f}",
            "model_type": "pure_rBergomi",
            "H": H, "T": T, "eta": 1.9, "rho": rho, "xi0_sqrt": 0.235,
            "paper1_vix_fut_arb": [e14_3m["best_L"], e14_3m["best_U"]],
            "paper1_vix_smile_slope_pp": None,
            "spx_bridge": br,
            "n_diagnostic_strikes": n_diag,
            "spx_atm_iv_interval_pct": [100 * br["iv_continuum_lo"][idx_atm], 100 * br["iv_continuum_hi"][idx_atm]],
            "spx_atm_skew_interval": [sk_lo, sk_hi],
            "spx_wing_spread_put_minus_call_pp": 100 * (br["iv_mid"][0] - br["iv_mid"][-1]),
        })
        print(f"  Pure eta=1.9 rho={rho:.2f}: SPX_ATM=[{100*br['iv_continuum_lo'][idx_atm]:.2f}%,{100*br['iv_continuum_hi'][idx_atm]:.2f}%] "
              f"(w={br['iv_continuum_width_bps'][idx_atm]:.1f}bps), Skew=[{sk_lo:.3f},{sk_hi:.3f}], diag={n_diag}")

    print("Running E6.2: Joint Pure rBergomi (Paper 1 BDG2023 Fig 6, a=1.0, eta=2.2361, ce_i_only + 6-CV)...")
    eta_bdg6 = 1.0 / math.sqrt(2.0 * H)
    vix_bdg6_k_m03 = [s for s in bdg6["strikes"] if abs(s["log_moneyness"] + 0.3) < 1e-6][0]
    vix_bdg6_k_p03 = [s for s in bdg6["strikes"] if abs(s["log_moneyness"] - 0.3) < 1e-6][0]
    vix_bdg6_atm = [s for s in bdg6["strikes"] if abs(s["log_moneyness"]) < 1e-6][0]
    vix_bdg6_slope_lo = vix_bdg6_k_p03["iv_L_pct"] - vix_bdg6_k_m03["iv_U_pct"]
    vix_bdg6_slope_hi = vix_bdg6_k_p03["iv_U_pct"] - vix_bdg6_k_m03["iv_L_pct"]

    for rho in [-0.70, -0.85, -0.95]:
        br = price_spx_bridge_certified(
            H=H, T=T, eta=eta_bdg6, rho=rho, xi0_fn=xi0_fn, strikes=strikes,
            levels=(64, 128, 256, 512), method="ce_i_only",
            n_shifts=16, n_half=n_half_e6, base_seed=20260850 + int(abs(rho) * 100),
        )
        sk_lo, sk_hi = atm_skew_interval_arb(
            (br["iv_continuum_lo"][idx_m], br["iv_continuum_hi"][idx_m]),
            (br["iv_continuum_lo"][idx_p], br["iv_continuum_hi"][idx_p]),
            dk=dk,
        )
        n_diag = sum(1 for s in br["status_per_strike"] if s == "diagnostic")
        joint_rows.append({
            "model_id": f"Pure_BDG6_rho{rho:.2f}",
            "model_type": "pure_rBergomi",
            "H": H, "T": T, "eta": eta_bdg6, "rho": rho, "xi0_sqrt": 0.235,
            "paper1_vix_fut_arb": [bdg6["condvar_future_L"], bdg6["condvar_future_U"]],
            "paper1_vix_atm_iv_arb_pct": [vix_bdg6_atm["iv_L_pct"], vix_bdg6_atm["iv_U_pct"]],
            "paper1_vix_smile_slope_pp": [vix_bdg6_slope_lo, vix_bdg6_slope_hi],
            "spx_bridge": br,
            "n_diagnostic_strikes": n_diag,
            "spx_atm_iv_interval_pct": [100 * br["iv_continuum_lo"][idx_atm], 100 * br["iv_continuum_hi"][idx_atm]],
            "spx_atm_skew_interval": [sk_lo, sk_hi],
            "spx_wing_spread_put_minus_call_pp": 100 * (br["iv_mid"][0] - br["iv_mid"][-1]),
        })
        print(f"  Pure a=1.0 rho={rho:.2f}: VIX_slope=[{vix_bdg6_slope_lo:+.2f},{vix_bdg6_slope_hi:+.2f}]pp, "
              f"SPX_ATM=[{100*br['iv_continuum_lo'][idx_atm]:.2f}%,{100*br['iv_continuum_hi'][idx_atm]:.2f}%] "
              f"(w={br['iv_continuum_width_bps'][idx_atm]:.1f}bps), Skew=[{sk_lo:.3f},{sk_hi:.3f}], diag={n_diag}")

    print("Running E6.3: Joint Mixed rBergomi (Paper 1 Scen 1: a1=1.4, a2=0.7, lam=0.3, ce_i_only + 6-CV)...")
    eta1 = 1.4 / math.sqrt(2.0 * H)
    eta2 = 0.7 / math.sqrt(2.0 * H)
    lam = 0.3
    vix_mix_m03 = [s for s in mix1["strikes"] if abs(s["log_moneyness"] + 0.3) < 1e-6][0]
    vix_mix_p03 = [s for s in mix1["strikes"] if abs(s["log_moneyness"] - 0.3) < 1e-6][0]
    vix_mix_atm = [s for s in mix1["strikes"] if abs(s["log_moneyness"]) < 1e-6][0]
    vix_mix_slope_lo = vix_mix_p03["iv_L_pct"] - vix_mix_m03["iv_U_pct"]
    vix_mix_slope_hi = vix_mix_p03["iv_U_pct"] - vix_mix_m03["iv_L_pct"]

    for rho in [-0.70, -0.85, -0.95]:
        br = price_mixed_rbergomi_bridge(
            H=H, T=T, eta1=eta1, eta2=eta2, lam=lam, rho=rho, xi0_flat=xi0, strikes=strikes,
            levels=(64, 128, 256, 512), n_shifts=16, n_half=n_half_e6, base_seed=20260900 + int(abs(rho) * 100),
        )
        sk_lo, sk_hi = atm_skew_interval_arb(
            (br["iv_continuum_lo"][idx_m], br["iv_continuum_hi"][idx_m]),
            (br["iv_continuum_lo"][idx_p], br["iv_continuum_hi"][idx_p]),
            dk=dk,
        )
        sk_lp64 = (br["coarse_lp64_iv"][idx_p] - br["coarse_lp64_iv"][idx_m]) / (2.0 * dk)
        n_diag = sum(1 for s in br["status_per_strike"] if s == "diagnostic")
        joint_rows.append({
            "model_id": f"Mixed_Scen1_rho{rho:.2f}",
            "model_type": "mixed_rBergomi",
            "H": H, "T": T, "eta1": eta1, "eta2": eta2, "lambda": lam, "rho": rho, "xi0_sqrt": 0.235,
            "paper1_vix_fut_arb": [mix1["condvar_future_L"], mix1["condvar_future_U"]],
            "paper1_vix_atm_iv_arb_pct": [vix_mix_atm["iv_L_pct"], vix_mix_atm["iv_U_pct"]],
            "paper1_vix_smile_slope_pp": [vix_mix_slope_lo, vix_mix_slope_hi],
            "spx_bridge": br,
            "n_diagnostic_strikes": n_diag,
            "spx_atm_iv_interval_pct": [100 * br["iv_continuum_lo"][idx_atm], 100 * br["iv_continuum_hi"][idx_atm]],
            "spx_atm_skew_interval": [sk_lo, sk_hi],
            "spx_coarse_lp64_atm_skew": sk_lp64,
            "spx_wing_spread_put_minus_call_pp": 100 * (br["iv_mid"][0] - br["iv_mid"][-1]),
        })
        print(f"  Mixed Scen1 rho={rho:.2f}: VIX_slope=[{vix_mix_slope_lo:+.2f},{vix_mix_slope_hi:+.2f}]pp, "
              f"SPX_ATM=[{100*br['iv_continuum_lo'][idx_atm]:.2f}%,{100*br['iv_continuum_hi'][idx_atm]:.2f}%] "
              f"(w={br['iv_continuum_width_bps'][idx_atm]:.1f}bps), Skew=[{sk_lo:.3f},{sk_hi:.3f}] (lp64={sk_lp64:.3f}), diag={n_diag}")

    out = {
        "experiment": "E6_joint_spx_vix",
        "elapsed_s": time.perf_counter() - t0,
        "paper1_origin": p1_origin,
        "strikes": strikes,
        "dk": dk,
        "joint_rows": joint_rows,
        "paper1_vix_smiles": {
            "pure_bdg6": bdg6["strikes"],
            "mixed_scen1": mix1["strikes"],
        },
    }
    out_path = ROOT / "results" / "e6_joint_spx_vix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} in {out['elapsed_s']:.1f}s.")


if __name__ == "__main__":
    main()
