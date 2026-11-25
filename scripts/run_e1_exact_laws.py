"""Experiment E1: Exact-Law Correlated Stochastic Volatility Validation (Zero-MC Ground Truth).

Validates RoughCert-SPX-Bridge, RoughCert-SPX-CE (ce_i_only), Left-Point, and the
Jensen-J smoothing ablation (ce_adapted) against exact analytical Fourier prices:
  1. Correlated Heston (H = 1/2, rho = -0.7) -- Little Heston Trap + Lewis Fourier.
  2. Classical Correlated Stein-Stein (H = 1/2, rho = -0.7) -- Fredholm Galerkin vs Riccati ODE.
  3. Correlated Rough Stein-Stein (Abi Jaber 2022, H = 0.10, rho = -0.7) -- Carleman-Fredholm det_3
     regularised Galerkin (M=192 vs M=128 with exact degree-4M quadrature n_gl=max(2M+32,128))
     + full coupled exact-bridge envelope containment test across 7 strikes K in [0.85, 1.15]
     and levels n in (32, 64, 128, 256) using independent pilot-fitted CV and unified
     compute_bridge_envelope (Blind Audit #1 fixes C1, C2, M3, m3).
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT))

import numpy as np

from src.bs_arb import bs_call_vec, bs_iv_point_arb
from src.bridge_rqmc import (
    build_volterra_grid_cache,
    generate_exact_paths,
    coarsen_paths,
    sample_rqmc_normals,
    _fit_cv_weights_from_pilot,
    _apply_fixed_cv,
    compute_bridge_envelope,
)
from src.exact_laws import heston_call_prices, rough_stein_stein_exact_prices


def run_rss_coupled_bridge(
    H: float,
    T: float,
    sigma0: float,
    nu: float,
    rho: float,
    strikes: np.ndarray,
    levels=(32, 64, 128, 256),
    n_shifts: int = 16,
    n_half: int = 1024,
    base_seed: int = 20260101,
    alpha: float = 0.05,
    safety_factor: float = 1.5,
    n_pilot_shifts: int = 2,
):
    """Run coupled exact-bridge multilevel experiment on Correlated Rough Stein-Stein for
    leftpoint, ce_i_only (RoughCert-SPX-CE), and ce_adapted (Jensen-J ablation) on the
    EXACT SAME Brownian sample paths, using independent pilot-fitted CV and unified
    compute_bridge_envelope."""
    levels_list = list(levels)
    n_max = levels_list[-1]
    L = len(levels_list)
    K_len = len(strikes)
    M = 2 * n_half

    caches = {n_l: build_volterra_grid_cache(H, T, n_l, need_ce=True) for n_l in levels_list}
    D_full_map = {n_l: np.maximum(caches[n_l].s_quad ** (2.0 * H) - caches[n_l].var_ce_full, 0.0) for n_l in levels_list}

    methods = ("leftpoint", "ce_i_only", "ce_adapted")

    def _eval_rss_all(n_l: int, dW: np.ndarray, Y: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        c_l = caches[n_l]
        dt = c_l.dt
        M_cur = dW.shape[0]

        Y_left = np.empty((M_cur, n_l), dtype=np.float64)
        Y_left[:, 0] = 0.0
        if n_l > 1:
            Y_left[:, 1:] = Y[:, :-1]
        sig_l = sigma0 + nu * Y_left
        I_l = dt * np.sum(sig_l * sig_l, axis=1)
        J_l = np.sum(sig_l * dW, axis=1)
        F_l = np.exp(rho * J_l - 0.5 * (rho * rho) * I_l)

        G_n = np.hstack([dW, Y])
        Y_hf = G_n @ c_l.A_ce_full.T
        I_CE = ((sigma0 + nu * Y_hf) ** 2 + (nu * nu) * D_full_map[n_l][None, :]) @ c_l.w_quad

        sig_ce = np.empty((M_cur, n_l), dtype=np.float64)
        sig_ce[:, 0] = sigma0
        for i in range(1, n_l):
            yc = dW[:, :i] @ c_l.causal_weights_dw[i].T + Y[:, :i] @ c_l.causal_weights_y[i].T
            sig_ce[:, i] = sigma0 + 0.5 * nu * (yc[:, 0] + yc[:, 1])
        I_ad = dt * np.sum(sig_ce * sig_ce, axis=1)
        J_ce = np.sum(sig_ce * dW, axis=1)
        F_ce = np.exp(rho * J_ce - 0.5 * (rho * rho) * I_ad)

        V_vc = np.maximum(I_CE - (rho * rho) * np.minimum(I_CE, I_ad), 0.0)
        return {
            "leftpoint": (bs_call_vec(F_l[:, None], strikes[None, :], (1.0 - rho * rho) * I_l[:, None]), F_l),
            "ce_i_only": (bs_call_vec(F_l[:, None], strikes[None, :], (1.0 - rho * rho) * I_CE[:, None]), F_l),
            "ce_adapted": (bs_call_vec(F_ce[:, None], strikes[None, :], V_vc[:, None]), F_ce),
        }

    t0 = time.perf_counter()
    # 1. Fit independent finest-level pilot CV weights (martingale F_T - 1) so R shifts are strictly i.i.d.
    p_C = {m: [] for m in methods}
    p_F = {m: [] for m in methods}
    pilot_half = max(256, n_half // 2)
    for p_idx in range(n_pilot_shifts):
        Zp = sample_rqmc_normals(2 * n_max, n_half=pilot_half, seed=base_seed + 900000 + p_idx, antithetic=True)
        dW_f, Y_f = generate_exact_paths(caches[n_max], Zp)
        res_p = _eval_rss_all(n_max, dW_f, Y_f)
        for m in methods:
            p_C[m].append(res_p[m][0])
            p_F[m].append(res_p[m][1])
    beta_f_map = {
        m: _fit_cv_weights_from_pilot(np.vstack(p_C[m]), None, np.concatenate(p_F[m]))[1]
        for m in methods
    }

    # 2. Evaluate R strictly independent RQMC shifts across nested levels
    C_store = {m: [np.empty((n_shifts, M, K_len), dtype=np.float64) for _ in range(L)] for m in methods}
    F_store = {m: [np.empty((n_shifts, M), dtype=np.float64) for _ in range(L)] for m in methods}
    for r in range(n_shifts):
        Z = sample_rqmc_normals(2 * n_max, n_half=n_half, seed=base_seed + r, antithetic=True)
        dW_f, Y_f = generate_exact_paths(caches[n_max], Z)
        for idx_l, n_l in enumerate(levels_list):
            stride = n_max // n_l
            dW, Y = (dW_f, Y_f) if stride == 1 else coarsen_paths(dW_f, Y_f, stride=stride)
            res_l = _eval_rss_all(n_l, dW, Y)
            for m in methods:
                C_store[m][idx_l][r] = res_l[m][0]
                F_store[m][idx_l][r] = res_l[m][1]
    solve_s = time.perf_counter() - t0

    out_methods = {}
    for m_name in methods:
        shift_lvl = np.empty((L, n_shifts, K_len), dtype=np.float64)
        for idx_l in range(L):
            shift_lvl[idx_l] = _apply_fixed_cv(
                C_store[m_name][idx_l], None, None, F_store[m_name][idx_l], None, beta_f_map[m_name]
            )
        env = compute_bridge_envelope(
            shift_means_lvl=shift_lvl,
            levels_list=levels_list,
            H=H,
            T=T,
            strikes_arr=strikes,
            alpha=alpha,
            safety_factor=safety_factor,
        )
        lvl_means = np.asarray(env["level_means"])
        lvl_ivs = [
            [bs_iv_point_arb(float(lvl_means[l_i, k_i]), float(strikes[k_i]), T) for k_i in range(K_len)]
            for l_i in range(L)
        ]
        out_methods[m_name] = {
            **env,
            "level_ivs": lvl_ivs,
            "martingale_mean_F_finest": float(F_store[m_name][-1].mean()),
        }

    return {"solve_s": solve_s, "n_shifts": n_shifts, "total_paths": n_shifts * M, "methods": out_methods}


def main():
    t_all = time.perf_counter()
    strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15], dtype=np.float64)

    print("Running E1.1: Exact Correlated Heston...")
    heston = heston_call_prices(T=0.5, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.4, rho=-0.7, strikes=strikes)

    print("Running E1.2: Classical Stein-Stein (H=0.5, rho=-0.7) Fredholm vs Riccati...")
    ss_half = rough_stein_stein_exact_prices(H=0.5, T=0.5, sigma0=0.20, nu=0.25, rho=-0.7, strikes=strikes, M_op=64, M_check=32)

    print("Running E1.3: Correlated Rough Stein-Stein (H=0.10, rho=-0.7) exact Carleman-Fredholm det_3 (M=192 vs 128)...")
    rss_exact = rough_stein_stein_exact_prices(H=0.10, T=0.5, sigma0=0.20, nu=0.25, rho=-0.7, strikes=strikes, M_op=192, M_check=128)
    print(f"  RSS exact M=192 vs M=128 max_price_diff={rss_exact['max_refinement_price_diff']:.2e}, "
          f"mart_err={rss_exact['martingale_error']:.2e}, tr_rel_err={rss_exact['trace_rel_error']:.3f}")

    print("Running E1.4: Coupled Exact-Bridge RQMC on Correlated Rough Stein-Stein (levels 32,64,128,256)...")
    rss_bridge = run_rss_coupled_bridge(
        H=0.10, T=0.5, sigma0=0.20, nu=0.25, rho=-0.7, strikes=strikes,
        levels=(32, 64, 128, 256), n_shifts=16, n_half=1024, base_seed=20260101,
    )

    exact_p = np.array(rss_exact["prices"])
    exact_iv = np.array(rss_exact["ivs"])
    containment = {}
    for m_name, m_data in rss_bridge["methods"].items():
        p_lo = np.array(m_data["continuum_price_lo"])
        p_hi = np.array(m_data["continuum_price_hi"])
        contained = [bool(lo <= ex <= hi) for lo, ex, hi in zip(p_lo, exact_p, p_hi)]
        iv_bias_by_lvl = {}
        for idx_l, n_l in enumerate(m_data["levels"]):
            errs_bps = [1e4 * (est - ex) for est, ex in zip(m_data["level_ivs"][idx_l], exact_iv)]
            iv_bias_by_lvl[str(n_l)] = {
                "errs_bps": errs_bps,
                "max_abs_bps": max(abs(x) for x in errs_bps),
                "atm_bps": errs_bps[3],
            }
        n_diag = sum(1 for s in m_data["status_per_strike"] if s == "diagnostic")
        containment[m_name] = {
            "contained_per_strike": contained,
            "containment_rate": sum(contained) / len(contained),
            "n_diagnostic_strikes": n_diag,
            "iv_bias_by_level": iv_bias_by_lvl,
            "max_continuum_width_bps": max(m_data["iv_continuum_width_bps"]),
            "atm_continuum_width_bps": m_data["iv_continuum_width_bps"][3],
        }
        print(f"  {m_name:12s}: contained={sum(contained)}/{len(contained)}, diag={n_diag}, "
              f"n=32 max={iv_bias_by_lvl['32']['max_abs_bps']:.2f} bps, "
              f"n=256 max={iv_bias_by_lvl['256']['max_abs_bps']:.2f} bps, "
              f"ATM_256={iv_bias_by_lvl['256']['atm_bps']:+.2f} bps, "
              f"ATM_w={m_data['iv_continuum_width_bps'][3]:.2f} bps")

    result = {
        "experiment": "E1_exact_laws",
        "elapsed_s": time.perf_counter() - t_all,
        "strikes": strikes.tolist(),
        "heston_exact": heston,
        "stein_stein_H05_exact": ss_half,
        "rough_stein_stein_H010_exact": rss_exact,
        "rough_stein_stein_bridge": rss_bridge,
        "containment_summary": containment,
    }
    out_path = ROOT / "results" / "e1_exact_laws.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out_path} in {result['elapsed_s']:.1f}s.")


if __name__ == "__main__":
    main()
