"""Experiment E5: Weak Convergence Rate & FSW (2025) Phase Transition Audit across H.

Empirically verifies the Friz-Salkeld-Wagenhofer (2025, Ann. Appl. Probab.) weak rate
  beta_FSW(H) = min(3H + 1/2, 1)  (phase transition at H = 1/6)
on correlated rough Bergomi (T=0.25, eta=1.9, rho=-0.9, xi0=0.04) across
  H in {0.05, 0.10, 0.15, 0.25, 0.40}
using EXACT coupled nested paths across levels (16, 32, 64, 128, 256, 512) with 65,536 RQMC paths
and increment-optimal pilot-fitted 6-control zero-mean banks for Left-Point, RoughCert-SPX-CE
(ce_i_only), and RoughCert-SPX-CE-Jensen (ce_adapted ablation).
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

from src.bridge_rqmc import (
    build_volterra_grid_cache,
    generate_exact_paths,
    coarsen_paths,
    sample_rqmc_normals,
    evaluate_leftpoint_paths,
    evaluate_ce_i_only_paths,
    evaluate_ce_adapted_paths,
    _fit_multicv_pilot,
)


def fit_weak_slope(levels: list[int], incr_abs: np.ndarray, incr_se: np.ndarray) -> dict:
    """Weighted log-log fit of |Delta_l| = C * n_l^{-beta} across resolved increments (SNR > 2)."""
    n_c = np.array(levels[:-1], dtype=np.float64)
    mask = (incr_abs > 2.0 * incr_se) & (incr_abs > 1e-12)
    if np.sum(mask) < 2:
        return {"beta_hat": None, "n_resolved": int(np.sum(mask)), "log_intercept": None}
    x = np.log2(n_c[mask])
    y = np.log2(incr_abs[mask])
    w = (incr_abs[mask] / np.maximum(incr_se[mask], 1e-15)) ** 2
    w = np.minimum(w, 1e6)
    coeffs = np.polyfit(x, y, deg=1, w=np.sqrt(w))
    return {"beta_hat": float(-coeffs[0]), "n_resolved": int(np.sum(mask)), "log_intercept": float(coeffs[1])}


def main():
    t0 = time.perf_counter()
    H_list = [0.05, 0.10, 0.15, 0.25, 0.40]
    T, eta, rho, xi0 = 0.25, 1.9, -0.9, 0.04
    strikes = np.array([0.95, 1.00, 1.06], dtype=np.float64)
    K_len = len(strikes)
    xi0_fn = lambda s: np.full_like(s, xi0, dtype=np.float64)
    levels = [16, 32, 64, 128, 256, 512]
    n_max = levels[-1]
    L = len(levels)
    n_shifts, n_half = 16, 2048  # 65,536 paths
    n_pilot_shifts = 2
    M = 2 * n_half

    results_by_H = []
    for h_idx, H in enumerate(H_list):
        t_h = time.perf_counter()
        print(f"\nRunning E5 for H = {H:.2f} (beta_FSW theory = {min(3*H+0.5, 1.0):.3f})...")
        caches = {n_l: build_volterra_grid_cache(H, T, n_l, need_ce=True) for n_l in levels}
        methods = ("leftpoint", "ce_i_only", "ce_adapted")

        def _eval_all(n_l: int, dW: np.ndarray, Y: np.ndarray) -> dict[str, dict]:
            return {
                "leftpoint": evaluate_leftpoint_paths(dW, Y, H, T, eta, rho, xi0_fn, strikes, build_cv_bank=True),
                "ce_i_only": evaluate_ce_i_only_paths(caches[n_l], dW, Y, eta, rho, xi0_fn, strikes, build_cv_bank=True),
                "ce_adapted": evaluate_ce_adapted_paths(caches[n_l], dW, Y, eta, rho, xi0_fn, strikes, build_cv_bank=True),
            }

        # 1. Fit per-level and increment-optimal 6-control weights strictly on independent pilot shifts
        p_C = {m: [[] for _ in range(L)] for m in methods}
        p_X = {m: [[] for _ in range(L)] for m in methods}
        pilot_half = max(512, n_half // 2)
        for p_idx in range(n_pilot_shifts):
            Zp = sample_rqmc_normals(2 * n_max, n_half=pilot_half, seed=20260700 + 100 * h_idx + 9000 + p_idx, antithetic=True)
            dW_f, Y_f = generate_exact_paths(caches[n_max], Zp)
            for idx_l, n_l in enumerate(levels):
                stride = n_max // n_l
                dW, Y = (dW_f, Y_f) if stride == 1 else coarsen_paths(dW_f, Y_f, stride=stride)
                res_m = _eval_all(n_l, dW, Y)
                for m in methods:
                    p_C[m][idx_l].append(res_m[m]["C_paths"])
                    p_X[m][idx_l].append(res_m[m]["X_cv"])

        W_lvl = {}
        W_inc = {}
        for m in methods:
            c_st = [np.vstack(p_C[m][l]) for l in range(L)]
            x_st = [np.vstack(p_X[m][l]) for l in range(L)]
            W_lvl[m] = [_fit_multicv_pilot(c_st[l], x_st[l]) for l in range(L)]
            W_inc[m] = [_fit_multicv_pilot(c_st[l + 1] - c_st[l], x_st[l + 1] - x_st[l]) for l in range(L - 1)]

        # 2. Evaluate R strictly independent RQMC shifts across all nested levels
        sh_lvl = {m: np.empty((L, n_shifts, K_len)) for m in methods}
        sh_inc = {m: np.empty((L - 1, n_shifts, K_len)) for m in methods}

        for r in range(n_shifts):
            Z = sample_rqmc_normals(2 * n_max, n_half=n_half, seed=20260700 + 100 * h_idx + r, antithetic=True)
            dW_f, Y_f = generate_exact_paths(caches[n_max], Z)
            c_bars = {m: [] for m in methods}
            x_bars = {m: [] for m in methods}
            for idx_l, n_l in enumerate(levels):
                stride = n_max // n_l
                dW, Y = (dW_f, Y_f) if stride == 1 else coarsen_paths(dW_f, Y_f, stride=stride)
                res_m = _eval_all(n_l, dW, Y)
                for m in methods:
                    cb = res_m[m]["C_paths"].mean(axis=0)
                    xb = res_m[m]["X_cv"].mean(axis=0)
                    c_bars[m].append(cb)
                    x_bars[m].append(xb)
                    sh_lvl[m][idx_l, r] = cb - np.sum(xb * W_lvl[m][idx_l], axis=1)
            for m in methods:
                for idx_l in range(L - 1):
                    dcb = c_bars[m][idx_l + 1] - c_bars[m][idx_l]
                    dxb = x_bars[m][idx_l + 1] - x_bars[m][idx_l]
                    sh_inc[m][idx_l, r] = dcb - np.sum(dxb * W_inc[m][idx_l], axis=1)

        m_out = {}
        for m_name in methods:
            incr = sh_inc[m_name]  # (L-1, R, K)
            incr_m = incr.mean(axis=1)
            incr_se = incr.std(axis=1, ddof=1) / math.sqrt(n_shifts)
            slopes = [fit_weak_slope(levels, np.abs(incr_m[:, k]), incr_se[:, k]) for k in range(K_len)]

            # Shift-wise equal-weighted strip mean increment with EXACT Student-t SE (m4 fix)
            strip_incr = incr.mean(axis=2)  # (L-1, R)
            strip_mean = strip_incr.mean(axis=1)
            strip_se = strip_incr.std(axis=1, ddof=1) / math.sqrt(n_shifts)
            strip_slope = fit_weak_slope(levels, np.abs(strip_mean), strip_se)

            # Highlight K=1.06 (index 2, strongest monotone call-wing weak bias) AND ATM (index 1)
            m_out[m_name] = {
                "level_means": sh_lvl[m_name].mean(axis=1).tolist(),
                "incr_means": incr_m.tolist(),
                "incr_ses": incr_se.tolist(),
                "slopes_per_strike": slopes,
                "atm_slope": slopes[1],
                "call106_slope": slopes[2],
                "strip_mean_incr": strip_mean.tolist(),
                "strip_mean_se": strip_se.tolist(),
                "strip_mean_slope": strip_slope,
            }
            print(f"  {m_name:12s}: K=1.06 beta={slopes[2]['beta_hat']}, ATM beta={slopes[1]['beta_hat']}, "
                  f"strip={strip_slope['beta_hat']} | K=1.06 |16->32|={abs(incr_m[0,2]):.2e}, |256->512|={abs(incr_m[-1,2]):.2e}")

        results_by_H.append({
            "H": H,
            "beta_fsw_theory": min(3.0 * H + 0.5, 1.0),
            "elapsed_s": time.perf_counter() - t_h,
            "methods": m_out,
        })

    out = {
        "experiment": "E5_weak_rates",
        "elapsed_s": time.perf_counter() - t0,
        "params": {"T": T, "eta": eta, "rho": rho, "xi0": xi0, "strikes": strikes.tolist(), "levels": levels, "total_paths": n_shifts * M},
        "results_by_H": results_by_H,
    }
    out_path = ROOT / "results" / "e5_weak_rates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} in {out['elapsed_s']:.1f}s.")


if __name__ == "__main__":
    main()
