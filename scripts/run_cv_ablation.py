"""Reproducible control-variate variance-reduction ablation for RoughCert-SPX-Bridge.

Measures the exact variance-reduction factor of the pilot-fitted six-control bank
(three multi-budget discrete stopped-timer option controls + F_T - 1 + I_left - E[I_left] + J_left)
versus raw (uncontrolled) payoffs on Regime A (H=0.10, T=0.25, eta=1.9, rho=-0.9, xi0=0.04)
for both single-level payoffs (n=128) and coupled dyadic increments (n=64 -> 128)
across R=16 scrambled Sobol (LMS + digital shift) shifts (32,768 paths + R_pilot=2 independent pilot shifts).

Saves verified statistics to results/cv_ablation.json for direct macro injection into spx_numbers.tex.
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
    sample_rqmc_normals,
    generate_exact_paths,
    coarsen_paths,
    evaluate_ce_i_only_paths,
    _fit_multicv_pilot,
)


def main() -> None:
    t0 = time.perf_counter()
    H, T, eta, rho, xi0 = 0.10, 0.25, 1.9, -0.9, 0.04
    dk = 0.02
    strikes = np.array([0.85, 0.90, 0.95, math.exp(-dk), 1.00, math.exp(dk), 1.06, 1.12], dtype=np.float64)
    xi0_fn = lambda s: np.full_like(s, xi0, dtype=np.float64)
    n_shifts = 16
    n_half = 1024
    n_pilot = 2
    base_seed = 20260405
    n_K = len(strikes)

    cache_64 = build_volterra_grid_cache(H, T, 64, need_ce=True)
    cache_128 = build_volterra_grid_cache(H, T, 128, need_ce=True)

    # Strictly independent pilot shifts (matching price_spx_bridge_certified)
    C64_p, X64_p, C128_p, X128_p = [], [], [], []
    for p_idx in range(n_pilot):
        Zp = sample_rqmc_normals(dim=2 * 128, n_half=max(256, n_half // 2), seed=base_seed + 900000 + p_idx, antithetic=True)
        dW_f, Y_f = generate_exact_paths(cache_128, Zp)
        dW_c, Y_c = coarsen_paths(dW_f, Y_f, stride=2)
        r_f = evaluate_ce_i_only_paths(cache_128, dW_f, Y_f, eta, rho, xi0_fn, strikes, build_cv_bank=True)
        r_c = evaluate_ce_i_only_paths(cache_64, dW_c, Y_c, eta, rho, xi0_fn, strikes, build_cv_bank=True)
        C128_p.append(r_f["C_paths"]); X128_p.append(r_f["X_cv"])
        C64_p.append(r_c["C_paths"]); X64_p.append(r_c["X_cv"])

    C128_p_mat = np.vstack(C128_p); X128_p_mat = np.vstack(X128_p)
    C64_p_mat = np.vstack(C64_p); X64_p_mat = np.vstack(X64_p)
    W_lvl = _fit_multicv_pilot(C128_p_mat, X128_p_mat)
    W_inc = _fit_multicv_pilot(C128_p_mat - C64_p_mat, X128_p_mat - X64_p_mat)

    raw_lvl = np.empty((n_shifts, n_K), dtype=np.float64)
    cv_lvl = np.empty((n_shifts, n_K), dtype=np.float64)
    raw_inc = np.empty((n_shifts, n_K), dtype=np.float64)
    cv_inc = np.empty((n_shifts, n_K), dtype=np.float64)

    for r in range(n_shifts):
        Z = sample_rqmc_normals(dim=2 * 128, n_half=n_half, seed=base_seed + r, antithetic=True)
        dW_f, Y_f = generate_exact_paths(cache_128, Z)
        dW_c, Y_c = coarsen_paths(dW_f, Y_f, stride=2)
        r_f = evaluate_ce_i_only_paths(cache_128, dW_f, Y_f, eta, rho, xi0_fn, strikes, build_cv_bank=True)
        r_c = evaluate_ce_i_only_paths(cache_64, dW_c, Y_c, eta, rho, xi0_fn, strikes, build_cv_bank=True)

        cf_bar = r_f["C_paths"].mean(axis=0)
        xf_bar = r_f["X_cv"].mean(axis=0)
        cc_bar = r_c["C_paths"].mean(axis=0)
        xc_bar = r_c["X_cv"].mean(axis=0)

        raw_lvl[r] = cf_bar
        cv_lvl[r] = cf_bar - np.sum(xf_bar * W_lvl, axis=1)
        raw_inc[r] = cf_bar - cc_bar
        cv_inc[r] = (cf_bar - cc_bar) - np.sum((xf_bar - xc_bar) * W_inc, axis=1)

    vrf_lvl = (raw_lvl.var(axis=0, ddof=1) / cv_lvl.var(axis=0, ddof=1)).tolist()
    vrf_inc = (raw_inc.var(axis=0, ddof=1) / cv_inc.var(axis=0, ddof=1)).tolist()

    out = {
        "experiment": "CV_ablation_RegimeA_64_128",
        "elapsed_s": time.perf_counter() - t0,
        "params": {
            "H": H, "T": T, "eta": eta, "rho": rho, "xi0": xi0,
            "levels": [64, 128], "n_shifts": n_shifts, "n_half": n_half,
            "n_pilot": n_pilot, "base_seed": base_seed,
            "strikes": strikes.tolist(),
        },
        "level_n128_var_reduction_factor": vrf_lvl,
        "increment_64_128_var_reduction_factor": vrf_inc,
        "level_vrf_min": float(min(vrf_lvl)),
        "level_vrf_max": float(max(vrf_lvl)),
        "level_vrf_atm": float(vrf_lvl[4]),
        "incr_vrf_min": float(min(vrf_inc)),
        "incr_vrf_max": float(max(vrf_inc)),
        "incr_vrf_atm": float(vrf_inc[4]),
    }
    out_path = ROOT / "results" / "cv_ablation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Level n=128 VRF: min={out['level_vrf_min']:.2f}x, ATM={out['level_vrf_atm']:.2f}x, max={out['level_vrf_max']:.2f}x", flush=True)
    print(f"Incr 64->128 VRF: min={out['incr_vrf_min']:.2f}x, ATM={out['incr_vrf_atm']:.2f}x, max={out['incr_vrf_max']:.2f}x", flush=True)
    print(f"Saved {out_path} in {out['elapsed_s']:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
