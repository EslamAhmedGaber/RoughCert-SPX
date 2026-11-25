"""Experiment E3: 176-bit Arb Kernel-Approximation Audit (Markovian Bayer-Breneis & BLP Hybrid).

Implements the Three-Way Error Splitting audit (Theorems 1 & 2), incorporating all Blind Audit #1
remediations (C1: ce_i_only reference; C3: unclipped L-BFGS-B Bayer-Breneis; M2: rigorous
root-splitting 176-bit Arb L^1 enclosure; m6: outward-propagated BLP IV gap across [ref_lo, ref_hi]):
  1. Bayer-Breneis (2023) M-factor Markovian rBergomi across M in {2, 4, 6, 8, 12, 16}
     and H in {0.07, 0.10, 0.30} at standard SPX parameters (T=0.25, eta=1.9, rho=-0.9, xi0=0.04).
  2. BLP Hybrid Kernel (kappa in {1, 2, 3, 4, 6, 8}) across n in {64, 128, 256}:
     Exact 176-bit Arb continuous-time kernel gap E_J(K_{kappa,n}) and Theorem 2 price/IV bound.
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
from flint import arb

from src.bs_arb import (
    arb_hi_float,
    arb_lo_float,
    bs_iv_interval_from_price_bounds,
    atm_skew_interval_arb,
)
from src.volterra_arb import markov_kernel_l1_arb
from src.deterministic_spx import (
    certify_kernel_gap_markov_arb,
    certify_kernel_gap_blp_hybrid_arb,
)
from src.competitors import bayer_breneis_l2_nodes_weights, price_bayer_breneis_markovian
from src.bridge_rqmc import price_spx_bridge_certified


def main():
    t0 = time.perf_counter()
    T = 0.25
    eta = 1.9
    rho = -0.9
    xi0 = 0.04
    xi0_fn = lambda s: np.full_like(s, xi0, dtype=np.float64)
    dk = 0.02
    strikes = [0.85, math.exp(-dk), 1.0, math.exp(dk), 1.12]

    # 1. Exact-kernel reference at H = 0.10 via RoughCert-SPX-Bridge (ce_i_only, 131,072 paths)
    print("Running E3.0: Exact-kernel RoughCert-SPX-Bridge (ce_i_only, 131,072 paths) reference for H=0.10...")
    ref_H010 = price_spx_bridge_certified(
        H=0.10, T=T, eta=eta, rho=rho, xi0_fn=xi0_fn, strikes=strikes,
        levels=(64, 128, 256, 512), method="ce_i_only",
        n_shifts=16, n_half=4096, base_seed=20260300,
    )
    ref_skew_lo, ref_skew_hi = atm_skew_interval_arb(
        (ref_H010["iv_continuum_lo"][1], ref_H010["iv_continuum_hi"][1]),
        (ref_H010["iv_continuum_lo"][3], ref_H010["iv_continuum_hi"][3]),
        dk=dk,
    )
    print(f"  Ref H=0.10 ATM IV=[{100*ref_H010['iv_continuum_lo'][2]:.3f}%, {100*ref_H010['iv_continuum_hi'][2]:.3f}%] "
          f"(width={ref_H010['iv_continuum_width_bps'][2]:.2f} bps), ATM Skew=[{ref_skew_lo:.4f}, {ref_skew_hi:.4f}]")
    print(f"  Ref statuses: {ref_H010['status_per_strike']}")

    # 2. Markovian Bayer-Breneis Kernel Audit (unclipped L-BFGS-B + rigorous Arb L^1)
    print("Running E3.1: 176-bit Arb Markovian Kernel Audit (Bayer-Breneis 2023, unclipped L-BFGS-B)...")
    markov_rows = []
    for H in [0.07, 0.10, 0.30]:
        for M_fac in [2, 4, 6, 8, 12, 16]:
            t_fit = time.perf_counter()
            nodes, weights, l2_fit = bayer_breneis_l2_nodes_weights(H=H, M_factors=M_fac, T=T)
            bb_fit_s = time.perf_counter() - t_fit
            cert = certify_kernel_gap_markov_arb(
                H=H, T=T, eta=eta, rho=rho, xi0_flat=xi0, K=1.0,
                nodes=nodes, weights=weights, n_boxes=64,
            )
            l1_arb = markov_kernel_l1_arb(
                arb(str(H)), arb(str(T)),
                [arb(str(float(x))) for x in nodes],
                [arb(str(float(w))) for w in weights],
                n_scan=1200,
            )
            row = {
                "H": H,
                "M_factors": M_fac,
                "bb_fit_s": bb_fit_s,
                "kernel_L2_lo": cert["kernel_L2_err_lo"],
                "kernel_L2_hi": cert["kernel_L2_err_hi"],
                "kernel_L1_lo": arb_lo_float(l1_arb),
                "kernel_L1_hi": arb_hi_float(l1_arb),
                "E_J_lo": cert["E_J_lo"],
                "E_J_hi": cert["E_J_hi"],
                "arb_price_gap_upper_atm": cert["price_gap_upper"],
                "nodes_min": float(nodes.min()),
                "nodes_max": float(nodes.max()),
            }
            if abs(H - 0.10) < 1e-12:
                mk = price_bayer_breneis_markovian(
                    H=H, T=T, eta=eta, rho=rho, xi0_fn=xi0_fn, strikes=strikes,
                    M_factors=M_fac, n=256, n_batches=16, n_half=1024,
                    base_seed=20260350 + M_fac, nodes_weights=(nodes, weights, l2_fit),
                )
                skew_mk = (mk["iv_mid"][3] - mk["iv_mid"][1]) / (2.0 * dk)
                iv_err_intervals = []
                for idx_k in range(len(strikes)):
                    v = mk["iv_mid"][idx_k]
                    lo_r, hi_r = ref_H010["iv_continuum_lo"][idx_k], ref_H010["iv_continuum_hi"][idx_k]
                    d_min = 0.0 if lo_r <= v <= hi_r else min(abs(v - lo_r), abs(v - hi_r))
                    d_max = max(abs(v - lo_r), abs(v - hi_r))
                    iv_err_intervals.append([1e4 * d_min, 1e4 * d_max])
                skew_err_min = 0.0 if ref_skew_lo <= skew_mk <= ref_skew_hi else min(abs(skew_mk - ref_skew_lo), abs(skew_mk - ref_skew_hi))
                skew_err_max = max(abs(skew_mk - ref_skew_lo), abs(skew_mk - ref_skew_hi))

                row.update({
                    "mk_ivs": mk["iv_mid"],
                    "mk_atm_skew": skew_mk,
                    "iv_err_intervals_bps": iv_err_intervals,
                    "max_iv_err_lo_bps": max(x[0] for x in iv_err_intervals),
                    "max_iv_err_hi_bps": max(x[1] for x in iv_err_intervals),
                    "atm_iv_err_interval_bps": iv_err_intervals[2],
                    "skew_err_interval": [skew_err_min, skew_err_max],
                    "mk_setup_s": bb_fit_s,
                    "mk_solve_s": mk["solve_s"],
                })
                print(f"  Markov H=0.10 M={M_fac:2d}: L2={cert['kernel_L2_err_hi']:.5f}, L1={arb_hi_float(l1_arb):.5f}, "
                      f"fit={bb_fit_s:.2f}s, ArbGap={cert['price_gap_upper']:.5f}, "
                      f"actual_ATM=[{iv_err_intervals[2][0]:.1f},{iv_err_intervals[2][1]:.1f}]bps, "
                      f"max=[{row['max_iv_err_lo_bps']:.1f},{row['max_iv_err_hi_bps']:.1f}]bps, skew={skew_mk:.4f}")
            else:
                print(f"  Markov H={H:.2f} M={M_fac:2d}: L2={cert['kernel_L2_err_hi']:.5f}, L1={arb_hi_float(l1_arb):.5f}, "
                      f"fit={bb_fit_s:.2f}s, E_J={cert['E_J_hi']:.5f}, ArbGap={cert['price_gap_upper']:.5f}")
            markov_rows.append(row)

    # 3. BLP Hybrid Kernel 176-bit Arb Audit (m6: outward-propagated across [ref_p_lo, ref_p_hi])
    print("Running E3.2: 176-bit Arb BLP Hybrid Continuous-Time Kernel Audit...")
    blp_rows = []
    ref_p_lo = ref_H010["continuum_price_lo"][2]
    ref_p_hi = ref_H010["continuum_price_hi"][2]
    ref_iv_lo = ref_H010["iv_continuum_lo"][2]
    ref_iv_hi = ref_H010["iv_continuum_hi"][2]
    for H in [0.07, 0.10, 0.30]:
        for n_steps in [64, 128, 256]:
            for kappa in [1, 2, 3, 4, 6, 8]:
                cert = certify_kernel_gap_blp_hybrid_arb(
                    H=H, T=T, eta=eta, rho=rho, xi0_flat=xi0, K=1.0, n=n_steps, kappa=kappa,
                )
                iv_gap_upper_bps = None
                iv_total_enclosure_bps = None
                if abs(H - 0.10) < 1e-12:
                    g = cert["price_gap_upper"]
                    # Outward propagation across the full reference ATM price enclosure [ref_p_lo, ref_p_hi]
                    il_prop, ih_prop = bs_iv_interval_from_price_bounds(
                        max(ref_p_lo - g, 1e-12), min(ref_p_hi + g, 1.0), 1.0, T
                    )
                    # Worst-case marginal IV shift attributable purely to continuous-time kernel gap g:
                    # max(ref_iv_lo - il_prop, ih_prop - ref_iv_hi)
                    iv_gap_upper_bps = (
                        1e4 * max(ref_iv_lo - il_prop, ih_prop - ref_iv_hi)
                        if math.isfinite(ih_prop) else None
                    )
                    iv_total_enclosure_bps = 1e4 * (ih_prop - il_prop) if math.isfinite(ih_prop) else None
                blp_rows.append({
                    "H": H,
                    "n": n_steps,
                    "kappa": kappa,
                    "kernel_scope": "continuous-time BLP kernel gap only (excludes discrete time-stepping)",
                    "E_J_lo": cert["E_J_lo"],
                    "E_J_hi": cert["E_J_hi"],
                    "arb_price_gap_upper_atm": cert["price_gap_upper"],
                    "arb_iv_marginal_gap_upper_bps_atm": iv_gap_upper_bps,
                    "arb_iv_propagated_total_width_bps_atm": iv_total_enclosure_bps,
                })
                if abs(H - 0.10) < 1e-12 and n_steps == 256:
                    print(f"  BLP Hybrid H=0.10 n=256 kappa={kappa}: E_J={cert['E_J_hi']:.6f}, "
                          f"ArbPriceGap={cert['price_gap_upper']:.6f}, ArbIV_gap_upper={iv_gap_upper_bps:.2f} bps")

    out = {
        "experiment": "E3_kernel_audit",
        "elapsed_s": time.perf_counter() - t0,
        "params": {"T": T, "eta": eta, "rho": rho, "xi0": xi0, "strikes": strikes, "dk": dk},
        "ref_H010_bridge": ref_H010,
        "ref_H010_skew_interval": [ref_skew_lo, ref_skew_hi],
        "markov_audit": markov_rows,
        "blp_hybrid_audit": blp_rows,
    }
    out_path = ROOT / "results" / "e3_kernel_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path} in {out['elapsed_s']:.1f}s.")


if __name__ == "__main__":
    main()
