"""Experiment E2: Deterministic Certifiability Phase Map (Testing the Full-Determinism Hypothesis).

Computes 176-bit Arb sampling-free deterministic brackets (RoughCert-SPX-Det, Theorems 2 & 4a)
across a systematic grid of (H, eta_eff = eta * T^H, rho) to answer quantitatively:
  Where does full deterministic Arb certification achieve desk-grade IV tolerances (<= 25 bps
  or <= 100 bps), and where does roughness of the leverage integral J_T mathematically force
  transition to Certified-Bias / Asymptotic-Envelope RQMC?

Also runs 7 concrete regime spotlights pairing RoughCert-SPX-Det with RoughCert-SPX-Bridge
(ce_i_only, unified compute_bridge_envelope) to prove 100% containment and quantify the exact
width ratio (Blind Audit #1 fixes C1, C2, M4, m1).
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

from src.deterministic_spx import (
    deterministic_spx_bracket_k0_arb,
    deterministic_spx_rho0_atm_twomoment_arb,
)
from src.bridge_rqmc import price_spx_bridge_certified


def _json_num(val: float):
    """Strict RFC-8259 JSON sanitizer: map non-finite values to None."""
    return float(val) if math.isfinite(val) else None


def main():
    t0 = time.perf_counter()
    H_grid = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45]
    eta_eff_grid = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50]
    T_base = 0.25
    xi0_base = 0.04
    K_atm = 1.0

    print("Running E2.1: 176-bit Arb Phase Map across (H, eta*T^H, rho)...")
    phase_cells = []
    for H in H_grid:
        for eta_eff in eta_eff_grid:
            eta = eta_eff / (T_base ** H)
            b_r0_k0 = deterministic_spx_bracket_k0_arb(H=H, T=T_base, eta=eta, rho=0.0, xi0_flat=xi0_base, K=K_atm)
            b_r0_2m = deterministic_spx_rho0_atm_twomoment_arb(H=H, T=T_base, eta=eta, xi0_flat=xi0_base, n_boxes=24)
            b_r5 = deterministic_spx_bracket_k0_arb(H=H, T=T_base, eta=eta, rho=-0.5, xi0_flat=xi0_base, K=K_atm)
            b_r9 = deterministic_spx_bracket_k0_arb(H=H, T=T_base, eta=eta, rho=-0.9, xi0_flat=xi0_base, K=K_atm)
            phase_cells.append({
                "H": H,
                "T": T_base,
                "eta_eff": eta_eff,
                "eta": eta,
                "E_J0_hi": _json_num(b_r0_k0["E_J_hi"]),
                "iv_width_bps_rho0_k0": _json_num(b_r0_k0["iv_width_bps"]),
                "iv_width_bps_rho0_2m": _json_num(b_r0_2m["iv_width_bps"]),
                "iv_width_bps_rho_m05": _json_num(b_r5["iv_width_bps"]),
                "iv_width_bps_rho_m09": _json_num(b_r9["iv_width_bps"]),
                "cert_25bps_rho0_2m": bool(b_r0_2m["iv_width_bps"] <= 25.0),
                "cert_25bps_rho_m09": bool(b_r9["iv_width_bps"] <= 25.0),
            })

    print("Running E2.2: Exact 10/25/50-bps Certifiability Phase Boundaries...")
    boundaries = []
    for H in H_grid:
        row = {"H": H}
        for label, rho_v, use_2m in [("rho0_2m", 0.0, True), ("rho_m05_k0", -0.5, False), ("rho_m09_k0", -0.9, False)]:
            for tol_bps in [10.0, 25.0, 50.0]:
                lo_e, hi_e = 1e-4, 2.5
                for _ in range(18):
                    mid = 0.5 * (lo_e + hi_e)
                    eta_m = mid / (T_base ** H)
                    if use_2m:
                        w = deterministic_spx_rho0_atm_twomoment_arb(H=H, T=T_base, eta=eta_m, xi0_flat=xi0_base, n_boxes=20)["iv_width_bps"]
                    else:
                        w = deterministic_spx_bracket_k0_arb(H=H, T=T_base, eta=eta_m, rho=rho_v, xi0_flat=xi0_base, K=K_atm)["iv_width_bps"]
                    if w <= tol_bps:
                        lo_e = mid
                    else:
                        hi_e = mid
                row[f"eta_eff_star_{label}_{int(tol_bps)}bps"] = lo_e
        boundaries.append(row)
        print(f"  H={H:.2f}: 25bps boundary eta*T^H: rho=0(2m-sharp)={row['eta_eff_star_rho0_2m_25bps']:.4f}, "
              f"rho=-0.5={row['eta_eff_star_rho_m05_k0_25bps']:.4f}, rho=-0.9={row['eta_eff_star_rho_m09_k0_25bps']:.4f}")

    print("Running E2.3: 7 Spotlight Regimes (Arb-Det vs Bridge Containment & Width Ratio)...")
    spotlights_spec = [
        {"id": "S1_mild_uncorr", "desc": "Mild vol-of-vol, rho=0", "H": 0.30, "T": 0.10, "eta": 0.30, "rho": 0.0, "xi0": 0.04},
        {"id": "S2_rough_uncorr_mod", "desc": "Rough H=0.10, moderate eta, rho=0", "H": 0.10, "T": 0.25, "eta": 0.20, "rho": 0.0, "xi0": 0.04},
        {"id": "S3_short_1d_corr", "desc": "Ultra-short 1-day correlated, mild eta", "H": 0.30, "T": 1.0/252.0, "eta": 0.03, "rho": -0.7, "xi0": 0.04},
        {"id": "S4_mild_rough_corr", "desc": "Rough H=0.10, mild eta=0.10", "H": 0.10, "T": 0.25, "eta": 0.10, "rho": -0.7, "xi0": 0.04},
        {"id": "S5_moderate_rough", "desc": "Rough H=0.10, moderate eta=0.50", "H": 0.10, "T": 0.25, "eta": 0.50, "rho": -0.7, "xi0": 0.04},
        {"id": "S6_cal_spx_uncorr", "desc": "Calibrated eta=1.9, rho=0", "H": 0.10, "T": 0.25, "eta": 1.90, "rho": 0.0, "xi0": 0.04},
        {"id": "S7_cal_spx_full", "desc": "Calibrated SPX rBergomi", "H": 0.10, "T": 0.25, "eta": 1.90, "rho": -0.9, "xi0": 0.04},
    ]
    spotlights = []
    for sp in spotlights_spec:
        det_best = deterministic_spx_bracket_k0_arb(sp["H"], sp["T"], sp["eta"], sp["rho"], sp["xi0"], K=1.0)
        if sp["rho"] == 0.0:
            det_2m = deterministic_spx_rho0_atm_twomoment_arb(sp["H"], sp["T"], sp["eta"], sp["xi0"], n_boxes=32)
            if det_2m["iv_width_bps"] < det_best["iv_width_bps"]:
                det_best = det_2m
        xi0_fn = lambda s, val=sp["xi0"]: np.full_like(s, val, dtype=np.float64)
        bridge = price_spx_bridge_certified(
            H=sp["H"], T=sp["T"], eta=sp["eta"], rho=sp["rho"], xi0_fn=xi0_fn,
            strikes=[1.0], levels=(32, 64, 128, 256), method="ce_i_only",
            n_shifts=16, n_half=1024, base_seed=20260200 + len(spotlights),
        )
        p_mid = bridge["finest_price_mean"][0]
        iv_mid = bridge["iv_mid"][0]
        contained = bool(det_best["price_lo"] <= p_mid <= det_best["price_hi"])
        overlaps = bool(max(det_best["price_lo"], bridge["continuum_price_lo"][0]) <= min(det_best["price_hi"], bridge["continuum_price_hi"][0]))
        det_w = det_best["iv_width_bps"]
        br_w = bridge["iv_continuum_width_bps"][0]
        ratio = (det_w / max(br_w, 1e-9)) if math.isfinite(det_w) else None
        assert det_best["iv_lo"] >= 0.0, f"m1 regression: negative det_iv_lo={det_best['iv_lo']}"
        entry = {
            **sp,
            "eta_eff": sp["eta"] * (sp["T"] ** sp["H"]),
            "det_method": det_best["method"],
            "det_iv_lo": _json_num(det_best["iv_lo"]),
            "det_iv_hi": _json_num(det_best["iv_hi"]),
            "det_iv_unbounded": bool(not math.isfinite(det_best["iv_hi"])),
            "det_iv_width_bps": _json_num(det_w),
            "det_price_lo": det_best["price_lo"],
            "det_price_hi": det_best["price_hi"],
            "bridge_iv_mid": iv_mid,
            "bridge_iv_lo": bridge["iv_continuum_lo"][0],
            "bridge_iv_hi": bridge["iv_continuum_hi"][0],
            "bridge_iv_width_bps": br_w,
            "bridge_status": bridge["status_per_strike"][0],
            "det_contains_bridge_mid": contained,
            "enclosures_overlap": overlaps,
            "det_to_bridge_width_ratio": ratio,
            "gate_decision": "Arb-certified (Det)" if det_w <= 25.0 else "Asymptotic-Envelope (Bridge)",
        }
        spotlights.append(entry)
        print(f"  {sp['id']:18s}: Det={det_w:8.2f} bps | Bridge={br_w:6.2f} bps "
              f"| contained={contained} | status={entry['bridge_status']} | gate={entry['gate_decision']}")

    out = {
        "experiment": "E2_det_phase_map",
        "elapsed_s": time.perf_counter() - t0,
        "H_grid": H_grid,
        "eta_eff_grid": eta_eff_grid,
        "T_base": T_base,
        "xi0_base": xi0_base,
        "phase_cells": phase_cells,
        "boundaries": boundaries,
        "spotlights": spotlights,
    }
    out_path = ROOT / "results" / "e2_det_phase_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path} in {out['elapsed_s']:.1f}s.")


if __name__ == "__main__":
    main()
