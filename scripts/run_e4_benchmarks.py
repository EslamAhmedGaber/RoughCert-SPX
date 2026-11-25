"""Experiment E4: Head-to-Head SPX Benchmarks, Smiles, ATM Skew, and Work-Precision Curves.

Compares ALL 6 published rough-Bergomi SPX pricing families head-to-head against our continuum
enclosure (RoughCert-SPX-Bridge, ce_i_only) on the same machine, single-threaded
(OMP/MKL/OPENBLAS_NUM_THREADS=1), with separate cold (setup) and amortised (solve) timings,
incorporating ALL Blind Audit #1 remediations (C1, C2, C3, M1, m3, m5):
  1. McCrickerd & Pakkanen (2018) -- Turbocharged BLP Hybrid (kappa = 1, 2) [MC]
  2. Bayer, Ben Hammouda & Tempone (2020) -- Hierarchical Bridge-RQMC (+ Richardson diagnostic) [RQMC]
  3. Bayer & Breneis (2023) -- Unclipped L-BFGS-B Markovian Multifactor (M = 4, 8, 12, 16) [MC]
  4. Horvath, Jacquier & Muguruza (2024) -- rDonsker: BOTH variance-matched (default) AND b_k* [MC]
  5. Bergomi & Guyon (2012) -- Order-2 Closed-Form IV & Skew Expansion [Analytical]
  6. RoughCert-SPX-CE (ce_i_only: Channel-I exact conditional expectation + energy-conserving J_T) [RQMC]
  7. RoughCert-SPX-CE-Jensen (ce_adapted ablation documenting the Theta(n^-2H) Jensen-J pitfall) [RQMC]
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.bs_arb import atm_skew_interval_arb
from src.bridge_rqmc import price_spx_bridge_certified, price_spx_single_level
from src.competitors import (
    price_mp2018_hybrid,
    price_bht2020_bridge_rqmc,
    price_bayer_breneis_markovian,
    price_hjm2021_rdonsker,
    bergomi_guyon_order2_iv,
)


def err_interval_vs_enclosure(val: float, lo: float, hi: float) -> list[float]:
    """Rigorous error interval [dist(val, [lo, hi]), max(|val - lo|, |val - hi|)]."""
    d_min = 0.0 if lo <= val <= hi else min(abs(val - lo), abs(val - hi))
    d_max = max(abs(val - lo), abs(val - hi))
    return [d_min, d_max]


def audit_method_against_ref(
    res: dict,
    ref: dict,
    ref_skew: tuple[float, float],
    idx_minus: int,
    idx_atm: int,
    idx_plus: int,
    dk: float,
    sampling_label: str = "unspecified",
) -> dict:
    """Compute rigorous IV error intervals (bps), ATM skew error interval, sampling label & SEs."""
    ivs = res["iv_mid"]
    iv_errs_bps = []
    for k_i, v in enumerate(ivs):
        el, eh = err_interval_vs_enclosure(v, ref["iv_continuum_lo"][k_i], ref["iv_continuum_hi"][k_i])
        iv_errs_bps.append([1e4 * el, 1e4 * eh])

    skew_val = (ivs[idx_plus] - ivs[idx_minus]) / (2.0 * dk)
    skew_err = err_interval_vs_enclosure(skew_val, ref_skew[0], ref_skew[1])
    return {
        "method": res["method"],
        "status": res["status"],
        "sampling": res.get("sampling", sampling_label),
        "n": res.get("n"),
        "cold_setup_s": res.get("cold_setup_s", 0.0),
        "solve_s": res.get("solve_s", 0.0),
        "iv_mid": ivs,
        "iv_ci_lo": res.get("iv_ci_lo"),
        "iv_ci_hi": res.get("iv_ci_hi"),
        "price_se": res.get("price_se"),
        "iv_err_intervals_bps": iv_errs_bps,
        "atm_iv_err_bps": iv_errs_bps[idx_atm],
        "max_smile_err_lo_bps": max(x[0] for x in iv_errs_bps),
        "max_smile_err_hi_bps": max(x[1] for x in iv_errs_bps),
        "atm_skew": skew_val,
        "atm_skew_err_interval": skew_err,
        "martingale_check_mean_F": res.get("martingale_check_mean_F"),
        "terminal_var_ratio": res.get("terminal_var_ratio"),
    }


def run_regime_A():
    print("\n=== Running E4 Regime A (Standard SPX: H=0.10, T=0.25, eta=1.9, rho=-0.9, xi0=0.04) ===")
    H, T, eta, rho, xi0 = 0.10, 0.25, 1.9, -0.9, 0.04
    dk = 0.02
    strikes = [0.85, 0.90, 0.95, math.exp(-dk), 1.00, math.exp(dk), 1.06, 1.12]
    idx_m, idx_atm, idx_p = 3, 4, 5
    xi0_fn = lambda s: np.full_like(s, xi0, dtype=np.float64)

    print("  Computing RoughCert-SPX-Bridge (ce_i_only) continuum reference (levels 64,128,256,512; 131,072 paths)...")
    ref = price_spx_bridge_certified(
        H=H, T=T, eta=eta, rho=rho, xi0_fn=xi0_fn, strikes=strikes,
        levels=(64, 128, 256, 512), method="ce_i_only",
        n_shifts=16, n_half=4096, base_seed=20260400,
    )
    ref_skew = atm_skew_interval_arb(
        (ref["iv_continuum_lo"][idx_m], ref["iv_continuum_hi"][idx_m]),
        (ref["iv_continuum_lo"][idx_p], ref["iv_continuum_hi"][idx_p]),
        dk=dk,
    )
    print(f"  Ref ATM IV: [{100*ref['iv_continuum_lo'][idx_atm]:.3f}%, {100*ref['iv_continuum_hi'][idx_atm]:.3f}%] "
          f"(width={ref['iv_continuum_width_bps'][idx_atm]:.2f} bps, samp={ref['iv_sampling_width_bps'][idx_atm]:.2f} bps)")
    print(f"  Ref ATM Skew: [{ref_skew[0]:.4f}, {ref_skew[1]:.4f}], statuses={ref['status_per_strike']}")

    entries = []
    n_shifts, n_half = 16, 1024  # 32,768 paths

    # 1. Bergomi-Guyon Order-2 closed-form expansion
    bg = bergomi_guyon_order2_iv(H, T, eta, rho, xi0, strikes)
    entries.append(audit_method_against_ref(bg, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Analytical O(eta^2)"))
    print(f"  BG2012-O2: max_err=[{entries[-1]['max_smile_err_lo_bps']:.1f}, {entries[-1]['max_smile_err_hi_bps']:.1f}] bps, "
          f"skew={entries[-1]['atm_skew']:.4f}")

    # 2. Grid-based methods across n in {32, 64, 128, 256}
    for n in [32, 64, 128, 256]:
        # RoughCert-SPX-CE (primary FSW-rate Channel-I CE pricer)
        r_ce = price_spx_single_level(H, T, eta, rho, xi0_fn, strikes, n=n, method="ce_i_only", n_shifts=n_shifts, n_half=n_half, base_seed=20260410 + n)
        r_ce["method"] = f"RoughCert-SPX-CE(n={n})"
        entries.append(audit_method_against_ref(r_ce, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "scrambled-Sobol RQMC"))

        # RoughCert-SPX-CE-Jensen (explicit Jensen-J smoothing ablation)
        r_cej = price_spx_single_level(H, T, eta, rho, xi0_fn, strikes, n=n, method="ce_adapted", n_shifts=n_shifts, n_half=n_half, base_seed=20260410 + n)
        r_cej["method"] = f"RoughCert-SPX-CE-Jensen(n={n})"
        entries.append(audit_method_against_ref(r_cej, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "scrambled-Sobol RQMC"))

        # Exact-Grid LeftPoint / BHT2020 Bridge-RQMC
        r_bht = price_bht2020_bridge_rqmc(H, T, eta, rho, xi0_fn, strikes, n=n, n_shifts=n_shifts, n_half=n_half, base_seed=20260420 + n, richardson=(n == 128))
        entries.append(audit_method_against_ref(r_bht, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "scrambled-Sobol RQMC"))
        if n == 128 and "richardson_diagnostic_iv" in r_bht:
            r_rich_wrap = {
                "method": "BHT2020-Richardson(64,128)",
                "status": "diagnostic",
                "sampling": "scrambled-Sobol RQMC (extrapolated)",
                "n": 128,
                "cold_setup_s": r_bht["cold_setup_s"],
                "solve_s": r_bht["richardson_total_solve_s"],
                "iv_mid": r_bht["richardson_diagnostic_iv"],
            }
            entries.append(audit_method_against_ref(r_rich_wrap, ref, ref_skew, idx_m, idx_atm, idx_p, dk))

        # MP2018 Turbocharged Hybrid (kappa=1 and kappa=2)
        for kap in (1, 2):
            r_mp = price_mp2018_hybrid(H, T, eta, rho, xi0_fn, strikes, n=n, kappa=kap, n_batches=n_shifts, n_half=n_half, base_seed=20260430 + 10 * kap + n)
            r_mp["method"] = f"MP2018-Hybrid(k={kap},n={n})"
            entries.append(audit_method_against_ref(r_mp, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Pseudo-random MC"))

        # HJM2024 rDonsker: BOTH variance-matched (default) and b_k* ablation
        r_rd_vm = price_hjm2021_rdonsker(H, T, eta, rho, xi0_fn, strikes, n=n, n_batches=n_shifts, n_half=n_half, base_seed=20260460 + n, variant="var_matched")
        r_rd_vm["method"] = f"HJM2024-rDonsker(vm,n={n})"
        entries.append(audit_method_against_ref(r_rd_vm, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Pseudo-random MC"))

        r_rd_bs = price_hjm2021_rdonsker(H, T, eta, rho, xi0_fn, strikes, n=n, n_batches=n_shifts, n_half=n_half, base_seed=20260460 + n, variant="b_star")
        r_rd_bs["method"] = f"HJM2024-rDonsker(b*,n={n})"
        entries.append(audit_method_against_ref(r_rd_bs, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Pseudo-random MC"))

        print(f"  n={n:3d}: CE=[{entries[-7 if n!=128 else -8]['max_smile_err_lo_bps']:.1f},{entries[-7 if n!=128 else -8]['max_smile_err_hi_bps']:.1f}] | "
              f"CE-Jen=[{entries[-6 if n!=128 else -7]['max_smile_err_lo_bps']:.1f},{entries[-6 if n!=128 else -7]['max_smile_err_hi_bps']:.1f}] | "
              f"MP(k=2)=[{entries[-3]['max_smile_err_lo_bps']:.1f},{entries[-3]['max_smile_err_hi_bps']:.1f}] | "
              f"rD(vm)=[{entries[-2]['max_smile_err_lo_bps']:.1f},{entries[-2]['max_smile_err_hi_bps']:.1f}] | "
              f"rD(b*)=[{entries[-1]['max_smile_err_lo_bps']:.1f},{entries[-1]['max_smile_err_hi_bps']:.1f}] bps")

    # 3. Bayer-Breneis (2023) Unclipped L-BFGS-B Markovian Multifactor (M = 4, 8, 12, 16 at n=128, plus M=8,12 at n=256)
    for M_fac in [4, 8, 12, 16]:
        r_bb = price_bayer_breneis_markovian(H, T, eta, rho, xi0_fn, strikes, M_factors=M_fac, n=128, n_batches=n_shifts, n_half=n_half, base_seed=20260480 + M_fac)
        r_bb["method"] = f"BayerBreneis2023(M={M_fac},n=128)"
        entries.append(audit_method_against_ref(r_bb, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Pseudo-random MC"))
        print(f"  BB2023(M={M_fac:2d},n=128): max=[{entries[-1]['max_smile_err_lo_bps']:.1f},{entries[-1]['max_smile_err_hi_bps']:.1f}]bps, "
              f"setup={r_bb['cold_setup_s']:.2f}s, solve={r_bb['solve_s']:.2f}s")
    for M_fac in [8, 12]:
        r_bb = price_bayer_breneis_markovian(H, T, eta, rho, xi0_fn, strikes, M_factors=M_fac, n=256, n_batches=n_shifts, n_half=n_half, base_seed=20260490 + M_fac)
        r_bb["method"] = f"BayerBreneis2023(M={M_fac},n=256)"
        entries.append(audit_method_against_ref(r_bb, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Pseudo-random MC"))
        print(f"  BB2023(M={M_fac:2d},n=256): max=[{entries[-1]['max_smile_err_lo_bps']:.1f},{entries[-1]['max_smile_err_hi_bps']:.1f}]bps, "
              f"setup={r_bb['cold_setup_s']:.2f}s, solve={r_bb['solve_s']:.2f}s")

    return {
        "name": "Regime_A_Standard_SPX",
        "params": {"H": H, "T": T, "eta": eta, "rho": rho, "xi0": xi0, "dk": dk, "strikes": strikes},
        "ref_bridge": ref,
        "ref_skew_interval": list(ref_skew),
        "entries": entries,
    }


def run_regime_B():
    print("\n=== Running E4 Regime B (Ultra-Rough 1M SPX: H=0.07, T=1/12, eta=1.9, rho=-0.9, xi0=0.235^2) ===")
    H, T, eta, rho, xi0 = 0.07, 1.0 / 12.0, 1.9, -0.9, 0.235 ** 2
    dk = 0.015
    strikes = [0.90, 0.95, math.exp(-dk), 1.00, math.exp(dk), 1.05, 1.08]
    idx_m, idx_atm, idx_p = 2, 3, 4
    xi0_fn = lambda s: np.full_like(s, xi0, dtype=np.float64)

    ref = price_spx_bridge_certified(
        H=H, T=T, eta=eta, rho=rho, xi0_fn=xi0_fn, strikes=strikes,
        levels=(64, 128, 256, 512), method="ce_i_only",
        n_shifts=16, n_half=4096, base_seed=20260500,
    )
    ref_skew = atm_skew_interval_arb(
        (ref["iv_continuum_lo"][idx_m], ref["iv_continuum_hi"][idx_m]),
        (ref["iv_continuum_lo"][idx_p], ref["iv_continuum_hi"][idx_p]),
        dk=dk,
    )
    print(f"  Ref ATM IV: [{100*ref['iv_continuum_lo'][idx_atm]:.3f}%, {100*ref['iv_continuum_hi'][idx_atm]:.3f}%] "
          f"(width={ref['iv_continuum_width_bps'][idx_atm]:.2f} bps), ATM Skew: [{ref_skew[0]:.4f}, {ref_skew[1]:.4f}]")
    print(f"  Ref statuses: {ref['status_per_strike']}")

    n_shifts, n_half = 16, 1024
    entries = []
    entries.append(audit_method_against_ref(bergomi_guyon_order2_iv(H, T, eta, rho, xi0, strikes), ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Analytical O(eta^2)"))
    for n in [64, 128]:
        r_ce = price_spx_single_level(H, T, eta, rho, xi0_fn, strikes, n=n, method="ce_i_only", n_shifts=n_shifts, n_half=n_half, base_seed=20260510 + n)
        r_ce["method"] = f"RoughCert-SPX-CE(n={n})"
        entries.append(audit_method_against_ref(r_ce, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "scrambled-Sobol RQMC"))

        r_bht = price_bht2020_bridge_rqmc(H, T, eta, rho, xi0_fn, strikes, n=n, n_shifts=n_shifts, n_half=n_half, base_seed=20260520 + n)
        entries.append(audit_method_against_ref(r_bht, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "scrambled-Sobol RQMC"))

        r_mp = price_mp2018_hybrid(H, T, eta, rho, xi0_fn, strikes, n=n, kappa=2, n_batches=n_shifts, n_half=n_half, base_seed=20260530 + n)
        r_mp["method"] = f"MP2018-Hybrid(k=2,n={n})"
        entries.append(audit_method_against_ref(r_mp, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Pseudo-random MC"))

        r_rd = price_hjm2021_rdonsker(H, T, eta, rho, xi0_fn, strikes, n=n, n_batches=n_shifts, n_half=n_half, base_seed=20260540 + n, variant="var_matched")
        r_rd["method"] = f"HJM2024-rDonsker(vm,n={n})"
        entries.append(audit_method_against_ref(r_rd, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Pseudo-random MC"))

    for M_fac in [8, 12]:
        r_bb = price_bayer_breneis_markovian(H, T, eta, rho, xi0_fn, strikes, M_factors=M_fac, n=128, n_batches=n_shifts, n_half=n_half, base_seed=20260550 + M_fac)
        r_bb["method"] = f"BayerBreneis2023(M={M_fac},n=128)"
        entries.append(audit_method_against_ref(r_bb, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Pseudo-random MC"))

    for e in entries:
        print(f"  {e['method']:28s}: max=[{e['max_smile_err_lo_bps']:.1f},{e['max_smile_err_hi_bps']:.1f}]bps, skew={e['atm_skew']:.4f}")

    return {
        "name": "Regime_B_UltraRough_1M",
        "params": {"H": H, "T": T, "eta": eta, "rho": rho, "xi0": xi0, "dk": dk, "strikes": strikes},
        "ref_bridge": ref,
        "ref_skew_interval": list(ref_skew),
        "entries": entries,
    }


def run_regime_C():
    print("\n=== Running E4 Regime C (Curved xi0(t)=0.234^2*(1+t)^2, H=0.10, T=0.50, eta=1.9, rho=-0.85) ===")
    H, T, eta, rho = 0.10, 0.50, 1.9, -0.85
    dk = 0.02
    strikes = [0.80, 0.90, math.exp(-dk), 1.00, math.exp(dk), 1.10, 1.20]
    idx_m, idx_atm, idx_p = 2, 3, 4
    xi0_fn = lambda s: (0.234 ** 2) * ((1.0 + np.asarray(s, dtype=np.float64)) ** 2)

    ref = price_spx_bridge_certified(
        H=H, T=T, eta=eta, rho=rho, xi0_fn=xi0_fn, strikes=strikes,
        levels=(64, 128, 256, 512), method="ce_i_only",
        n_shifts=16, n_half=4096, base_seed=20260600,
    )
    ref_skew = atm_skew_interval_arb(
        (ref["iv_continuum_lo"][idx_m], ref["iv_continuum_hi"][idx_m]),
        (ref["iv_continuum_lo"][idx_p], ref["iv_continuum_hi"][idx_p]),
        dk=dk,
    )
    print(f"  Ref ATM IV: [{100*ref['iv_continuum_lo'][idx_atm]:.3f}%, {100*ref['iv_continuum_hi'][idx_atm]:.3f}%] "
          f"(width={ref['iv_continuum_width_bps'][idx_atm]:.2f} bps), ATM Skew: [{ref_skew[0]:.4f}, {ref_skew[1]:.4f}]")
    print(f"  Ref statuses: {ref['status_per_strike']}")

    n_shifts, n_half = 16, 1024
    entries = []
    for n in [32, 64, 128]:
        r_ce = price_spx_single_level(H, T, eta, rho, xi0_fn, strikes, n=n, method="ce_i_only", n_shifts=n_shifts, n_half=n_half, base_seed=20260610 + n)
        r_ce["method"] = f"RoughCert-SPX-CE(n={n})"
        entries.append(audit_method_against_ref(r_ce, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "scrambled-Sobol RQMC"))

        r_bht = price_bht2020_bridge_rqmc(H, T, eta, rho, xi0_fn, strikes, n=n, n_shifts=n_shifts, n_half=n_half, base_seed=20260620 + n)
        entries.append(audit_method_against_ref(r_bht, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "scrambled-Sobol RQMC"))

        r_mp = price_mp2018_hybrid(H, T, eta, rho, xi0_fn, strikes, n=n, kappa=2, n_batches=n_shifts, n_half=n_half, base_seed=20260630 + n)
        r_mp["method"] = f"MP2018-Hybrid(k=2,n={n})"
        entries.append(audit_method_against_ref(r_mp, ref, ref_skew, idx_m, idx_atm, idx_p, dk, "Pseudo-random MC"))

    for e in entries:
        print(f"  {e['method']:28s}: max=[{e['max_smile_err_lo_bps']:.1f},{e['max_smile_err_hi_bps']:.1f}]bps, ATM=[{e['atm_iv_err_bps'][0]:.1f},{e['atm_iv_err_bps'][1]:.1f}]bps")

    return {
        "name": "Regime_C_Curved_ForwardVar",
        "params": {"H": H, "T": T, "eta": eta, "rho": rho, "xi0": "0.234^2*(1+t)^2", "dk": dk, "strikes": strikes},
        "ref_bridge": ref,
        "ref_skew_interval": list(ref_skew),
        "entries": entries,
    }


def main():
    t0 = time.perf_counter()
    regA = run_regime_A()
    regB = run_regime_B()
    regC = run_regime_C()
    out = {
        "experiment": "E4_benchmarks",
        "elapsed_s": time.perf_counter() - t0,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "threads": 1,
        },
        "regime_A": regA,
        "regime_B": regB,
        "regime_C": regC,
    }
    out_path = ROOT / "results" / "e4_benchmarks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} in {out['elapsed_s']:.1f}s.")


if __name__ == "__main__":
    main()
