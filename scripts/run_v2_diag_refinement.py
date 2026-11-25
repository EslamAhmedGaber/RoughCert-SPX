"""v2 study: can one finer dyadic level resolve the remaining Theorem 4(iii) diagnostic flags?

Re-runs the RoughCert-SPX-Bridge (ce_i_only + pilot-fitted six-control bank) envelopes that carried
diagnostic strikes in the v1 manuscript, keeping the original dyadic ladder (64, 128, 256, 512)
and ADDING the finer level n = 1024 at a matched path budget:
  * E4 Regime A reference (H=0.10, T=0.25, eta=1.9, rho=-0.9, xi0=0.04): 32 shifts x 4,096 paths;
  * E4 Regime B reference (H=0.07, T=1/12, eta=1.9, rho=-0.9, xi0=0.235^2): 32 shifts x 4,096 paths;
  * E6 Pure BDG6 (eta=sqrt(5)) and Mixed Scenario 1 at rho=-0.85 (xi0=0.235^2): 16 shifts x 4,096 paths.
Seeds start at 20262000 and are disjoint from every v1 run (v1 used 20260101-20261100 plus pilot offsets).
Results are appended case by case to results/v2_diag_refinement.json (labels: asymptotic-envelope
statistical(0.95) or diagnostic, exactly as in Theorem 4).
"""
from __future__ import annotations

import importlib.util
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

import numpy as np  # noqa: E402

from src.bs_arb import atm_skew_interval_arb  # noqa: E402
from src.bridge_rqmc import price_spx_bridge_certified  # noqa: E402

LEVELS = (64, 128, 256, 512, 1024)
OUT = ROOT / "results" / "v2_diag_refinement.json"


def _load_e6_module():
    spec = importlib.util.spec_from_file_location("run_e6_joint_spx_vix", ROOT / "scripts" / "run_e6_joint_spx_vix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summ(res: dict, idx_m: int, idx_p: int, dk: float) -> dict:
    skew = atm_skew_interval_arb(
        (res["iv_continuum_lo"][idx_m], res["iv_continuum_hi"][idx_m]),
        (res["iv_continuum_lo"][idx_p], res["iv_continuum_hi"][idx_p]),
        dk=dk,
    )
    keep = [
        "levels", "n_shifts", "paths_per_shift", "total_paths", "base_seed", "cold_setup_s", "solve_s", "strikes",
        "martingale_check_mean_F", "incr_means", "incr_ses", "finest_price_mean", "sampling_half_price",
        "bias_envelope_price", "continuum_price_lo", "continuum_price_hi", "iv_mid", "iv_continuum_lo",
        "iv_continuum_hi", "iv_continuum_width_bps", "emp_rate_per_strike", "beta_used_per_strike",
        "sign_flip_per_strike", "status_per_strike", "beta_fsw_theory",
    ]
    out = {k: res[k] for k in keep if k in res}
    out["atm_skew_interval"] = list(skew)
    out["n_diagnostic"] = int(sum(1 for s in res["status_per_strike"] if s == "diagnostic"))
    return out


def _write(payload: dict) -> None:
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, OUT)


def main() -> None:
    t0 = time.perf_counter()
    payload = {
        "experiment": "v2_diagnostic_flag_refinement",
        "levels": list(LEVELS),
        "status_note": "Theorem 4 labels; diagnostic widths are unverified display widths",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "threads": 1},
        "cases": {},
    }
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
            payload["cases"].update(prev.get("cases", {}))
        except Exception:
            pass

    # --- E4 Regime A reference
    if "regime_A" not in payload["cases"]:
        dk = 0.02
        strikes = [0.85, 0.90, 0.95, math.exp(-dk), 1.00, math.exp(dk), 1.06, 1.12]
        xi0 = 0.04
        res = price_spx_bridge_certified(
            H=0.10, T=0.25, eta=1.9, rho=-0.9, xi0_fn=lambda s: np.full_like(s, xi0, dtype=np.float64),
            strikes=strikes, levels=LEVELS, method="ce_i_only", n_shifts=32, n_half=2048, base_seed=20262000,
        )
        payload["cases"]["regime_A"] = {"params": {"H": 0.10, "T": 0.25, "eta": 1.9, "rho": -0.9, "xi0": xi0, "dk": dk}, **_summ(res, 3, 5, dk)}
        _write(payload)
        print(f"regime_A done: diag={payload['cases']['regime_A']['n_diagnostic']} t={time.perf_counter()-t0:.0f}s", flush=True)

    # --- E4 Regime B reference
    if "regime_B" not in payload["cases"]:
        dk = 0.015
        strikes = [0.90, 0.95, math.exp(-dk), 1.00, math.exp(dk), 1.05, 1.08]
        xi0 = 0.235 ** 2
        res = price_spx_bridge_certified(
            H=0.07, T=1.0 / 12.0, eta=1.9, rho=-0.9, xi0_fn=lambda s: np.full_like(s, xi0, dtype=np.float64),
            strikes=strikes, levels=LEVELS, method="ce_i_only", n_shifts=32, n_half=2048, base_seed=20262100,
        )
        payload["cases"]["regime_B"] = {"params": {"H": 0.07, "T": 1.0 / 12.0, "eta": 1.9, "rho": -0.9, "xi0": xi0, "dk": dk}, **_summ(res, 2, 4, dk)}
        _write(payload)
        print(f"regime_B done: diag={payload['cases']['regime_B']['n_diagnostic']} t={time.perf_counter()-t0:.0f}s", flush=True)

    dk = 0.02
    strikes6 = [0.85, 0.90, 0.95, math.exp(-dk), 1.00, math.exp(dk), 1.06, 1.12]
    xi06 = 0.235 ** 2
    H6, T6 = 0.10, 0.25

    # --- E6 Pure BDG6 rho = -0.85
    if "e6_pure_bdg6_rho-0.85" not in payload["cases"]:
        eta_bdg6 = 1.0 / math.sqrt(2.0 * H6)
        res = price_spx_bridge_certified(
            H=H6, T=T6, eta=eta_bdg6, rho=-0.85, xi0_fn=lambda s: np.full_like(s, xi06, dtype=np.float64),
            strikes=strikes6, levels=LEVELS, method="ce_i_only", n_shifts=16, n_half=2048, base_seed=20262200,
        )
        payload["cases"]["e6_pure_bdg6_rho-0.85"] = {"params": {"H": H6, "T": T6, "eta": eta_bdg6, "rho": -0.85, "xi0": xi06, "dk": dk}, **_summ(res, 3, 5, dk)}
        _write(payload)
        print(f"e6_pure_bdg6 done: diag={payload['cases']['e6_pure_bdg6_rho-0.85']['n_diagnostic']} t={time.perf_counter()-t0:.0f}s", flush=True)

    # --- E6 Mixed Scenario 1 rho = -0.85
    if "e6_mixed_scen1_rho-0.85" not in payload["cases"]:
        e6mod = _load_e6_module()
        eta1 = 1.4 / math.sqrt(2.0 * H6)
        eta2 = 0.7 / math.sqrt(2.0 * H6)
        res = e6mod.price_mixed_rbergomi_bridge(
            H=H6, T=T6, eta1=eta1, eta2=eta2, lam=0.3, rho=-0.85, xi0_flat=xi06, strikes=strikes6,
            levels=LEVELS, n_shifts=16, n_half=2048, base_seed=20262300,
        )
        payload["cases"]["e6_mixed_scen1_rho-0.85"] = {"params": {"H": H6, "T": T6, "eta1": eta1, "eta2": eta2, "lambda": 0.3, "rho": -0.85, "xi0": xi06, "dk": dk}, **_summ(res, 3, 5, dk)}
        _write(payload)
        print(f"e6_mixed done: diag={payload['cases']['e6_mixed_scen1_rho-0.85']['n_diagnostic']} t={time.perf_counter()-t0:.0f}s", flush=True)

    payload["elapsed_s"] = time.perf_counter() - t0
    _write(payload)
    print(f"all cases done in {payload['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
