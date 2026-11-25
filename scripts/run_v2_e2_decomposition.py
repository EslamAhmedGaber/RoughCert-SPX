"""v2 study: decomposition of the E2 deterministic-certifiability contraction.

E2 reports a 25 bps boundary for rho = 0 computed with the sharp second-order two-moment
bracket (Proposition 1) and boundaries for rho = -0.5, -0.9 computed with the first-order
Put-Lipschitz bracket (Theorem 2, k = 0 surrogate). Their ratio mixes two effects:
  (1) losing second-order sharpness (two-moment -> first-order bracket at the SAME rho = 0);
  (2) switching on leverage inside the SAME first-order bracket (rho = 0 -> rho = -0.9).
This script bisects the rho = 0 first-order (k = 0) boundary with exactly the E2 settings
(initial bracket [1e-4, 2.5] in eta*T^H, 18 halvings, stored value = last certified point)
and writes results/v2_e2_decomposition.json. Every width is an Arb-certified ATM IV width.
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

from src.deterministic_spx import deterministic_spx_bracket_k0_arb  # noqa: E402

T_BASE = 0.25
XI0_BASE = 0.04
H_GRID = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45]
TOLS_BPS = [10.0, 25.0, 50.0]


def bisect_boundary(H: float, rho: float, tol_bps: float) -> float:
    lo_e, hi_e = 1e-4, 2.5
    for _ in range(18):
        mid = 0.5 * (lo_e + hi_e)
        eta_m = mid / (T_BASE ** H)
        w = deterministic_spx_bracket_k0_arb(H=H, T=T_BASE, eta=eta_m, rho=rho, xi0_flat=XI0_BASE, K=1.0)["iv_width_bps"]
        if w <= tol_bps:
            lo_e = mid
        else:
            hi_e = mid
    return lo_e


def main() -> None:
    t0 = time.perf_counter()
    e2 = json.loads((ROOT / "results" / "e2_det_phase_map.json").read_text(encoding="utf-8"))
    by_H = {round(r["H"], 4): r for r in e2["boundaries"]}
    rows = []
    for H in H_GRID:
        row = {"H": H}
        for tol in TOLS_BPS:
            row[f"eta_eff_star_rho0_k0_{int(tol)}bps"] = bisect_boundary(H, 0.0, tol)
        b = by_H[round(H, 4)]
        r0_2m = b["eta_eff_star_rho0_2m_25bps"]
        r0_k0 = row["eta_eff_star_rho0_k0_25bps"]
        r9_k0 = b["eta_eff_star_rho_m09_k0_25bps"]
        row["factor_total_25bps"] = r0_2m / r9_k0
        row["factor_second_order_25bps"] = r0_2m / r0_k0
        row["factor_leverage_25bps"] = r0_k0 / r9_k0
        rows.append(row)
        print(f"H={H:.2f}: rho0 2m={r0_2m:.5f} rho0 k0={r0_k0:.5f} rho-0.9 k0={r9_k0:.5f} | "
              f"total={row['factor_total_25bps']:.2f} = 2nd-order {row['factor_second_order_25bps']:.2f} x leverage {row['factor_leverage_25bps']:.2f}")
    out = {
        "experiment": "v2_E2_boundary_decomposition",
        "status": "Arb-certified widths; boundaries are last certified bisection points (18 halvings of [1e-4, 2.5])",
        "T": T_BASE,
        "xi0": XI0_BASE,
        "K": 1.0,
        "rows": rows,
        "environment": {"python": platform.python_version(), "threads": 1},
        "elapsed_s": time.perf_counter() - t0,
    }
    (ROOT / "results" / "v2_e2_decomposition.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote results/v2_e2_decomposition.json in {out['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
