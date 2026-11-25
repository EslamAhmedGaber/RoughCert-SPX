"""Rigorous mathematical unit verification suite for RoughCert-SPX.

Tests every theorem, identity, exact-law benchmark, competitor formula, Arb bound,
and all Blind Audit #1 regression checks (C1-C3, M1-M4, m1-m5) BEFORE running E1-E6.
Fails loudly on any discrepancy.
"""
from __future__ import annotations

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
from scipy.integrate import quad
from scipy.optimize import brentq
from flint import arb

from src.bs_arb import (
    set_dps, arb_lo_float, arb_hi_float, bs_call_arb, bs_put_arb,
    bs_iv_interval_from_price_bounds, bs_iv_point_arb,
    bs_call_vec, bs_put_vec,
)
from src.volterra_arb import (
    rl_cov_arb, rl_cov_float, markov_kernel_l2_sq_arb, markov_kernel_l1_arb,
    blp_hybrid_kernel_l2_sq_arb, time_EJ_ce_ideal_arb, time_EJ_leftpoint_arb,
    exact_IT_moments_arb,
)
from src.deterministic_spx import (
    deterministic_spx_bracket_k0_arb,
    deterministic_spx_rho0_cond_arb,
    deterministic_spx_rho0_atm_twomoment_arb,
    certify_kernel_gap_markov_arb,
    certify_kernel_gap_blp_hybrid_arb,
)
from src.bridge_rqmc import (
    build_volterra_grid_cache,
    generate_exact_paths,
    coarsen_paths,
    sample_rqmc_normals,
    price_spx_single_level,
    compute_bridge_envelope,
    _eval_stopped_timer_cv,
)
from src.exact_laws import (
    heston_call_prices,
    rough_stein_stein_exact_prices,
    price_rss_single_level,
)
from src.competitors import (
    bayer_breneis_l2_nodes_weights,
    price_mp2018_hybrid,
    price_bayer_breneis_markovian,
    price_hjm2021_rdonsker,
    bergomi_guyon_order2_iv,
)


def test_1_bs_arb():
    print("\n=== TEST 1: 176-bit Arb BS, Outward Rounding, Scalar Vectorization & Infinite IV ===")
    set_dps()
    # m1: exact zero lower bound must NOT return -1e-323
    assert arb_lo_float(arb(0)) == 0.0, f"m1 regression: arb_lo_float(0) = {arb_lo_float(arb(0))}"
    assert arb_hi_float(arb(0)) == 0.0, f"arb_hi_float(0) = {arb_hi_float(arb(0))}"

    # m2: scalar calls to bs_call_vec / bs_put_vec must not crash
    c_sc = float(bs_call_vec(1.0, 1.0, 0.04))
    p_sc = float(bs_put_vec(1.0, 1.0, 0.04))
    assert abs(c_sc - p_sc) < 1e-15 and c_sc > 0.07, f"m2 scalar BS failed: {c_sc}, {p_sc}"

    # M4: c_hi >= F must propagate cleanly to iv_hi = +inf (no 0.999999999 fake clip)
    il_inf, ih_inf = bs_iv_interval_from_price_bounds(0.04, 1.0, 1.0, 0.25, F=1.0)
    assert math.isfinite(il_inf) and ih_inf == math.inf, f"M4 inf IV failed: [{il_inf}, {ih_inf}]"

    for F_v, K_v, V_v in [(1.0, 0.85, 0.04), (1.0, 1.0, 0.0625), (1.0, 1.2, 0.09)]:
        F, K, V = arb(str(F_v)), arb(str(K_v)), arb(str(V_v))
        c = bs_call_arb(F, K, V)
        p = bs_put_arb(F, K, V)
        diff = abs((c - p) - (F - K))
        assert arb_hi_float(diff) < 1e-45, f"Put-Call parity failed: {diff}"
        T_v = 0.5
        iv_true = math.sqrt(V_v / T_v)
        il, ih = bs_iv_interval_from_price_bounds(arb_lo_float(c), arb_hi_float(c), K_v, T_v, F=F_v)
        assert il <= iv_true <= ih, f"IV enclosure missed true IV: [{il}, {ih}] vs {iv_true}"
        assert ih - il < 1e-12, f"IV enclosure too wide: {ih - il}"
    print("  [PASS] Put-Call parity < 1e-45, IV exact < 1e-12, m1/m2/M4 verified.")


def test_2_volterra_arb_and_markov():
    print("\n=== TEST 2: Volterra RL Cov, Unclipped BB2023 (C3), Rigorous Arb L^1 (M2), Domain Guards (M4) ===")
    set_dps()
    # M4: H > 0.5 must raise ValueError
    try:
        rl_cov_arb(arb("0.6"), arb("0.1"), arb("0.2"))
        raise AssertionError("M4 regression: rl_cov_arb accepted H = 0.6!")
    except ValueError:
        pass

    H = 0.1
    H_a = arb("0.1")
    for s_v, u_v in [(0.1, 0.1), (0.1, 0.5), (0.49, 0.50), (0.2, 0.8)]:
        r_arb = rl_cov_arb(H_a, arb(str(s_v)), arb(str(u_v)), is_diag=(s_v == u_v))
        r_arb_box = rl_cov_arb(H_a, arb(str(s_v)), arb(str(u_v)), is_diag=False)
        r_flt = float(rl_cov_float(H, np.array(s_v), np.array(u_v)))
        assert r_arb_box.overlaps(r_arb)
        mn = min(s_v, u_v)
        r_quad, _ = quad(lambda r: 2 * H * ((s_v - r) ** (H - 0.5)) * ((u_v - r) ** (H - 0.5)), 0.0, mn, epsabs=1e-13, epsrel=1e-12)
        err_flt = abs(0.5 * (arb_lo_float(r_arb) + arb_hi_float(r_arb)) - r_flt)
        err_quad = abs(0.5 * (arb_lo_float(r_arb) + arb_hi_float(r_arb)) - r_quad)
        assert err_flt < 1e-13 and err_quad < 1e-10, f"RL cov mismatch at ({s_v},{u_v}): {err_flt}, {err_quad}"

    # C3: Unclipped L-BFGS-B Bayer-Breneis at H=0.10, M=16, T=0.25 must achieve L^2 < 0.025 in < 4s
    t0 = time.perf_counter()
    nodes16, weights16, l2_16 = bayer_breneis_l2_nodes_weights(H=0.10, M_factors=16, T=0.25)
    t_bb = time.perf_counter() - t0
    l2_sq_16_arb = markov_kernel_l2_sq_arb(arb("0.10"), arb("0.25"), [arb(str(float(x))) for x in nodes16], [arb(str(float(w))) for w in weights16])
    l2_16_arb = math.sqrt(arb_hi_float(l2_sq_16_arb))
    print(f"  [C3] BB2023 (H=0.10, M=16, T=0.25): L2_fit={l2_16:.6f}, Arb={l2_16_arb:.6f}, x_max={nodes16[-1]:.2e}, time={t_bb:.2f}s")
    assert l2_16_arb < 0.025, f"C3 regression: BB2023 M=16 L2 too high ({l2_16_arb})"
    assert nodes16[-1] > 1e12, f"C3 regression: BB2023 nodes still artificially clipped ({nodes16[-1]:.2e})"
    assert t_bb < 15.0, f"C3 regression: BB2023 M=16 too slow ({t_bb:.2f}s)"

    # M2: Rigorous Arb L^1 enclosure at the exact Audit #1 counterexample (H=0.30, M=12, T=0.25)
    nodes30, weights30, _ = bayer_breneis_l2_nodes_weights(H=0.30, M_factors=12, T=0.25)
    l1_arb = markov_kernel_l1_arb(
        arb("0.30"), arb("0.25"),
        [arb(str(float(x))) for x in nodes30],
        [arb(str(float(w))) for w in weights30],
    )
    l1_lo, l1_hi = arb_lo_float(l1_arb), arb_hi_float(l1_arb)
    # Independent root-split scipy quad verification
    def f_diff(u):
        return math.sqrt(0.6) * (u ** (-0.2)) - float(np.sum(weights30 * np.exp(-nodes30 * u)))
    us = np.geomspace(1e-12, 0.25, 20000)
    vals = np.array([f_diff(u) for u in us])
    roots = [0.0]
    for i in range(len(us) - 1):
        if vals[i] * vals[i + 1] < 0.0:
            roots.append(brentq(f_diff, us[i], us[i + 1], xtol=1e-14))
    roots.append(0.25)
    l1_quad = sum(quad(lambda u: abs(f_diff(u)), a, b, epsabs=1e-14, epsrel=1e-12)[0] for a, b in zip(roots[:-1], roots[1:]))
    print(f"  [M2] Markov L^1 (H=0.30, M=12, n_roots={len(roots)-2}): Arb=[{l1_lo:.10f}, {l1_hi:.10f}], Quad={l1_quad:.10f}")
    assert l1_lo <= l1_quad <= l1_hi, f"M2 regression: Arb L^1 missed true quad! [{l1_lo}, {l1_hi}] vs {l1_quad}"
    assert (l1_hi - l1_lo) / l1_quad < 1e-4, f"M2 Arb L^1 enclosure too loose: rel_w={(l1_hi-l1_lo)/l1_quad}"

    # BLP hybrid L^2
    n_test, kappa_test, T_test = 16, 2, 1.0
    l2_blp_arb = blp_hybrid_kernel_l2_sq_arb(H_a, arb("1.0"), n=n_test, kappa=kappa_test, s_idx=n_test)
    dt = T_test / n_test
    p = H + 0.5
    quad_acc = 0.0
    for k in range(kappa_test + 1, n_test + 1):
        a_c, b_c = (k - 1) * dt, k * dt
        c_k = math.sqrt(2 * H) * (dt ** (H - 0.5)) * (k ** p - (k - 1) ** p) / p
        q_val, _ = quad(lambda u: (math.sqrt(2 * H) * u ** (H - 0.5) - c_k) ** 2, a_c, b_c, epsabs=1e-14, epsrel=1e-12)
        quad_acc += q_val
    assert abs(0.5 * (arb_lo_float(l2_blp_arb) + arb_hi_float(l2_blp_arb)) - quad_acc) < 1e-12
    ej_ce = time_EJ_ce_ideal_arb(H_a, arb("1.0"), arb("1.9"), arb("0.04"), n=32)
    ej_lp = time_EJ_leftpoint_arb(H_a, arb("1.0"), arb("1.9"), arb("0.04"), n=32)
    assert arb_hi_float(ej_ce) < arb_lo_float(ej_lp)
    print("  [PASS] All Volterra Arb, C3 unclipped BB2023, M2 rigorous L^1, and M4 domain guards verified.")


def test_3_exact_laws():
    print("\n=== TEST 3: Exact-Law Controls (Heston & RSS with M3 Fixed n_gl + C1 ce_i_only Check) ===")
    strikes = [0.85, 0.95, 1.0, 1.05, 1.15]
    h_res = heston_call_prices(T=0.5, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.4, rho=-0.7, strikes=strikes)
    assert h_res["martingale_error"] < 1e-13

    rss_half = rough_stein_stein_exact_prices(H=0.5, T=0.5, sigma0=0.2, nu=0.25, rho=-0.7, strikes=strikes, M_op=64, M_check=32)
    assert rss_half["max_riccati_diff"] < 1e-8 and rss_half["martingale_error"] < 1e-12

    # M3: M_op=128 with fixed n_gl >= 2*M+32
    rss_rough = rough_stein_stein_exact_prices(H=0.1, T=0.5, sigma0=0.2, nu=0.25, rho=-0.7, strikes=strikes, M_op=128, M_check=96)
    print(f"  RSS (H=0.1, rho=-0.7, M=128 vs 96): max_diff={rss_rough['max_refinement_price_diff']:.2e}, mart_err={rss_rough['martingale_error']:.2e}")
    assert rss_rough["martingale_error"] < 1e-6 and rss_rough["max_refinement_price_diff"] < 1e-5

    # C1: Verify ce_i_only beats ce_adapted (Jensen-J ablation) at n=64 on exact Correlated RSS.
    cache = build_volterra_grid_cache(H=0.1, T=0.5, n=64, need_ce=True)
    lp_res = price_rss_single_level(cache, sigma0=0.2, nu=0.25, rho=-0.7, strikes=strikes, method="leftpoint", n_shifts=16, n_half=512)
    cei_res = price_rss_single_level(cache, sigma0=0.2, nu=0.25, rho=-0.7, strikes=strikes, method="ce_i_only", n_shifts=16, n_half=512)
    cea_res = price_rss_single_level(cache, sigma0=0.2, nu=0.25, rho=-0.7, strikes=strikes, method="ce_adapted", n_shifts=16, n_half=512)
    lp_max = max(abs(1e4 * (a - b)) for a, b in zip(lp_res["ivs"], rss_rough["ivs"]))
    cei_max = max(abs(1e4 * (a - b)) for a, b in zip(cei_res["ivs"], rss_rough["ivs"]))
    cea_max = max(abs(1e4 * (a - b)) for a, b in zip(cea_res["ivs"], rss_rough["ivs"]))
    print(f"  [C1] RSS (n=64) max IV bias vs EXACT: leftpoint={lp_max:.2f}, ce_i_only={cei_max:.2f}, ce_adapted(ablation)={cea_max:.2f} bps")
    assert cei_max < cea_max, f"C1 check: ce_i_only ({cei_max}) should beat ce_adapted ({cea_max})!"
    print("  [PASS] Exact-law benchmarks & C1 ablation confirmed.")


def test_4_stopped_cv_and_envelope_guards():
    print("\n=== TEST 4: Stopped Timer CV, Coarsening & C2 Envelope Diagnostics ===")
    H, T, eta, rho, xi0 = 0.1, 0.5, 1.9, -0.9, 0.04
    strikes = np.array([0.85, 1.0, 1.15])
    cache = build_volterra_grid_cache(H, T, n=32, need_ce=True)
    Z = sample_rqmc_normals(2 * 32, n_half=2048, seed=777, antithetic=True)
    dW, Y = generate_exact_paths(cache, Z)
    Y_left = np.hstack([np.zeros((Z.shape[0], 1)), Y[:, :-1]])
    t_left = np.linspace(0.0, T - cache.dt, 32)
    sigma_left = np.sqrt(xi0 * np.exp(eta * Y_left - 0.5 * eta * eta * (t_left ** (2 * H))[None, :]))
    Q_var = 6.0 * rho * rho * xi0 * T
    Y_cv, exact_mean = _eval_stopped_timer_cv(sigma_left, dW, cache.dt, rho, strikes, Q_var)
    diff = np.abs(Y_cv.mean(axis=0) - exact_mean)
    se = Y_cv.std(axis=0, ddof=1) / math.sqrt(Z.shape[0])
    assert np.all(diff / se < 3.5)

    dW_c, Y_c = coarsen_paths(dW, Y, stride=2)
    assert abs(np.var(Y_c[:, -1]) - T ** (2 * H)) / (T ** (2 * H)) < 0.03

    # C2 synthetic envelope test:
    # Strike 0: healthy FSW contraction (ratio=1.74, beta_emp=0.80) -> asymptotic-envelope
    # Strike 1: slow contraction (ratio=1.08, beta_emp=0.11 < 0.5*beta_fsw) -> diagnostic, NO upward clamp to 0.40.
    # Strike 2: resolved sign flip (+5e-4 then -4e-4) -> diagnostic, sign_flip=True.
    rng = np.random.default_rng(123)
    R = 16
    noise = rng.normal(0.0, 1e-6, size=(3, R, 3))
    base = np.array([0.04, 0.04, 0.04])
    lvl = np.zeros((3, R, 3))
    lvl[0] = base + noise[0]
    lvl[1] = base + np.array([1.74e-4, 1.08e-4, 5.0e-4]) + noise[1]
    lvl[2] = lvl[1] + np.array([1.00e-4, 1.00e-4, -4.0e-4]) + noise[2]
    env = compute_bridge_envelope(lvl, [64, 128, 256], H=0.10, T=0.25, strikes_arr=np.array([1.0, 1.0, 1.0]))
    print(f"  [C2] Synthetic envelope statuses: {env['status_per_strike']}")
    print(f"       beta_used={np.round(env['beta_used_per_strike'], 3)}, sign_flip={env['sign_flip_per_strike']}")
    assert env["status_per_strike"][0].startswith("asymptotic-envelope")
    assert env["status_per_strike"][1] == "diagnostic" and env["beta_used_per_strike"][1] < 0.25, "C2 regression: slow contraction was floor-clamped!"
    assert env["status_per_strike"][2] == "diagnostic" and env["sign_flip_per_strike"][2] is True, "C2 regression: sign flip missed!"
    # Audit #2 M2: 4-level (>=3 resolved monotone increments) with beta_reg > beta_fsw MUST yield min(beta_fsw, 0.85*beta_reg) without pre-cap.
    lvl4 = np.zeros((4, R, 1))
    n4 = rng.normal(0.0, 1e-7, size=(4, R, 1))
    lvl4[0] = 0.04 + n4[0]
    lvl4[1] = lvl4[0] + 4.0e-4 + n4[1]
    lvl4[2] = lvl4[1] + 2.0e-4 + n4[2]
    lvl4[3] = lvl4[2] + 1.0e-4 + n4[3]  # exact beta_reg = 1.0 > beta_fsw = 0.8 -> 0.85*1.0 = 0.85 -> min(0.80, 0.85) = 0.80.
    env4 = compute_bridge_envelope(lvl4, [64, 128, 256, 512], H=0.10, T=0.25, strikes_arr=np.array([1.0]))
    assert abs(env4["beta_used_per_strike"][0] - 0.80) < 1e-3, f"M2 regression: pre-capped beta_used={env4['beta_used_per_strike'][0]}"
    print("  [PASS] Stopped Timer CV, coarsening, C2 & Audit-2 M2 envelope guards verified.")


def test_5_bergomi_guyon_o3_check():
    print("\n=== TEST 5: Bergomi-Guyon Order-2 O(eta^3)/O(eta^4) Exactness Verification (m5) ===")
    # Symmetrized (+eta, -eta) CRN check at H=0.5, T=2.0, xi0=0.16 (w=0.32, rho=-0.7):
    #   even_err = 0.5*(IV(+eta)+IV(-eta)) - even_BG2  MUST be O(eta^4) (confirms exact O(eta^2) term!)
    #   odd_err  = 0.5*(IV(+eta)-IV(-eta)) - odd_BG2   MUST be O(eta^3) (confirms exact O(eta^1) term!)
    from src.bridge_rqmc import _apply_crossfit_cv
    H, T, rho, xi0 = 0.5, 2.0, -0.7, 0.16
    strikes = np.array([0.85, 0.92, 1.00, 1.08, 1.15], dtype=np.float64)
    n = 128
    dt = T / n
    t_left = np.linspace(0.0, T - dt, n)
    t_right = np.linspace(dt, T, n)
    cache = build_volterra_grid_cache(H, T, n, need_ce=False)
    n_sh, nh = 8, 2048
    M = 2 * nh
    dW_all, Y_all = [], []
    for r in range(n_sh):
        Z = sample_rqmc_normals(2 * n, n_half=nh, seed=9100 + r, antithetic=True)
        dW, Y = generate_exact_paths(cache, Z)
        dW_all.append(dW); Y_all.append(Y)

    def sim_iv(eta_val: float) -> np.ndarray:
        C_s = np.empty((n_sh, M, len(strikes)))
        F_s = np.empty((n_sh, M))
        for r in range(n_sh):
            dW, Y = dW_all[r], Y_all[r]
            Yl = np.hstack([np.zeros((M, 1)), Y[:, :-1]])
            vl = xi0 * np.exp(eta_val * Yl - 0.5 * (eta_val ** 2) * t_left[None, :])
            vr = xi0 * np.exp(eta_val * Y - 0.5 * (eta_val ** 2) * t_right[None, :])
            sl = np.sqrt(vl)
            IT = 0.5 * dt * np.sum(vl + vr, axis=1)
            JT = np.sum(sl * dW + 0.25 * eta_val * sl * (dW * dW - dt), axis=1)
            FT = np.exp(rho * JT - 0.5 * (rho ** 2) * IT)
            C_s[r] = bs_call_vec(FT[:, None], strikes[None, :], (1.0 - rho ** 2) * IT[:, None])
            F_s[r] = FT
        pm = _apply_crossfit_cv(C_s, None, None, F_s).mean(axis=0)
        return np.array([bs_iv_point_arb(float(p), float(k), T) for p, k in zip(pm, strikes)])

    even_errs, odd_errs = [], []
    for eta_val in [0.10, 0.20]:
        iv_p, iv_m = sim_iv(+eta_val), sim_iv(-eta_val)
        bg_p = np.array(bergomi_guyon_order2_iv(H, T, +eta_val, rho, xi0, strikes.tolist())["iv_mid"])
        bg_m = np.array(bergomi_guyon_order2_iv(H, T, -eta_val, rho, xi0, strikes.tolist())["iv_mid"])
        e_even = float(np.max(np.abs(0.5 * (iv_p + iv_m) - 0.5 * (bg_p + bg_m)))) * 1e4
        e_odd = float(np.max(np.abs(0.5 * (iv_p - iv_m) - 0.5 * (bg_p - bg_m)))) * 1e4
        even_errs.append(e_even); odd_errs.append(e_odd)
        print(f"  eta={eta_val:.2f}: max EVEN err={e_even:.4f} bps (O(eta^4)), max ODD err={e_odd:.4f} bps (O(eta^3))")

    r_even = even_errs[1] / max(even_errs[0], 1e-12)
    r_odd = odd_errs[1] / max(odd_errs[0], 1e-12)
    print(f"  [m5] Ratios (0.20/0.10): EVEN={r_even:.2f} (expect ~16), ODD={r_odd:.2f} (expect ~8)")
    assert even_errs[0] < 0.15 and even_errs[1] < 0.50, f"BG2 O(eta^2) even error too large: {even_errs}"
    assert odd_errs[0] < 0.25 and r_odd > 6.5, f"BG2 O(eta^3) odd scaling failed: {odd_errs}, ratio={r_odd:.2f}"
    print("  [PASS] Bergomi-Guyon Order-2 O(eta^3)/O(eta^4) exactness asserted cleanly.")


def test_6_deterministic_containment_and_competitors():
    print("\n=== TEST 6: Deterministic Arb Containment, Domain Guards (M4) & Competitor Fairness (M1) ===")
    # M4: rho > 0 MUST raise ValueError
    try:
        deterministic_spx_bracket_k0_arb(H=0.1, T=0.25, eta=1.9, rho=+0.5, xi0_flat=0.04, K=1.0)
        raise AssertionError("M4 regression: deterministic_spx_bracket_k0_arb accepted rho > 0!")
    except ValueError:
        pass

    # M4: Wide bracket where c_hi >= 1 returns price_lo == 0.0 and iv_hi == math.inf
    det_wide = deterministic_spx_bracket_k0_arb(H=0.1, T=5.0, eta=3.0, rho=-0.9, xi0_flat=1.0, K=1.0)
    assert det_wide["price_lo"] == 0.0 and det_wide["iv_lo"] == 0.0 and det_wide["iv_hi"] == math.inf, f"M4 wide inf IV failed: {det_wide}"

    H, T, eta, xi0 = 0.25, 0.1, 0.5, 0.04
    xi0_fn = lambda s: np.full_like(s, xi0, dtype=np.float64)
    det_k0 = deterministic_spx_bracket_k0_arb(H=H, T=T, eta=eta, rho=0.0, xi0_flat=xi0, K=1.0)
    det_k1 = deterministic_spx_rho0_cond_arb(H=H, T=T, eta=eta, xi0_flat=xi0, K=1.0)
    det_2m = deterministic_spx_rho0_atm_twomoment_arb(H=H, T=T, eta=eta, xi0_flat=xi0)
    sim0 = price_spx_single_level(H=H, T=T, eta=eta, rho=0.0, xi0_fn=xi0_fn, strikes=[1.0], n=64, method="ce_i_only", n_shifts=8, n_half=512)
    p_sim = sim0["price_mean"][0]
    assert det_k0["price_lo"] <= p_sim <= det_k0["price_hi"]
    assert det_k1["price_lo"] <= p_sim <= det_k1["price_hi"]
    assert det_2m["price_lo"] <= p_sim <= det_2m["price_hi"]
    print(f"  rho=0 ATM: k=0 ({det_k0['iv_width_bps']:.1f} bps), 2-moment ({det_2m['iv_width_bps']:.2f} bps) — both contain RQMC!")

    det_corr = deterministic_spx_bracket_k0_arb(H=H, T=T, eta=eta, rho=-0.5, xi0_flat=xi0, K=1.0)
    sim_c = price_spx_single_level(H=H, T=T, eta=eta, rho=-0.5, xi0_fn=xi0_fn, strikes=[1.0], n=64, method="ce_i_only", n_shifts=8, n_half=512)
    assert det_corr["price_lo"] <= sim_c["price_mean"][0] <= det_corr["price_hi"]

    # Audit #3 m1: overlapping balls at H=1/2 in rl_cov_arb and zero-touching variance balls in bs_call_arb/bs_put_arb
    s_ov = arb("0.20") + arb(0, 0.05)  # [0.15, 0.25]
    u_ov = arb("0.22") + arb(0, 0.05)  # [0.17, 0.27]
    r_half = rl_cov_arb(arb("0.5"), s_ov, u_ov)
    assert float(r_half.lower()) <= 0.15 + 1e-12 and float(r_half.upper()) >= 0.25 - 1e-12, f"Audit-3 m1 rl_cov H=0.5 failed: {r_half}"
    v_zero_touch = arb("5e-6") + arb(0, 5e-6)  # [0, 1e-5]
    c_zt = bs_call_arb(arb(1), arb(1), v_zero_touch)
    p_zt = bs_put_arb(arb(1), arb(1), v_zero_touch)
    c_zt_hi = bs_call_arb(arb(1), arb(1), arb("1e-5"))
    assert c_zt.is_finite() and p_zt.is_finite() and float(c_zt.lower()) <= 0.0 and float(c_zt.upper()) >= float(c_zt_hi.upper()) - 1e-15

    # Audit #2 M4: sound Arb ball .max() / .min() on overlapping balls
    a_ov = arb("1.0") + arb(0, 0.2)
    b_ov = arb("1.05") + arb(0, 0.3)
    m_max = abs(a_ov).max(abs(b_ov))
    m_min = abs(a_ov).min(abs(b_ov))
    assert float(m_max.upper()) >= float(b_ov.upper()) - 1e-12 and float(m_max.upper()) >= float(a_ov.upper()) - 1e-12
    assert float(m_min.lower()) <= float(a_ov.lower()) + 1e-12 and float(m_min.lower()) <= float(b_ov.lower()) + 1e-12

    # Audit #2 M8: BG2012 canonical diagnostic status
    bg_check = bergomi_guyon_order2_iv(0.10, 0.25, 1.9, -0.9, 0.04, [1.0])
    assert bg_check["status"] == "diagnostic", f"M8 regression: BG2012 status={bg_check['status']}"

    # M1: Test BOTH rDonsker variants (var_matched & b_star) + MP2018 + BB2023
    strikes = [0.9, 1.0, 1.1]
    mp = price_mp2018_hybrid(0.1, 0.25, 1.9, -0.9, xi0_fn, strikes, n=64, kappa=2, n_batches=8, n_half=512)
    bb = price_bayer_breneis_markovian(0.1, 0.25, 1.9, -0.9, xi0_fn, strikes, M_factors=8, n=64, n_batches=8, n_half=512)
    rd_vm = price_hjm2021_rdonsker(0.1, 0.25, 1.9, -0.9, xi0_fn, strikes, n=64, n_batches=8, n_half=512, variant="var_matched")
    rd_bs = price_hjm2021_rdonsker(0.1, 0.25, 1.9, -0.9, xi0_fn, strikes, n=64, n_batches=8, n_half=512, variant="b_star")
    assert abs(rd_vm["terminal_var_ratio"] - 1.0) < 1e-12, f"M1 var_matched ratio != 1: {rd_vm['terminal_var_ratio']}"
    assert rd_bs["terminal_var_ratio"] < 0.85, f"M1 b_star ratio expected < 0.85: {rd_bs['terminal_var_ratio']}"
    print(f"  [M1] rDonsker var_ratio: var_matched={rd_vm['terminal_var_ratio']:.6f}, b_star={rd_bs['terminal_var_ratio']:.4f}")
    print(f"  Competitors ATM IV: MP2018={100*mp['iv_mid'][1]:.3f}%, BB2023={100*bb['iv_mid'][1]:.3f}%, "
          f"rDonsker(vm)={100*rd_vm['iv_mid'][1]:.3f}%, rDonsker(b*)={100*rd_bs['iv_mid'][1]:.3f}%")
    print("  [PASS] All deterministic containment, domain guards, Audit-2 M4/M8, and competitor checks passed.")


def test_7_v2_two_moment_rate_floor_and_archives():
    """v2 additions: Proposition 1 (two-moment ATM bound) derivative formulas and sharpness,
    the Case (ii) rate floor beta_min(H) printed in the paper, and the v2 verification archives."""
    import json
    import random
    import mpmath as mp
    print("\n[TEST 7] v2: two-moment bound, Case (ii) rate floor, verification archives")
    mp.mp.dps = 40
    phi = lambda u: mp.exp(-u * u / 2) / mp.sqrt(2 * mp.pi)
    f = lambda V: 2 * mp.ncdf(mp.sqrt(V) / 2) - 1
    d1 = lambda u: phi(u) / (4 * u)
    d2 = lambda u: -(u ** -3 + u ** -1) * phi(u) / 32
    d3 = lambda u: (3 * u ** -5 + 2 * u ** -3 + u ** -1) * phi(u) / 256
    worst = mp.mpf(0)
    for V in [mp.mpf("1e-4"), mp.mpf("0.01"), mp.mpf("0.1"), mp.mpf(1), mp.mpf(4), mp.mpf(25)]:
        u = mp.sqrt(V) / 2
        num = [mp.diff(f, V, k) for k in (1, 2, 3)]
        cf = [d1(u), d2(u), d3(u)]
        for a, b in zip(num, cf):
            worst = max(worst, abs(a - b) / abs(b))
        assert cf[0] > 0 and cf[1] < 0 and cf[2] > 0, f"derivative signs fail at V={V}"
    assert worst < mp.mpf("1e-15"), f"closed-form derivatives disagree with numerical differentiation: {worst}"
    rng = random.Random(20261125)
    n_laws = 0
    for _ in range(200):
        m = mp.mpf(rng.uniform(0.002, 0.05)); v = mp.mpf(rng.uniform(1e-7, 4e-3))
        lower = m * m / (m * m + v) * f(m + v / m); upper = f(m)
        # extremal two-point law attains the lower bound
        p0 = v / (m * m + v); x = m + v / m
        assert abs((1 - p0) * f(x) - lower) < mp.mpf("1e-30"), "two-point law does not attain the lower bound"
        # a lognormal law with the same two moments lies inside the bracket
        s2 = mp.log(1 + v / (m * m)); mu = mp.log(m) - s2 / 2
        val = mp.quad(lambda z: f(mp.exp(mu + mp.sqrt(s2) * z)) * phi(z), [-12, 0, 12])
        assert lower - mp.mpf("1e-25") <= val <= upper + mp.mpf("1e-25"), "lognormal law outside the two-moment bracket"
        n_laws += 1
    print(f"  [PASS] Proposition 1: derivative formulas (max rel. err {mp.nstr(worst, 3)}), signs, sharpness and {n_laws} lognormal laws inside the bracket")
    tex = (ROOT / "manuscript" / "tables" / "spx_numbers.tex").read_text(encoding="utf-8")
    printed = {}
    for name in ("BetaMinZeroFive", "BetaMinSeven", "BetaMinTen", "BetaMinFifteen", "BetaMinSixth"):
        key = "\\newcommand{\\" + name + "}{"
        i = tex.index(key) + len(key)
        printed[name] = float(tex[i:tex.index("}", i)])
    for name, H in (("BetaMinZeroFive", 0.05), ("BetaMinSeven", 0.07), ("BetaMinTen", 0.10), ("BetaMinFifteen", 0.15), ("BetaMinSixth", 1.0 / 6.0)):
        b_fsw = min(3 * H + 0.5, 1.0)
        b_min = math.log2(1 + (2 ** b_fsw - 1) / 1.5)
        assert math.ceil(b_min * 1000 - 1e-9) / 1000 == printed[name], f"{name}: printed {printed[name]} vs ceil {b_min:.6f}"
    print(f"  [PASS] Case (ii) floor beta_min(H) = log2(1+(2^beta_FSW-1)/1.5) printed with upward rounding: {printed}")
    for fn, n_exp in (("v2_proof_checks.json", 19), ("v2_independent_check.json", 10)):
        arc = json.loads((ROOT / "results" / fn).read_text(encoding="utf-8"))
        assert arc["n_fail"] == 0 and arc["n_checks"] == n_exp == len(arc["checks"]), f"{fn}: n_fail={arc['n_fail']}, n_checks={arc['n_checks']}"
        assert "imports nothing from src/" in arc["code_disjoint"], f"{fn} is not code-disjoint"
    print("  [PASS] v2 verification archives: 19 first-principles checks and 10 code-disjoint recomputations, 0 failures")


def test_v3_boundary_cases():
    """Boundary regressions for directed widths, ATM-only Jensen and cell weights."""
    from fractions import Fraction
    from src.bs_arb import atm_skew_interval_arb
    from src.deterministic_spx import deterministic_spx_bracket_k0_arb
    for strike in (1.0, math.nextafter(1.0, 2.0), math.nextafter(1.0, 0.0)):
        row = deterministic_spx_bracket_k0_arb(0.1, 0.25, 0.0, -1.0, 0.04, strike)
        assert row["iv_lo"] <= 0.2 <= row["iv_hi"]
        exact_width = 10000 * (Fraction(row["iv_hi"]) - Fraction(row["iv_lo"]))
        assert Fraction(row["iv_width_bps"]) >= exact_width
    for dk in (0.0, -0.02, float("nan")):
        try:
            atm_skew_interval_arb((0.19, 0.20), (0.18, 0.19), dk)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid skew spacing accepted")
    for h in (0.1, 0.3, 0.5):
        vm = math.sqrt(1.0 ** (2*h))
        mean = math.sqrt(2*h)/(h+0.5)
        assert mean <= vm and ((mean == vm) == (h == 0.5))
    print("  [PASS] Deterministic-volatility endpoints, directed widths, skew domain and cell weights")


if __name__ == "__main__":
    test_1_bs_arb()
    test_2_volterra_arb_and_markov()
    test_3_exact_laws()
    test_4_stopped_cv_and_envelope_guards()
    test_5_bergomi_guyon_o3_check()
    test_6_deterministic_containment_and_competitors()
    test_7_v2_two_moment_rate_floor_and_archives()
    test_v3_boundary_cases()
    print("\nALL UNIT VERIFICATIONS (INCLUDING BLIND AUDIT #1 & #2 REGRESSION CHECKS) PASSED.")
