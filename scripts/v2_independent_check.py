"""Code-disjoint recomputation of representative RoughCert-SPX results (v2 audit).

This script imports NOTHING from ``src/`` and re-derives, from first principles,
  I1  E2 deterministic brackets (two-moment rho = 0 and Put-Lipschitz k = 0, mpmath);
  I2  E2 25 bps boundaries (exact-formula width at the stored certified points);
  I3  E3 BLP hybrid E_J, price gap and IV mapping (closed-form D(s), mpmath);
  I4  E1 Heston and classical Stein-Stein prices (own Riccati ODE / CF + Lewis formula);
  I5  RoughCert-SPX-CE level means (own Cholesky/projection, scrambled Sobol, new seeds);
  I6  E6 VIX provenance against the RoughCert v3.0.0 and v4 result files (read only; folders given by the
    environment variables ROUGHCERT_V3_RESULTS and ROUGHCERT_V4_RESULTS);
  I7  robustness of the E4 Regime A/B conclusions against the refined n_L = 1024 references.
Reads results/*.json; writes results/v2_independent_check.json; exits 1 on any failure.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import cholesky, cho_solve, cho_factor
from scipy.special import hyp2f1, ndtr, ndtri
from scipy.stats import qmc
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
mp.mp.dps = 30
CHECKS: list[dict] = []


def record(cid, what, metric, value, tol, ok, **extra):
    CHECKS.append({"id": cid, "what": what, "metric": metric, "value": float(value), "tolerance": float(tol), "pass": bool(ok), **extra})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {what} | {metric} = {float(value):.4g} (tol {tol:.3g})", flush=True)


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ Black-Scholes in mpmath and float
def mbs_call(F, K, V):
    F, K, V = mp.mpf(F), mp.mpf(K), mp.mpf(V)
    if V <= 0:
        return max(F - K, mp.mpf(0))
    s = mp.sqrt(V); d1 = mp.log(F / K) / s + s / 2
    return F * mp.ncdf(d1) - K * mp.ncdf(d1 - s)


def miv(price, K, T):
    price = mp.mpf(price); K = mp.mpf(K); T = mp.mpf(T)
    intrinsic = max(1 - K, mp.mpf(0))
    if price <= intrinsic:
        return mp.mpf(0)
    if price >= 1:
        return mp.inf
    f = lambda sg: mbs_call(1, K, sg * sg * T) - price
    return mp.findroot(f, (mp.mpf("1e-6"), mp.mpf(5)), solver="anderson")


def fbs_call(F, K, V):
    F = np.asarray(F, float); V = np.maximum(np.asarray(V, float), 1e-300)
    s = np.sqrt(V); d1 = np.log(F / K) / s + 0.5 * s
    return F * ndtr(d1) - K * ndtr(d1 - s)


def fiv(price, K, T):
    intrinsic = max(1.0 - K, 0.0)
    if price <= intrinsic + 1e-16:
        return 0.0
    return brentq(lambda sg: float(fbs_call(1.0, K, sg * sg * T)) - price, 1e-6, 5.0, xtol=1e-14)


# ------------------------------------------------------------------ RL covariance (own implementation)
def R_mp(H, s, u):
    H, s, u = mp.mpf(H), mp.mpf(s), mp.mpf(u)
    if s <= 0 or u <= 0:
        return mp.mpf(0)
    a, b = min(s, u), max(s, u)
    if a == b:
        return a ** (2 * H)
    return 2 * H / (H + mp.mpf(1) / 2) * a ** (H + mp.mpf(1) / 2) * b ** (H - mp.mpf(1) / 2) * mp.hyp2f1(1, mp.mpf(1) / 2 - H, H + mp.mpf(3) / 2, a / b)


def R_np(H, s, u):
    s, u = np.broadcast_arrays(np.asarray(s, float), np.asarray(u, float))
    a = np.minimum(s, u); b = np.maximum(s, u)
    out = np.zeros(a.shape)
    pos = a > 0
    z = np.where(pos, a / np.where(b > 0, b, 1.0), 0.0)
    with np.errstate(all="ignore"):
        val = 2 * H / (H + 0.5) * a ** (H + 0.5) * b ** (H - 0.5) * hyp2f1(1.0, 0.5 - H, H + 1.5, z)
    out = np.where(pos, val, 0.0)
    eq = pos & (np.abs(b - a) <= 1e-14 * b)
    return np.where(eq, a ** (2 * H), out)


# ------------------------------------------------------------------ I1 / I2  E2 deterministic brackets
def var_IT_exact(H, T, eta, xi0):
    f = lambda s, u: mp.expm1(mp.mpf(eta) ** 2 * R_mp(H, s, u))
    inner = lambda u: mp.quad(lambda s: f(s, u), [0, u])
    return 2 * mp.mpf(xi0) ** 2 * mp.quad(inner, [0, T])


def twomoment_bracket(H, T, eta, xi0):
    m = mp.mpf(xi0) * T; v = var_IT_exact(H, T, eta, xi0)
    phi1 = lambda V: mbs_call(1, 1, V)
    return m * m / (m * m + v) * phi1(m + v / m), phi1(m)


def EJ0_exact(H, T, eta, xi0):
    H, T, eta, xi0 = mp.mpf(H), mp.mpf(T), mp.mpf(eta), mp.mpf(xi0)
    return mp.sqrt(mp.quad(lambda s: xi0 * (-mp.expm1(-eta ** 2 / 4 * s ** (2 * H))), [0, T * mp.mpf("1e-6"), T]))


def k0_bracket(H, T, eta, rho, xi0, K):
    rho = mp.mpf(rho); K = mp.mpf(K)
    m = mp.mpf(xi0) * T; EJ = EJ0_exact(H, T, eta, xi0); Ihat = m - EJ ** 2
    Cdet = mbs_call(1, K, Ihat)
    rb = mp.sqrt(1 - rho * rho)
    gap = K * (abs(rho) + rb / mp.sqrt(2 * mp.pi)) * EJ + K * rho * rho / 2 * EJ * (mp.sqrt(m) + mp.sqrt(Ihat))
    lo, hi = Cdet - gap, Cdet + gap
    if rho == 0 and K == 1:
        hi = min(hi, mbs_call(1, 1, m))
    lo = max(lo, max(1 - K, mp.mpf(0))); hi = min(hi, mp.mpf(1))
    return lo, hi


def check_e2() -> None:
    e2 = load("e2_det_phase_map.json")
    worst_cont = mp.mpf(0); worst_tight = 0.0; rows = []
    for sp in e2["spotlights"]:
        H, T, eta, rho, xi0 = sp["H"], sp["T"], sp["eta"], sp["rho"], sp["xi0"]
        if rho == 0.0:
            lo, hi = twomoment_bracket(H, T, eta, xi0)
        else:
            lo, hi = k0_bracket(H, T, eta, rho, xi0, 1.0)
        ivlo, ivhi = miv(lo, 1, T), miv(hi, 1, T)
        # stored Arb bracket must contain the exact-formula bracket
        cont = max(mp.mpf(sp["det_price_lo"]) - lo, hi - mp.mpf(sp["det_price_hi"]))
        worst_cont = max(worst_cont, cont)
        w_exact = float(1e4 * (ivhi - ivlo)); w_arb = sp["det_iv_width_bps"]
        worst_tight = max(worst_tight, (w_arb - w_exact) / w_exact)
        rows.append({"id": sp["id"], "exact_width_bps": w_exact, "arb_width_bps": w_arb, "relative_overhead": (w_arb - w_exact) / w_exact})
        print(f"   {sp['id']}: exact width {w_exact:.4f} bps vs Arb {w_arb:.4f} bps", flush=True)
    record("I1", "E2 spotlights: stored 176-bit Arb brackets contain the exact-formula brackets (mpmath)",
           "max containment violation in price (must be <= 0)", worst_cont, 0.0, worst_cont <= 0,
           max_relative_width_overhead=worst_tight, rows=rows)
    # I2: exact width at the stored certified boundary points (H = 0.10 and 0.30)
    rows2 = []; ok = True
    for b in e2["boundaries"]:
        if round(b["H"], 2) not in (0.10, 0.30):
            continue
        H = b["H"]; T = e2["T_base"]; xi0 = e2["xi0_base"]
        for key, rho in (("eta_eff_star_rho0_2m_25bps", 0.0), ("eta_eff_star_rho_m09_k0_25bps", -0.9)):
            eta = b[key] / T ** H
            lo, hi = twomoment_bracket(H, T, eta, xi0) if rho == 0.0 else k0_bracket(H, T, eta, rho, xi0, 1.0)
            w = float(1e4 * (miv(hi, 1, T) - miv(lo, 1, T)))
            rows2.append({"H": H, "boundary": key, "eta_eff": b[key], "exact_width_bps": w})
            ok = ok and w <= 25.0
            print(f"   boundary H={H:.2f} {key}: exact width at stored point = {w:.4f} bps", flush=True)
    record("I2", "E2 boundaries: exact-formula ATM width at every stored certified point is <= 25 bps",
           "max exact width (bps)", max(r["exact_width_bps"] for r in rows2), 25.0, ok, rows=rows2)


# ------------------------------------------------------------------ I3  E3 BLP hybrid gap
def blp_EJ_exact(H, T, eta, xi0, n, kappa):
    H, T, eta, xi0 = mp.mpf(H), mp.mpf(T), mp.mpf(eta), mp.mpf(xi0)
    dt = T / n; p = H + mp.mpf(1) / 2; sq = mp.sqrt(2 * H)
    c = [None] + [sq * dt ** (H - mp.mpf(1) / 2) * (mp.mpf(k) ** p - mp.mpf(k - 1) ** p) / p for k in range(1, n + 1)]
    cell = lambda k: dt ** (2 * H) * ((mp.mpf(k) ** (2 * H) - mp.mpf(k - 1) ** (2 * H)) - 2 * H * ((mp.mpf(k) ** p - mp.mpf(k - 1) ** p) / p) ** 2)
    Dgrid = [mp.mpf(0)]
    for k in range(1, n + 1):
        Dgrid.append(Dgrid[-1] + (cell(k) if k > kappa else 0))
    total = mp.mpf(0)
    for k in range(1, n + 1):
        a = (k - 1) * dt
        if k <= kappa:
            total += 2 * xi0 * dt * (-mp.expm1(-eta ** 2 / 8 * Dgrid[k - 1]))
            continue
        ck = c[k]
        Dpart = lambda s: Dgrid[k - 1] + (s ** (2 * H) - a ** (2 * H)) - 2 * ck * sq * (s ** p - a ** p) / p + ck * ck * (s - a)
        total += mp.quad(lambda s: 2 * xi0 * (-mp.expm1(-eta ** 2 / 8 * Dpart(s))), [a, a + dt])
    return mp.sqrt(total)


def check_e3() -> None:
    e3 = load("e3_kernel_audit.json")
    P = e3["params"]; T, eta, rho, xi0 = P["T"], P["eta"], P["rho"], P["xi0"]
    ref = e3["ref_H010_bridge"]; plo, phi = ref["continuum_price_lo"][2], ref["continuum_price_hi"][2]
    ivlo, ivhi = ref["iv_continuum_lo"][2], ref["iv_continuum_hi"][2]
    rows = []; worst = 0.0; worst_iv = 0.0
    rb = math.sqrt(1 - rho * rho)
    for r in e3["blp_hybrid_audit"]:
        if abs(r["H"] - 0.10) > 1e-12 or r["n"] != 256 or r["kappa"] not in (1, 2, 4, 8):
            continue
        EJ = blp_EJ_exact(0.10, T, eta, xi0, 256, r["kappa"])
        m = xi0 * T
        gap = float((abs(rho) + rb / mp.sqrt(2 * mp.pi)) * EJ + rho * rho / 2 * EJ * 2 * mp.sqrt(m))
        inside = r["E_J_lo"] <= float(EJ) <= r["E_J_hi"]
        rel_gap = (r["arb_price_gap_upper_atm"] - gap) / gap
        worst = max(worst, 0.0 if inside and rel_gap >= 0 else 1.0)
        def ivmap(g):
            return 1e4 * max(ivlo - float(miv(max(plo - g, 1e-12), 1, T)), float(miv(min(phi + g, 1.0), 1, T)) - ivhi)
        iv_gap = ivmap(gap)                                   # mapping of the exact (midpoint) gap
        iv_gap_arb = ivmap(r["arb_price_gap_upper_atm"])      # mapping of the stored Arb upper gap
        # the stored IV gap must equal the mapping of the Arb gap, and dominate the mapping of the exact gap
        worst_iv = max(worst_iv, abs(iv_gap_arb - r["arb_iv_marginal_gap_upper_bps_atm"]))
        if iv_gap > r["arb_iv_marginal_gap_upper_bps_atm"] + 1e-9:
            worst = 1.0
        rows.append({"kappa": r["kappa"], "E_J_exact": float(EJ), "E_J_arb": [r["E_J_lo"], r["E_J_hi"]], "gap_exact": gap,
                     "gap_arb": r["arb_price_gap_upper_atm"], "iv_gap_exact_bps": iv_gap, "iv_gap_from_arb_gap_bps": iv_gap_arb,
                     "iv_gap_stored_bps": r["arb_iv_marginal_gap_upper_bps_atm"]})
        print(f"   BLP kappa={r['kappa']}: E_J exact {float(EJ):.6f} in [{r['E_J_lo']:.6f},{r['E_J_hi']:.6f}] ? {inside}; gap exact {gap:.6f} vs Arb {r['arb_price_gap_upper_atm']:.6f}; IV gap (exact) {iv_gap:.3f} <= (Arb) {iv_gap_arb:.3f} vs stored {r['arb_iv_marginal_gap_upper_bps_atm']:.3f}", flush=True)
    record("I3", "E3 BLP hybrid (H = 0.10, n = 256): exact E_J inside the Arb ball, Arb price gap >= exact gap, IV mapping reproduced",
           "failures (0 = none); max |IV-mapping difference| (bps, tol 0.005) reported separately", worst, 0.0, worst == 0.0 and worst_iv < 0.005,
           max_iv_gap_abs_diff_bps=worst_iv, rows=rows)
    # Markov rows: consistency of the stored E_J with the terminal L2 error via the Mehler bound, and gap formula
    worst_m = 0.0
    for r in e3["markov_audit"]:
        if abs(r["H"] - 0.10) > 1e-12:
            continue
        L2 = r["kernel_L2_hi"]
        bound = math.sqrt(2 * xi0 * T * (-math.expm1(-eta * eta / 8 * L2 * L2)))
        gap = (abs(rho) + rb / math.sqrt(2 * math.pi)) * r["E_J_hi"] + rho * rho / 2 * r["E_J_hi"] * 2 * math.sqrt(xi0 * T)
        worst_m = max(worst_m, max(r["E_J_lo"] - bound, 0.0), abs(gap - r["arb_price_gap_upper_atm"]) / gap)
    record("I3b", "E3 Markov (H = 0.10): E_J <= sqrt(2 xi0 T (1 - exp(-eta^2 ||K-K_M||^2/8))) and gap = Theorem 2(E_J)",
           "max violation / relative gap mismatch", worst_m, 1e-9, worst_m < 1e-9)


# ------------------------------------------------------------------ I4  E1 Heston and classical Stein-Stein
def lewis_prices(cf_vec, strikes, umax=400.0, nu=20000):
    u = np.linspace(1e-9, umax, nu)
    phi = cf_vec(u - 0.5j)
    out = []
    for K in strikes:
        integ = np.real(np.exp(-1j * u * math.log(K)) * phi) / (u * u + 0.25)
        out.append(1.0 - math.sqrt(K) / math.pi * np.trapezoid(integ, u))
    return np.array(out)


def heston_cf_vec(T, v0, kap, th, sv, rho):
    def cf(u):
        iu = 1j * u
        d = np.sqrt((rho * sv * iu - kap) ** 2 + sv * sv * (iu + u * u))
        g = (kap - rho * sv * iu - d) / (kap - rho * sv * iu + d)
        e = np.exp(-d * T)
        C = kap * th / sv ** 2 * ((kap - rho * sv * iu - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (kap - rho * sv * iu - d) / sv ** 2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0)
    return cf


def ss_cf_vec(T, s0, nu, rho, steps=4000):
    """sigma_t = s0 + nu W_t (H = 1/2, no mean reversion): phi(u) = exp(a(T) + c(T) s0^2 / 2) with
    c' = nu^2 c^2 + 2 rho nu i u c - (i u + u^2), a' = nu^2 c / 2, b = 0 (derived from Feynman-Kac); RK4 in tau."""
    def cf(u):
        iu = 1j * u
        c = np.zeros_like(u, dtype=complex); a = np.zeros_like(u, dtype=complex)
        h = T / steps
        fc = lambda cc: nu * nu * cc * cc + 2 * rho * nu * iu * cc - (iu + u * u)
        for _ in range(steps):
            k1 = fc(c); k2 = fc(c + h / 2 * k1); k3 = fc(c + h / 2 * k2); k4 = fc(c + h * k3)
            a += h / 6 * nu * nu / 2 * (c + 2 * (c + h / 2 * k1) + 2 * (c + h / 2 * k2) + (c + h * k3))
            c += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.exp(a + 0.5 * c * s0 * s0)
    return cf


def check_e1() -> None:
    e1 = load("e1_exact_laws.json")
    he = e1["heston_exact"]; strikes = he["strikes"]
    ph = lewis_prices(heston_cf_vec(0.5, 0.04, 2.0, 0.04, 0.4, -0.7), strikes)
    dev_h = float(np.max(np.abs(ph - np.array(he["prices"]))))
    record("I4a", "E1 Heston (v0 = theta = 0.04, kappa = 2, sigma_v = 0.4, rho = -0.7, T = 0.5): independent CF + Lewis",
           "max |price difference|", dev_h, 5e-8, dev_h < 5e-8)
    ss = e1["stein_stein_H05_exact"]
    ps = lewis_prices(ss_cf_vec(ss["T"], ss["sigma0"], ss["nu"], ss["rho"]), ss["strikes"], umax=300.0, nu=6000)
    dev_s = float(np.max(np.abs(ps - np.array(ss["prices"]))))
    record("I4b", "E1 classical Stein-Stein (sigma_t = 0.2 + 0.25 W_t, rho = -0.7, T = 0.5): own Riccati RK4 + Lewis",
           "max |price difference|", dev_s, 5e-7, dev_s < 5e-7)


# ------------------------------------------------------------------ I5  RoughCert-SPX-CE level means
class CEModel:
    """Own construction: G = (dW_0..dW_{n-1}, Y_{t_1}..Y_{t_n}); I_CE uses E[Y_s | G] at 2 Gauss-Legendre nodes per step."""

    def __init__(self, H, T, n):
        self.H, self.T, self.n = H, T, n
        dt = T / n; self.dt = dt
        t = dt * np.arange(n + 1)
        cw = math.sqrt(2 * H) / (H + 0.5)
        ty = t[1:]
        SWW = dt * np.eye(n)
        SYY = R_np(H, ty[:, None], ty[None, :])
        cyw = lambda s, j: cw * (np.maximum(s - t[j], 0) ** (H + 0.5) - np.maximum(s - t[j + 1], 0) ** (H + 0.5))
        SYW = cyw(ty[:, None], np.arange(n)[None, :])
        S = np.block([[SWW, SYW.T], [SYW, SYY]])
        self.L = cholesky(S + 1e-14 * np.eye(2 * n), lower=True)
        gl = np.array([0.5 - 0.5 / math.sqrt(3), 0.5 + 0.5 / math.sqrt(3)])
        sq = (t[:-1, None] + dt * gl[None, :]).ravel()
        self.wq = np.full(sq.size, dt / 2)
        C_sW = cyw(sq[:, None], np.arange(n)[None, :])
        C_sY = R_np(H, sq[:, None], ty[None, :])
        C = np.hstack([C_sW, C_sY])
        cf = cho_factor(S + 1e-14 * np.eye(2 * n), lower=True)
        self.A = cho_solve(cf, C.T).T          # (2n, 2n): Yhat(sq) = A @ G
        self.varhat = np.einsum("ij,ij->i", self.A, C)
        self.t_left = t[:-1]

    def prices(self, Zbatch, eta, rho, xi0, strikes):
        n, H = self.n, self.H
        G = Zbatch @ self.L.T
        dW, Y = G[:, :n], G[:, n:]
        Yl = np.hstack([np.zeros((G.shape[0], 1)), Y[:, :-1]])
        vl = xi0 * np.exp(eta * Yl - 0.5 * eta * eta * self.t_left[None, :] ** (2 * H))
        F = np.exp(rho * np.sum(np.sqrt(vl) * dW, axis=1) - 0.5 * rho * rho * self.dt * np.sum(vl, axis=1))
        Yh = G @ self.A.T
        Ice = (xi0 * np.exp(eta * Yh - 0.5 * eta * eta * self.varhat[None, :])) @ self.wq
        V = (1 - rho * rho) * Ice
        return np.stack([fbs_call(F, K, V) for K in strikes], axis=1)


def ce_estimate(H, T, n, eta, rho, xi0, strikes, n_scr=32, m_pow=11, seed0=777000):
    mdl = CEModel(H, T, n)
    means = []
    for r in range(n_scr):
        u = qmc.Sobol(d=2 * n, scramble=True, seed=seed0 + r).random_base2(m=m_pow)
        Z = ndtri(np.clip(u, 1e-15, 1 - 1e-15))
        acc = np.zeros(len(strikes)); cnt = 0
        for b in range(0, Z.shape[0], 1024):
            P = mdl.prices(Z[b:b + 1024], eta, rho, xi0, strikes)
            acc += P.sum(axis=0); cnt += P.shape[0]
        means.append(acc / cnt)
    means = np.array(means)
    return means.mean(axis=0), means.std(axis=0, ddof=1) / math.sqrt(n_scr)


def check_ce() -> None:
    e4 = load("e4_benchmarks.json"); ref = e4["regime_A"]["ref_bridge"]
    strikes = ref["strikes"]; lv = ref["levels"]; li = lv.index(128)
    m, se = ce_estimate(0.10, 0.25, 128, 1.9, -0.9, 0.04, strikes, seed0=777000)
    z = np.abs(m - np.array(ref["level_means"][li])) / np.sqrt(se ** 2 + np.array(ref["level_ses"][li]) ** 2)
    record("I5a", "RC-CE Regime A, n = 128: independent implementation (65,536 scrambled-Sobol paths, no CV) vs stored level mean (131,072 paths, 6-CV)",
           "max |z| over 8 strikes", float(np.max(z)), 3.5, float(np.max(z)) < 3.5, z=list(map(float, z)))
    e6 = load("e6_joint_spx_vix.json")
    row = [r for r in e6["joint_rows"] if r["model_id"] == "Pure_BDG6_rho-0.70"][0]["spx_bridge"]
    li = row["levels"].index(256)
    eta = 1.0 / math.sqrt(0.2)
    m6, se6 = ce_estimate(0.10, 0.25, 256, eta, -0.70, 0.235 ** 2, row["strikes"], seed0=778000)
    z6 = np.abs(m6 - np.array(row["level_means"][li])) / np.sqrt(se6 ** 2 + np.array(row["level_ses"][li]) ** 2)
    record("I5b", "RC-CE E6 Pure BDG6 rho = -0.70, n = 256: independent implementation vs stored level mean",
           "max |z| over 8 strikes", float(np.max(z6)), 3.5, float(np.max(z6)) < 3.5, z=list(map(float, z6)))


# ------------------------------------------------------------------ I6  E6 VIX provenance
def check_provenance() -> None:
    e6 = load("e6_joint_spx_vix.json")
    import hashlib
    import struct
    source = RES / "companion_v5"
    expected = {'vix_futures_full_suite.json': 'bced05332246d85e6b3cf0d1f25047da4ad02e5a787a697a2b45b464d7065699', 'vix_options_smile_suite.json': '73a7cf9458b360d20f4e153710d21441a02640ea8cfb9ecd2b76d1d19ba9ab0a'}
    files = {n: (source / n).read_bytes() for n in expected}
    assert all(hashlib.sha256(files[n]).hexdigest() == h for n, h in expected.items())
    fut, opt = (json.loads(files[n]) for n in expected)
    base = next(x for x in fut["block_a_e14_15cells"] if x["id"] == "base_H010_3m")
    keys = {"pure_bdg6": "bdg2023_fig6", "mixed_scen1": "mixed_rbergomi_smile"}
    count = 0
    for small, full in keys.items():
        a = e6["paper1_vix_smiles"][small]
        b = opt[full]["strikes"]
        assert len(a) == len(b) == (9 if small == "pure_bdg6" else 7)
        for x, y in zip(a,b):
            assert x.keys() == y.keys()
            for k in x:
                assert json.dumps(x[k]) == json.dumps(y[k])
        count += len(a)
    assert len(e6["joint_rows"]) == 9
    for row in e6["joint_rows"]:
        model = row["model_id"]
        if model.startswith("Pure_E14"):
            bounds = [base["best_L"],base["best_U"]]
            assert row["paper1_vix_smile_slope_pp"] is None
        else:
            suite = opt["bdg2023_fig6" if model.startswith("Pure_BDG6") else "mixed_rbergomi_smile"]
            bounds = [suite["condvar_future_L"],suite["condvar_future_U"]]
            strikes = {x["log_moneyness"]:x for x in suite["strikes"]}
            a, b = strikes[-0.3], strikes[0.3]
            # Sterbenz subtraction is exact for these comparable positive endpoints.
            from fractions import Fraction
            slope = [Fraction(b["iv_L_pct"])-Fraction(a["iv_U_pct"]), Fraction(b["iv_U_pct"])-Fraction(a["iv_L_pct"])]
            assert [Fraction(x) for x in row["paper1_vix_smile_slope_pp"]] == slope
        assert struct.pack(">2d",*bounds) == struct.pack(">2d",*row["paper1_vix_fut_arb"])
    record("I6", "E6 futures, complete smile records and derived spreads match RoughCert v5.0.0",
           "maximum endpoint difference", 0.0, 0.0, True, n_smile_points_compared=count, sha256=expected)


# ------------------------------------------------------------------ I7  E4 conclusions vs refined references
def check_robustness() -> None:
    e4 = load("e4_benchmarks.json"); ref2 = load("v2_diag_refinement.json")["cases"]
    rows = []
    for reg, key in (("regime_A", "regime_A"), ("regime_B", "regime_B")):
        c = ref2[key]
        lo = np.array(c["iv_continuum_lo"]); hi = np.array(c["iv_continuum_hi"])
        for e in e4[reg]["entries"]:
            v = np.array(e["iv_mid"], float)
            dist = np.maximum(np.maximum(lo - v, v - hi), 0.0) * 1e4
            mx = np.maximum(np.abs(v - lo), np.abs(v - hi)) * 1e4
            rows.append({"regime": reg, "method": e["method"], "max_smile_err_lo_bps_v1": e["max_smile_err_lo_bps"],
                         "max_smile_err_lo_bps_refined": float(dist.max()), "max_smile_err_hi_bps_refined": float(mx.max())})
    ce = [r for r in rows if r["method"].startswith("RoughCert-SPX-CE(n=256)") and r["regime"] == "regime_A"]
    jen = [r for r in rows if r["method"].startswith("RoughCert-SPX-CE-Jensen(n=256)")]
    bg = [r for r in rows if r["method"].startswith("BergomiGuyon")]
    ok = all(r["max_smile_err_lo_bps_refined"] > 250 for r in bg)
    record("I7", "E4 conclusions re-evaluated against the refined (n_L = 1024) Regime A/B references",
           "min BG2012 max-smile lower error vs refined reference (bps; > 250 keeps the diagnostic conclusion)",
           min(r["max_smile_err_lo_bps_refined"] for r in bg), 250.0, ok, rows=rows,
           ce256_regA=ce, jensen256=jen)


def main() -> None:
    t0 = time.perf_counter()
    allf = (check_e2, check_e3, check_e1, check_ce, check_provenance, check_robustness)
    only = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--only=")]
    sel = [f for f in allf if not only or f.__name__ in only[0].split(",")]
    for fn in sel:
        try:
            fn()
        except Exception as exc:  # fail loudly but keep the report
            record(fn.__name__, "exception", "raised", float("nan"), 0.0, False, error=repr(exc))
    if only and (RES / "v2_independent_check.json").exists():   # merge re-run checks into the stored report
        prev = json.loads((RES / "v2_independent_check.json").read_text(encoding="utf-8"))
        new_ids = {c["id"] for c in CHECKS}
        merged = [c for c in prev["checks"] if c["id"] not in new_ids and c["id"] not in [f.__name__ for f in sel]]
        CHECKS[:0] = merged
        CHECKS.sort(key=lambda c: c["id"])
    n_fail = sum(1 for c in CHECKS if not c["pass"])
    out = {"script": "scripts/v2_independent_check.py", "code_disjoint": "imports nothing from src/",
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "mpmath": mp.__version__},
           "n_checks": len(CHECKS), "n_fail": n_fail, "checks": CHECKS, "elapsed_s": time.perf_counter() - t0}
    (RES / "v2_independent_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(CHECKS)} checks, {n_fail} failed, {out['elapsed_s']:.0f}s")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
