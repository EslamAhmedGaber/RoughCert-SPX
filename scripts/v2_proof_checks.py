"""Independent numerical stress tests of the theoretical statements of RoughCert-SPX (v2 audit).

Code-disjoint by construction: this file imports NOTHING from ``src/``. Every quantity is
re-derived from first principles with mpmath (>= 40 significant digits), SymPy, or (for the exact
Gaussian conditioning of Proposition 2) NumPy/SciPy linear algebra written from scratch here.
A numerical check is evidence, not a proof; each entry names the statement it stresses.

Writes results/v2_proof_checks.json and exits with status 1 if any check fails.
"""
from __future__ import annotations

import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.linalg import cholesky, solve_triangular
from scipy.special import hyp2f1

ROOT = Path(__file__).resolve().parents[1]
mp.mp.dps = 40
RNG = random.Random(20261124)
CHECKS: list[dict] = []


def record(cid: str, ref: str, statement: str, metric: str, value: float, tol: float, ok: bool, **extra) -> None:
    CHECKS.append({"id": cid, "ref": ref, "statement": statement, "metric": metric,
                   "value": float(value), "tolerance": float(tol), "pass": bool(ok), **extra})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid} ({ref}): {metric} = {float(value):.3e} (tol {tol:.1e})", flush=True)


# ----------------------------------------------------------------------------------------------
# Black-Scholes helpers in mpmath (undiscounted, forward F, strike K, total variance V)
# ----------------------------------------------------------------------------------------------
def ncdf(x):
    return mp.ncdf(x)


def npdf(x):
    return mp.npdf(x)


def bs_call(F, K, V):
    if V <= 0:
        return max(F - K, mp.mpf(0))
    s = mp.sqrt(V)
    d1 = mp.log(F / K) / s + s / 2
    return F * ncdf(d1) - K * ncdf(d1 - s)


def bs_put(F, K, V):
    return bs_call(F, K, V) - F + K


def bs_vega_V(F, K, V):
    """dC/dV (derivative with respect to total variance)."""
    s = mp.sqrt(V)
    d1 = mp.log(F / K) / s + s / 2
    return F * npdf(d1) / (2 * s)


# ----------------------------------------------------------------------------------------------
# C1  Theorem 1(iii): exact Mehler identity for Gaussian-kernel surrogates
# ----------------------------------------------------------------------------------------------
def check_mehler() -> None:
    gh_x, gh_w = zip(*[(x, w) for x, w in zip(*np.polynomial.hermite_e.hermegauss(80))])
    worst = mp.mpf(0)
    for _ in range(20):
        a = mp.mpf(RNG.uniform(0.05, 1.5)); ah = mp.mpf(RNG.uniform(0.05, 1.5))
        c = mp.mpf(RNG.uniform(-0.95, 0.95)) * mp.sqrt(a * ah)
        eta = mp.mpf(RNG.uniform(0.2, 2.5)); xi = mp.mpf(RNG.uniform(0.01, 0.09))
        # (Y, Yh) = (sqrt(a) Z1, c/sqrt(a) Z1 + sqrt(ah - c^2/a) Z2)
        b1 = c / mp.sqrt(a); b2 = mp.sqrt(ah - c * c / a)
        acc = mp.mpf(0)
        for x1, w1 in zip(gh_x, gh_w):
            for x2, w2 in zip(gh_x, gh_w):
                y = mp.sqrt(a) * x1; yh = b1 * x1 + b2 * x2
                sv = mp.sqrt(xi) * mp.e ** (eta * y / 2 - eta ** 2 * a / 4)
                svh = mp.sqrt(xi) * mp.e ** (eta * yh / 2 - eta ** 2 * ah / 4)
                acc += mp.mpf(w1) * mp.mpf(w2) * (sv - svh) ** 2
        acc /= (2 * mp.pi)
        formula = 2 * xi * (1 - mp.e ** (-eta ** 2 / 8 * (a + ah - 2 * c)))
        worst = max(worst, abs(acc - formula) / formula)
    record("C1", "Theorem 1(iii)", "E[(sqrt(v)-sqrt(vhat))^2] = 2 xi0 (1 - exp(-eta^2 Var(Y-Yhat)/8))",
           "max relative deviation (80x80 Gauss-Hermite, 20 random cases)", worst, 1e-12, worst < 1e-12)


# ----------------------------------------------------------------------------------------------
# C2  Theorem 1(ii) and Theorem 2: Minkowski coupling and global Put-Lipschitz bound
# ----------------------------------------------------------------------------------------------
def check_ledger_inequalities() -> None:
    worst_ii = mp.mpf(-1); worst_ei = mp.mpf(-1)
    for _ in range(200):
        n_sc = RNG.randint(2, 6); n_t = RNG.randint(2, 8)
        p = [mp.mpf(RNG.random()) for _ in range(n_sc)]; s = sum(p); p = [x / s for x in p]
        sig = [[mp.mpf(RNG.uniform(0, 0.6)) for _ in range(n_t)] for _ in range(n_sc)]
        sigh = [[mp.mpf(RNG.uniform(0, 0.6)) for _ in range(n_t)] for _ in range(n_sc)]
        dt = mp.mpf(1) / n_t
        I = [dt * sum(x * x for x in row) for row in sig]; Ih = [dt * sum(x * x for x in row) for row in sigh]
        EJ2 = sum(pk * dt * sum((a - b) ** 2 for a, b in zip(r1, r2)) for pk, r1, r2 in zip(p, sig, sigh))
        lhs = mp.sqrt(sum(pk * (mp.sqrt(a) - mp.sqrt(b)) ** 2 for pk, a, b in zip(p, I, Ih)))
        worst_ii = max(worst_ii, lhs - mp.sqrt(EJ2))
        EI = sum(pk * abs(a - b) for pk, a, b in zip(p, I, Ih))
        bound = mp.sqrt(EJ2) * (mp.sqrt(sum(pk * a for pk, a in zip(p, I))) + mp.sqrt(sum(pk * b for pk, b in zip(p, Ih))))
        worst_ei = max(worst_ei, EI - bound)
    ok = worst_ii <= 0 and worst_ei <= 0
    record("C2a", "Theorem 1(ii)", "||sqrt(I)-sqrt(Ihat)||_2 <= E_J and E_I <= E_J (sqrt(E I) + sqrt(E Ihat))",
           "max violation over 200 random discrete laws (must be <= 0)", max(worst_ii, worst_ei), 0.0, ok)
    worst = mp.mpf(-1); worst_du = mp.mpf(-1); worst_ds = mp.mpf(-1)
    for _ in range(2000):
        K = mp.mpf(RNG.uniform(0.3, 3.0))
        u0, u1 = mp.mpf(RNG.uniform(-1.5, 1.5)), mp.mpf(RNG.uniform(-1.5, 1.5))
        S0, S1 = mp.mpf(RNG.uniform(0.0, 1.2)), mp.mpf(RNG.uniform(0.0, 1.2))
        lhs = abs(bs_put(mp.e ** u1, K, S1 ** 2) - bs_put(mp.e ** u0, K, S0 ** 2))
        rhs = K * abs(u1 - u0) + K / mp.sqrt(2 * mp.pi) * abs(S1 - S0)
        worst = max(worst, lhs - rhs)
        if S0 > mp.mpf("1e-6"):
            du = mp.diff(lambda uu: bs_put(mp.e ** uu, K, S0 ** 2), u0)
            ds = mp.diff(lambda ss: bs_put(mp.e ** u0, K, ss ** 2), S0)
            worst_du = max(worst_du, abs(du) - K)
            worst_ds = max(worst_ds, ds - K / mp.sqrt(2 * mp.pi))
    ok = worst <= 0 and worst_du <= 0 and worst_ds <= 0
    record("C2b", "Theorem 2", "|P(e^u,K,S^2)-P(e^u',K,S'^2)| <= K|u-u'| + K|S-S'|/sqrt(2 pi); |dP/du|<=K; dP/dS<=K/sqrt(2pi)",
           "max violation over 2000 random pairs (must be <= 0)", max(worst, worst_du, worst_ds), 0.0, ok)


# ----------------------------------------------------------------------------------------------
# C3  Proposition 1: derivative signs and the sharp two-moment bracket
# ----------------------------------------------------------------------------------------------
def phi1(V):
    return 2 * ncdf(mp.sqrt(V) / 2) - 1 if V > 0 else mp.mpf(0)


def check_twomoment() -> None:
    V = sp.symbols("V", positive=True)
    f = 2 * (sp.Rational(1, 2) * (1 + sp.erf(sp.sqrt(V) / 2 / sp.sqrt(2)))) - 1
    u = sp.sqrt(V) / 2
    phi = sp.exp(-u ** 2 / 2) / sp.sqrt(2 * sp.pi)
    stated = [phi / (4 * u), -(u ** -3 + u ** -1) / 32 * phi, (3 * u ** -5 + 2 * u ** -3 + u ** -1) / 256 * phi]
    sym_ok = all(sp.simplify(sp.diff(f, V, k + 1) - stated[k]) == 0 for k in range(3))
    record("C3a", "Proposition 1 proof", "phi_1', phi_1'', phi_1''' equal the stated closed forms (SymPy)",
           "symbolic residual is zero (1 = yes)", 1.0 if sym_ok else 0.0, 0.5, sym_ok)
    # Positivity from the closed form verified in C3a (all three terms positive), cross-checked against
    # numerical differentiation where the derivative is resolvable (V <= 50; e^{-V/8} underflows beyond).
    grid = [mp.mpf(10) ** (mp.mpf(x) / 4) for x in range(-24, 13)]

    def closed3(VV):
        uu = mp.sqrt(VV) / 2
        return (3 * uu ** -5 + 2 * uu ** -3 + uu ** -1) / 256 * npdf(uu)

    vals = [closed3(VV) for VV in grid]
    rel = max(abs(mp.diff(phi1, VV, 3) - closed3(VV)) / closed3(VV) for VV in grid if VV <= 50)
    record("C3b", "Proposition 1", "third derivative of phi_1 positive on V in [1e-6, 1e3] (closed form, 37-point log grid); numerical third derivative agrees for V <= 50",
           "min third derivative (closed form, must be > 0)", min(vals), 0.0, min(vals) > 0 and rel < 1e-20,
           max_rel_numeric_vs_closed=float(rel))
    worst_lo = mp.mpf(-1); worst_hi = mp.mpf(-1); worst_eq = mp.mpf(0)
    for _ in range(300):
        k = RNG.randint(2, 7)
        pts = [mp.mpf(RNG.uniform(0, 0.4)) for _ in range(k)]
        if RNG.random() < 0.3:
            pts[0] = mp.mpf(0)
        w = [mp.mpf(RNG.random()) for _ in range(k)]; s = sum(w); w = [x / s for x in w]
        m = sum(a * b for a, b in zip(w, pts)); v = sum(a * b * b for a, b in zip(w, pts)) - m * m
        if m <= 0:
            continue
        E = sum(a * phi1(b) for a, b in zip(w, pts))
        L = m * m / (m * m + v) * phi1(m + v / m)
        worst_lo = max(worst_lo, L - E); worst_hi = max(worst_hi, E - phi1(m))
        xs = m + v / m
        E2 = (m / xs) * phi1(xs)  # extremal law {0, x*}
        worst_eq = max(worst_eq, abs(E2 - L))
    ok = worst_lo <= mp.mpf("1e-35") and worst_hi <= mp.mpf("1e-35") and worst_eq < mp.mpf("1e-35")
    record("C3c", "Proposition 1", "m^2/(m^2+v) phi_1(m+v/m) <= E phi_1(I) <= phi_1(m), equality for the two-point law",
           "max violation over 300 random discrete laws (must be <= 0)", max(worst_lo, worst_hi), 1e-35, ok,
           extremal_equality_error=float(worst_eq))
    m = mp.mpf("0.01"); vals = [m * m / (m * m + v) * phi1(m + v / m) for v in [mp.mpf(i) * mp.mpf("1e-6") for i in range(0, 200)]]
    mono = all(b < a for a, b in zip(vals[:-1], vals[1:]))
    record("C3d", "Proposition 1", "L(v) strictly decreasing in v >= 0", "monotone on 200-point grid (1 = yes)",
           1.0 if mono else 0.0, 0.5, mono)
    # Theorem 3(a) at K=1: phi1(m) - L(v) = (v/m)(phi1(m)/m - phi1'(m)) + O(v^2)
    m = mp.mpf("0.02"); ratios = []
    for v in [mp.mpf("1e-6"), mp.mpf("1e-7"), mp.mpf("1e-8")]:
        gap = phi1(m) - m * m / (m * m + v) * phi1(m + v / m)
        ratios.append(gap / ((v / m) * (phi1(m) / m - mp.diff(phi1, m))))
    err = abs(ratios[-1] - 1)
    record("C3e", "Theorem 3(a), K = 1", "two-moment gap is first order in Var(I_T): ratio to (v/m)(phi1(m)/m - phi1'(m)) -> 1",
           "|ratio - 1| at v = 1e-8", err, 1e-4, err < 1e-4, ratios=[float(r) for r in ratios])


# ----------------------------------------------------------------------------------------------
# C4  Theorem 3(b): leverage constant (closed form, Stein-lemma derivative, Heston cross-check)
# ----------------------------------------------------------------------------------------------
def heston_cf(u, T, v0, kap, th, sv, rho):
    iu = 1j * u
    d = mp.sqrt((rho * sv * iu - kap) ** 2 + sv * sv * (iu + u * u))
    g = (kap - rho * sv * iu - d) / (kap - rho * sv * iu + d)
    e = mp.e ** (-d * T)
    C = (kap * th / sv ** 2) * ((kap - rho * sv * iu - d) * T - 2 * mp.log((1 - g * e) / (1 - g)))
    D = ((kap - rho * sv * iu - d) / sv ** 2) * (1 - e) / (1 - g * e)
    return mp.e ** (C + D * v0)


def heston_call(K, T, v0, kap, th, sv, rho):
    k = mp.log(K)
    f = lambda u: mp.re(mp.e ** (-1j * u * k) * heston_cf(u - 0.5j, T, v0, kap, th, sv, rho)) / (u * u + mp.mpf(1) / 4)
    return 1 - mp.sqrt(K) / mp.pi * mp.quad(f, [0, 5, 20, 80, mp.inf])


def check_leverage() -> None:
    worst = mp.mpf(0)
    for _ in range(6):
        H = mp.mpf(RNG.uniform(0.03, 0.45)); T = mp.mpf(RNG.uniform(0.05, 2)); xi = mp.mpf(RNG.uniform(0.01, 0.09))
        rho = mp.mpf(RNG.uniform(-1, 0)); eta = mp.mpf(RNG.uniform(0.1, 2))
        Kf = lambda x: mp.sqrt(2 * H) * x ** (H - mp.mpf(1) / 2)
        inner = lambda s: mp.quad(lambda uu: Kf(s - uu), [0, s])
        num = rho * eta * mp.sqrt(xi) * xi * mp.quad(inner, [0, T]) / (xi * T)
        closed = rho * eta * mp.sqrt(xi) * mp.sqrt(2 * H) * T ** (H + mp.mpf(1) / 2) / ((H + mp.mpf(1) / 2) * (H + mp.mpf(3) / 2))
        worst = max(worst, abs(num - closed) / abs(closed))
    record("C4a", "Theorem 3(b) proof", "flat xi0: C^{x xi}/I0 = rho eta sqrt(xi0) sqrt(2H) T^{H+1/2}/((H+1/2)(H+3/2))",
           "max relative deviation (6 random cases)", worst, 1e-15, worst < 1e-15)
    worst = mp.mpf(0)
    for _ in range(8):
        K = mp.mpf(RNG.uniform(0.7, 1.4)); I0 = mp.mpf(RNG.uniform(0.002, 0.2))
        def g(x):
            F = mp.e ** x; s = mp.sqrt(I0); d1 = mp.log(F / K) / s + s / 2
            return mp.e ** (2 * x) * npdf(d1) / (F * s)
        lhs = mp.diff(g, 0)
        rhs = 2 * bs_vega_V(1, K, I0) / I0 * (mp.log(K) + I0 / 2)
        worst = max(worst, abs(lhs - rhs) / max(abs(rhs), mp.mpf("1e-30")))
    record("C4b", "Theorem 3(b) proof", "d/dx[e^{2x} Gamma_BS(e^x,K,I0)]|_0 = 2 V_BS(1,K,I0)(log K + I0/2)/I0",
           "max relative deviation (8 random cases)", worst, 1e-25, worst < 1e-25)
    # Generic first-order (Bergomi-Guyon) leverage term checked on exact Heston prices at small vol-of-vol
    mp.mp.dps = 30
    T = mp.mpf("0.5"); th = v0 = mp.mpf("0.04"); kap = mp.mpf(2); rho = mp.mpf("-0.7")
    I0 = th * T
    rows = []
    for K in [mp.mpf("0.9"), mp.mpf(1), mp.mpf("1.1")]:
        for sv in [mp.mpf("0.02"), mp.mpf("0.01"), mp.mpf("0.005")]:
            Cxxi = rho * sv * v0 * (T / kap - (1 - mp.e ** (-kap * T)) / kap ** 2)
            first = bs_vega_V(1, K, I0) * Cxxi / I0 * (mp.log(K) + I0 / 2)
            exact = heston_call(K, T, v0, kap, th, sv, rho) - bs_call(1, K, I0)
            rows.append({"K": float(K), "sigma_v": float(sv), "ratio": float(exact / first)})
    mp.mp.dps = 40
    # First-order exactness: |ratio - 1| must vanish linearly in sigma_v (halve when sigma_v halves).
    halving = []
    for Kv in (0.9, 1.0, 1.1):
        e01 = abs([r for r in rows if abs(r["K"] - Kv) < 1e-12 and r["sigma_v"] == 0.01][0]["ratio"] - 1)
        e005 = abs([r for r in rows if abs(r["K"] - Kv) < 1e-12 and r["sigma_v"] == 0.005][0]["ratio"] - 1)
        halving.append(e005 / e01)
    dev = max(abs(h - 0.5) for h in halving)
    record("C4c", "Theorem 3(b)", "first-order leverage term V_BS (C^{x xi}/I0)(log K + I0/2) is exact at first order on Heston: |exact/first-order - 1| = O(sigma_v)",
           "max |err(0.005)/err(0.01) - 1/2| over K = 0.9, 1, 1.1", dev, 0.02, dev < 0.02, table=rows, halving_ratios=halving)


# ----------------------------------------------------------------------------------------------
# C5  Proposition 2(i): asymptotic constant, and exact G_i deficits for the calibrated regime
# ----------------------------------------------------------------------------------------------
def check_jensen_constant() -> None:
    H = mp.mpf("0.1"); eta = mp.mpf("1.9"); T = mp.mpf("0.25"); c = eta ** 2 / 4
    rows = []
    for k in [4, 6, 8, 10, 12, 14, 16]:
        n = 2 ** k; dt = T / n
        S = n * mp.quad(lambda u: 1 - mp.e ** (-c * u ** (2 * H)), [0, dt])
        lead = eta ** 2 * T / (4 * (2 * H + 1)) * dt ** (2 * H)
        rows.append({"n": n, "ratio": float(S / lead)})
    # ratio -> 1 with error O(dt^{2H}) -> check monotone convergence and |ratio-1| at n=65536 below 0.1
    err = abs(rows[-1]["ratio"] - 1)
    mono = all(abs(a["ratio"] - 1) > abs(b["ratio"] - 1) for a, b in zip(rows[:-1], rows[1:]))
    record("C5a", "Proposition 2(i)", "xi0-free lower bound / [eta^2 T dt^{2H}/(4(2H+1))] -> 1 with O(dt^{2H}) error",
           "|ratio - 1| at n = 65,536 (monotone decrease required)", err, 0.1, err < 0.1 and mono, table=rows)


def _rl_cov(H, s, u):
    s = np.asarray(s, float); u = np.asarray(u, float)
    mn = np.minimum(s, u); mx = np.maximum(s, u)
    out = np.zeros(np.broadcast(mn, mx).shape)
    pos = mn > 0
    z = np.where(pos, mn / np.where(mx > 0, mx, 1), 0)
    with np.errstate(all="ignore"):
        hyp = hyp2f1(1.0, 0.5 - H, H + 1.5, z)
        val = 2 * H / (H + 0.5) * mn ** (H + 0.5) * mx ** (H - 0.5) * hyp
    diag = pos & (np.abs(mx - mn) <= 1e-15 * np.maximum(mx, 1e-300))
    out = np.where(pos, val, 0.0)
    out = np.where(diag, mn ** (2 * H), out)
    return out


def exact_jensen_deficit(H: float, T: float, eta: float, n: int, n_gl: int) -> float:
    """Exact E[I_T - I_ad]/E[I_T] for sigma_i = cell average of E[sqrt(v_s) | G_i] (n_gl Gauss-Legendre nodes),
    G_i = sigma(dW_0..dW_{i-1}, Y_{t_1}..Y_{t_i}); flat xi0 cancels. Interleaved ordering makes every G_i a prefix."""
    dt = T / n
    t = dt * np.arange(n + 1)
    # interleaved vector: (dW_0, Y_{t_1}, dW_1, Y_{t_2}, ..., dW_{n-1}, Y_{t_n})
    N = 2 * n
    is_w = np.arange(N) % 2 == 0
    idx = np.arange(N) // 2
    # covariance blocks
    C = np.zeros((N, N))
    cw = math.sqrt(2 * H) / (H + 0.5)

    def cov_yw(ty, j):  # Cov(Y_ty, dW_j) over [t_j, t_{j+1}]
        a = np.maximum(ty - t[j], 0.0); b = np.maximum(ty - t[j + 1], 0.0)
        return cw * (a ** (H + 0.5) - b ** (H + 0.5))

    wi = np.where(is_w)[0]; yi = np.where(~is_w)[0]
    C[np.ix_(wi, wi)] = dt * np.eye(n)
    ty = t[idx[yi] + 1]
    C[np.ix_(yi, yi)] = _rl_cov(H, ty[:, None], ty[None, :])
    cyw = cov_yw(ty[:, None], idx[wi][None, :])
    C[np.ix_(yi, wi)] = cyw; C[np.ix_(wi, yi)] = cyw.T
    sd = np.sqrt(np.diag(C)); Cn = C / sd[:, None] / sd[None, :]
    L = cholesky(Cn + 1e-14 * np.eye(N), lower=True)
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_gl)
    acc = 0.0
    for i in range(n):
        s = t[i] + dt * (gl_x + 1) / 2
        w = gl_w / 2
        if i == 0:
            cov_hat = np.zeros((n_gl, n_gl))
        else:
            P = 2 * i
            c = np.zeros((P, n_gl))
            for q in range(n_gl):
                col = np.empty(P)
                pw = np.where(is_w[:P])[0]; py = np.where(~is_w[:P])[0]
                col[pw] = cov_yw(s[q], idx[pw])
                col[py] = _rl_cov(H, s[q], t[idx[py] + 1])
                c[:, q] = col / sd[:P]
            Z = solve_triangular(L[:P, :P], c, lower=True)
            cov_hat = Z.T @ Z
        s2 = s ** (2 * H)
        M = np.exp(-eta ** 2 / 8 * (s2[:, None] + s2[None, :]) + eta ** 2 / 4 * cov_hat)
        acc += dt * float(w @ M @ w)
    return 1.0 - acc / T


def check_jensen_exact() -> dict:
    H, T, eta, rho = 0.10, 0.25, 1.9, -0.9
    out = {}
    for n in [32, 64, 128, 256, 512]:
        d2 = exact_jensen_deficit(H, T, eta, n, 2)
        d8 = exact_jensen_deficit(H, T, eta, n, 8)
        d16 = exact_jensen_deficit(H, T, eta, n, 16) if n in (32, 512) else None
        lb = float(n * mp.quad(lambda u: 1 - mp.e ** (-mp.mpf(eta) ** 2 / 4 * u ** (2 * mp.mpf(H))), [0, mp.mpf(T) / n]) / mp.mpf(T))
        out[str(n)] = {"deficit_gl2": d2, "deficit_gl8": d8, "deficit_gl16": d16, "lower_bound_eq": lb,
                       "rho_eff_gl2": rho * math.sqrt(1 - d2), "rho_eff_gl8": rho * math.sqrt(1 - d8)}
        print(f"   n={n}: deficit GL2={d2:.5f} GL8={d8:.5f} lower bound={lb:.5f} rho_eff(GL2)={rho*math.sqrt(1-d2):.4f}", flush=True)
    ok = all(v["deficit_gl8"] >= v["lower_bound_eq"] - 1e-9 for v in out.values())
    record("C5b", "Proposition 2(i)", "exact deficit (Gaussian conditioning on G_i) >= closed-form lower bound (eq:jensen_deficit)",
           "min (exact GL8 deficit - lower bound) over n = 32..512", min(v["deficit_gl8"] - v["lower_bound_eq"] for v in out.values()),
           0.0, ok, table=out)
    return out


# ----------------------------------------------------------------------------------------------
# C6  Lemma (stopped timer): one-step Black-Scholes semigroup identity
# ----------------------------------------------------------------------------------------------
def check_semigroup() -> None:
    worst = mp.mpf(0)
    for _ in range(25):
        F = mp.mpf(RNG.uniform(0.5, 1.8)); K = mp.mpf(RNG.uniform(0.5, 1.8))
        Vtot = mp.mpf(RNG.uniform(0.005, 0.5)); dq = Vtot * mp.mpf(RNG.uniform(0.0, 1.0))
        f = lambda z: bs_call(F * mp.e ** (mp.sqrt(dq) * z - dq / 2), K, Vtot - dq) * npdf(z)
        lhs = mp.quad(f, [-mp.inf, -4, 0, 4, mp.inf])
        worst = max(worst, abs(lhs - bs_call(F, K, Vtot)))
    record("C6", "Lemma (stopped-timer CV)", "E[C_BS(F e^{sqrt(dq) Z - dq/2}, K, V - dq)] = C_BS(F, K, V) for 0 <= dq <= V",
           "max absolute deviation (25 random cases)", worst, 1e-30, worst < 1e-30)


# ----------------------------------------------------------------------------------------------
# C7  Theorem 4: identifiability constant, Bonferroni quantiles, Case (ii) rate floor vs E5 rates
# ----------------------------------------------------------------------------------------------
def check_bridge_constants() -> dict:
    from scipy.stats import t as student_t
    se = 1 / math.log(2) * math.sqrt(1 / 9 + 1 / 9)
    record("C7a", "Theorem 4(i)", "delta-method SE of log2(|D_{L-1}|/|D_L|) at SNR = 3 for both increments",
           "SE (paper: 'exceeding 0.68' below SNR 3)", se, 0.68, se > 0.68)
    floors = {}
    for H in [0.05, 0.07, 0.10, 0.15, 0.25, 0.40]:
        bf = min(3 * H + 0.5, 1.0)
        floors[f"{H:.2f}"] = math.log2(1 + (2 ** bf - 1) / 1.5)
    e5 = json.loads((ROOT / "results" / "e5_weak_rates.json").read_text(encoding="utf-8"))
    rows = []
    ok = True
    for r in e5["results_by_H"]:
        h = f"{r['H']:.2f}"
        for m in ("leftpoint", "ce_i_only"):
            for key in ("call106_slope", "atm_slope", "strip_mean_slope"):
                bh = r["methods"][m][key]["beta_hat"]
                if bh is None:
                    continue
                rows.append({"H": r["H"], "method": m, "slope": key, "beta_hat": bh, "beta_min": floors[h]})
                ok = ok and bh > floors[h]
    # Case (ii) envelopes are reported at H = 0.10 (E1, E3, E4 A/C, E6) and H = 0.30 (spotlights S1, S3),
    # bracketed in E5 by H = 0.25 and 0.40 (same beta_FSW = 1). H = 0.05 is characterised, not graded.
    used = [x for x in rows if x["H"] >= 0.10 - 1e-12]
    margin_used = min(x["beta_hat"] - x["beta_min"] for x in used)
    below = [x for x in rows if x["beta_hat"] <= x["beta_min"]]
    record("C7b", "Theorem 4(ii) hypothesis support",
           "at every E5 Hurst index H >= 0.10 all measured rates (left-point and RC-CE; K = 1.06, ATM, strip mean) exceed beta_min(H) = log2(1 + (2^beta_FSW - 1)/1.5)",
           "min (beta_hat - beta_min) over E5 rows with H >= 0.10", margin_used, 0.0, margin_used > 0, beta_min=floors, rows=rows,
           below_floor_rows=below,
           t_quantiles={"R=16": float(student_t.ppf(1 - 0.05 / 4, 15)), "R=32": float(student_t.ppf(1 - 0.05 / 4, 31))})
    return floors


# ----------------------------------------------------------------------------------------------
# C8  BLP optimal weights and cell L2 error; C9 RL covariance closed form and monotonicity
# ----------------------------------------------------------------------------------------------
def check_blp_and_rl() -> None:
    worst = mp.mpf(0); worst_cell = mp.mpf(0)
    for _ in range(10):
        H = mp.mpf(RNG.uniform(0.03, 0.45)); k = RNG.randint(2, 40); dt = mp.mpf(RNG.uniform(0.001, 0.05))
        p = H + mp.mpf(1) / 2; al = H - mp.mpf(1) / 2
        Kf = lambda x: mp.sqrt(2 * H) * x ** al
        b = ((mp.mpf(k) ** p - mp.mpf(k - 1) ** p) / p) ** (1 / al)
        mean = mp.quad(Kf, [(k - 1) * dt, k * dt]) / dt
        worst = max(worst, abs(Kf(b * dt) - mean) / mean)
        cell = mp.quad(lambda x: (Kf(x) - mean) ** 2, [(k - 1) * dt, k * dt])
        closed = dt ** (2 * H) * ((mp.mpf(k) ** (2 * H) - mp.mpf(k - 1) ** (2 * H)) - 2 * H * ((mp.mpf(k) ** p - mp.mpf(k - 1) ** p) / p) ** 2)
        worst_cell = max(worst_cell, abs(cell - closed) / closed)
    record("C8", "Section 3.1 (BLP hybrid)", "K(b_k* dt) equals the cell mean; cell L2 error closed form",
           "max relative deviation (10 random cases)", max(worst, worst_cell), 1e-20, max(worst, worst_cell) < 1e-20)
    worst = mp.mpf(0); worst_mono = mp.mpf(-1)
    for _ in range(10):
        H = mp.mpf(RNG.uniform(0.03, 0.49)); s = mp.mpf(RNG.uniform(0.01, 1)); u = s + mp.mpf(RNG.uniform(0.001, 1))
        direct = 2 * H * mp.quad(lambda r: (s - r) ** (H - mp.mpf(1) / 2) * (u - r) ** (H - mp.mpf(1) / 2), [0, s])
        closed = 2 * H / (H + mp.mpf(1) / 2) * s ** (H + mp.mpf(1) / 2) * u ** (H - mp.mpf(1) / 2) * mp.hyp2f1(1, mp.mpf(1) / 2 - H, H + mp.mpf(3) / 2, s / u)
        worst = max(worst, abs(direct - closed) / closed)
        R = lambda a, b: 2 * H / (H + mp.mpf(1) / 2) * a ** (H + mp.mpf(1) / 2) * b ** (H - mp.mpf(1) / 2) * mp.hyp2f1(1, mp.mpf(1) / 2 - H, H + mp.mpf(3) / 2, a / b)
        dRs = mp.diff(lambda a: R(a, u), s); dRu = mp.diff(lambda b: R(s, b), u)
        worst_mono = max(worst_mono, -dRs, dRu)
    ok = worst < 1e-15 and worst_mono < 0
    record("C9", "Section 3.1 (Var(I_T) boxes)", "R_H closed form = direct integral; dR/ds > 0 and dR/du < 0 for s < u",
           "max relative deviation of closed form", worst, 1e-15, ok, max_monotonicity_violation=float(worst_mono))


# ----------------------------------------------------------------------------------------------
# C10  Section 2.2: optimal deterministic surrogate and comparison with the Wick surrogate
# ----------------------------------------------------------------------------------------------
def check_surrogates() -> None:
    worst = mp.mpf(0); worst_cmp = mp.mpf(-1)
    for _ in range(20):
        eta = mp.mpf(RNG.uniform(0.1, 2.5)); a = mp.mpf(RNG.uniform(0.01, 1.5)); xi = mp.mpf(RNG.uniform(0.01, 0.09))
        Esv = mp.quad(lambda y: mp.sqrt(xi) * mp.e ** (eta * y / 2 - eta ** 2 * a / 4) * npdf(y / mp.sqrt(a)) / mp.sqrt(a), [-mp.inf, 0, mp.inf])
        worst = max(worst, abs(Esv - mp.sqrt(xi) * mp.e ** (-eta ** 2 * a / 8)))
        opt = xi * (1 - mp.e ** (-eta ** 2 * a / 4)); wick = 2 * xi * (1 - mp.e ** (-eta ** 2 * a / 8))
        worst_cmp = max(worst_cmp, opt - wick)
    ok = worst < 1e-30 and worst_cmp <= 0
    record("C10", "Section 2.2", "E[sqrt(v_s)] = sqrt(xi0) exp(-eta^2 s^{2H}/8); optimal residual <= Wick-surrogate residual",
           "max |E sqrt(v) - closed form|", worst, 1e-30, ok, max_comparison_violation=float(worst_cmp))


def main() -> None:
    t0 = time.perf_counter()
    check_mehler()
    check_ledger_inequalities()
    check_twomoment()
    check_leverage()
    check_jensen_constant()
    jens = check_jensen_exact()
    check_semigroup()
    floors = check_bridge_constants()
    check_blp_and_rl()
    check_surrogates()
    n_fail = sum(1 for c in CHECKS if not c["pass"])
    out = {
        "script": "scripts/v2_proof_checks.py",
        "code_disjoint": "imports nothing from src/",
        "mpmath_dps": 40,
        "seed": 20261124,
        "environment": {"python": platform.python_version(), "mpmath": mp.__version__, "numpy": np.__version__, "sympy": sp.__version__},
        "n_checks": len(CHECKS),
        "n_fail": n_fail,
        "jensen_deficit_exact": jens,
        "case_ii_rate_floor": floors,
        "checks": CHECKS,
        "elapsed_s": time.perf_counter() - t0,
    }
    (ROOT / "results" / "v2_proof_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(CHECKS)} checks, {n_fail} failed, {out['elapsed_s']:.0f}s")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
