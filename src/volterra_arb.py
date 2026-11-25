"""Exact 176-bit Arb Volterra kernel calculus and Two-Field Ito-Mehler Error Ledger.

Implements Theorem 1 (Two-Field Ito-Mehler Error Ledger) and kernel-gap integrals:
1. Exact Riemann-Liouville covariance R_RL(s, u) = 2H int_0^{min(s,u)} (s-r)^{H-1/2}(u-r)^{H-1/2} dr
   in 176-bit Arb ball arithmetic (with Pfaff / connection formula near the diagonal).
2. Exact L^2 kernel approximation errors:
   - Markovian multifactor kernel K_M(u) = sum_{j=1}^M w_j exp(-x_j u) vs K(u) = sqrt(2H) u^{H-1/2}
     evaluated via closed-form incomplete gamma functions in 176-bit Arb.
   - BLP Hybrid kernel K_{kappa, n}(u) (exact on [0, kappa * dt], piecewise constant on [kappa * dt, T])
     evaluated via exact cell integrals in 176-bit Arb.
3. Exact Two-Field Ito-Mehler quantities:
   - E_J^2 = int_0^T E[(sqrt(v_s) - sigma_hat_s)^2] ds  (exact Ito isometry for J_T)
   - E_I^2 = E[(I_T - I_hat_T)^2]                      (exact Mehler double integral for I_T)
   - Residual variance share P_I / Var(I_T) for Gaussian projections of Y on [0, T].
"""
from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple

import numpy as np
from flint import arb, acb, ctx

from .bs_arb import set_dps, arb_interval, arb_lo_float, arb_hi_float, arb_to_bounds

set_dps()


def _check_rough_H_arb(H: arb) -> None:
    """Enforce strict domain 0 < H <= 1/2 required for RL kernel monotonicity & Arb bounds."""
    half = arb(1) / arb(2)
    if not (H > arb(0) and H <= half):
        raise ValueError(f"Rough-volatility RL bounds require 0 < H <= 1/2; got H={H}")


def rl_cov_arb(H: arb, s: arb, u: arb, is_diag: bool = False) -> arb:
    """Exact R_RL(s, u) = 2H int_0^{min(s,u)} (s-r)^{H-1/2}(u-r)^{H-1/2} dr for 0 < H <= 1/2."""
    _check_rough_H_arb(H)
    if s.is_zero() or u.is_zero():
        return arb(0)
    half = arb(1) / arb(2)
    if H == half:
        return s.min(u)
    if is_diag or (s is u):
        return s ** (arb(2) * H)
    # Exact point equality (zero-radius Arb balls)
    if s == u:
        return s ** (arb(2) * H)
    if s.overlaps(u):
        # Rigorous monotone enclosure on overlapping balls [s_lo, s_hi] x [u_lo, u_hi]:
        # Since R_RL(a, b) increases in a = min(s,u) and decreases in b = max(s,u) for H < 1/2:
        #   R_RL(min(s_lo, u_lo), max(s_hi, u_hi)) <= R_RL(s, u) <= (s_hi * u_hi)^H.
        s_lo, s_hi = arb(s.lower()), arb(s.upper())
        u_lo, u_hi = arb(u.lower()), arb(u.upper())
        m0 = s_lo if s_lo <= u_lo else u_lo
        M1 = s_hi if s_hi >= u_hi else u_hi
        if m0 == M1:
            return m0 ** (arb(2) * H)
        lo_val = arb(0) if m0.is_zero() else rl_cov_arb(H, m0, M1)
        hi_val = (s_hi * u_hi) ** H
        return arb_interval(lo_val, hi_val)
    if s > u:
        s, u = u, s

    half = arb(1) / arb(2)
    z = s / u
    if z <= half:
        hyp = z.hypgeom_2f1(arb(1), half - H, H + arb(3) / arb(2))
    else:
        omz = arb(1) - z
        term1 = ((H + half) / (arb(2) * H)) * omz.hypgeom_2f1(arb(1), half - H, arb(1) - arb(2) * H)
        B = (H + arb(3) / arb(2)).gamma() * (-arb(2) * H).gamma() / (half - H).gamma()
        term2 = B * (omz ** (arb(2) * H)) * (z ** (-(H + half)))
        hyp = term1 + term2
    return (arb(2) * H / (H + half)) * (s ** (H + half)) * (u ** (H - half)) * hyp


def rl_cov_float(H: float, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Vectorized float64 R_RL(s, u) using exact Gauss hypergeometric 2F1 (valid for 0 < H < 1/2).

    Used for fast Cholesky / bridge matrix assembly; exact Arb version used for all certificates.
    """
    from scipy.special import hyp2f1, gamma as sc_gamma

    s_a, u_a = np.broadcast_arrays(np.asarray(s, dtype=np.float64), np.asarray(u, dtype=np.float64))
    mn = np.minimum(s_a, u_a)
    mx = np.maximum(s_a, u_a)
    if abs(H - 0.5) < 1e-14:
        return np.maximum(mn, 0.0)
    out = np.zeros_like(mn, dtype=np.float64)

    pos = mn > 0.0
    if not np.any(pos):
        return out

    mn_p = mn[pos]
    mx_p = mx[pos]
    z = mn_p / mx_p

    # Exact diagonal / near-diagonal
    res = np.empty_like(z)
    diag = np.abs(1.0 - z) < 1e-14
    res[diag] = mn_p[diag] ** (2.0 * H)

    off = ~diag
    if np.any(off):
        zo = z[off]
        mno = mn_p[off]
        mxo = mx_p[off]
        small = zo <= 0.5
        hyp = np.empty_like(zo)
        if np.any(small):
            hyp[small] = hyp2f1(1.0, 0.5 - H, H + 1.5, zo[small])
        if np.any(~small):
            omz = 1.0 - zo[~small]
            t1 = ((H + 0.5) / (2.0 * H)) * hyp2f1(1.0, 0.5 - H, 1.0 - 2.0 * H, omz)
            B = sc_gamma(H + 1.5) * sc_gamma(-2.0 * H) / sc_gamma(0.5 - H)
            t2 = B * (omz ** (2.0 * H)) * (zo[~small] ** (-(H + 0.5)))
            hyp[~small] = t1 + t2
        res[off] = (2.0 * H / (H + 0.5)) * (mno ** (H + 0.5)) * (mxo ** (H - 0.5)) * hyp

    out[pos] = res
    return out


def rl_cross_cov_float(H: float, t_y: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Exact Cov(Y_{t_y}, W_{t1} - W_{t0}) = sqrt(2H)/(H+1/2) * [(t_y - t0)_+^{H+1/2} - (t_y - t1)_+^{H+1/2}]."""
    ty = np.asarray(t_y, dtype=np.float64)
    a = np.asarray(t0, dtype=np.float64)
    b = np.asarray(t1, dtype=np.float64)
    pref = math.sqrt(2.0 * H) / (H + 0.5)
    p = H + 0.5
    return pref * (np.maximum(ty - a, 0.0) ** p - np.maximum(ty - b, 0.0) ** p)


# =============================================================================
# Exact 176-bit Arb L^2 kernel gap for Markovian (SOE) approximations
# =============================================================================

def _lower_gamma_arb(a: arb, x: arb) -> arb:
    """Lower incomplete gamma function gamma(a, x) = int_0^x t^{a-1} e^{-t} dt for a > 0, x >= 0.

    In python-flint, x.gamma_lower(a) computes gamma(a, x) directly (self=z, arg=s).
    """
    if x.is_zero():
        return arb(0)
    return x.gamma_lower(a)


def markov_kernel_l2_sq_arb(
    H: arb,
    s: arb,
    nodes: Sequence[arb],
    weights: Sequence[arb],
) -> arb:
    """Exact int_0^s (K(u) - K_M(u))^2 du in 176-bit Arb ball arithmetic.

    Convention:
      K(u)   = sqrt(2H) * u^{H - 1/2}
      K_M(u) = sum_{j=1}^M w_j * exp(-x_j * u)
    Notice weights w_j here are ALREADY scaled to match K(u) = sqrt(2H) u^{H-1/2}
    (i.e. if raw weights approximate u^{H-1/2}/Gamma(H+1/2), multiply them by sqrt(2H)*Gamma(H+1/2)).
    """
    set_dps()
    if s.is_zero():
        return arb(0)
    M = len(nodes)
    assert len(weights) == M
    # Term 1: int_0^s K(u)^2 du = 2H * s^{2H} / (2H) = s^{2H}
    t1 = s ** (arb(2) * H)

    # Term 2: -2 int_0^s K(u) K_M(u) du = -2 sqrt(2H) sum_j w_j int_0^s u^{H-1/2} e^{-x_j u} du
    # With v = x_j u: int_0^s u^{a-1} e^{-x u} du = x^{-a} gamma(a, x s), where a = H + 1/2
    a = H + arb(1) / arb(2)
    sqrt2H = (arb(2) * H).sqrt()
    cross = arb(0)
    for xj, wj in zip(nodes, weights):
        if xj.is_zero():
            cross += wj * (s ** a) / a
        else:
            cross += wj * (xj ** (-a)) * _lower_gamma_arb(a, xj * s)
    t2 = -arb(2) * sqrt2H * cross

    # Term 3: int_0^s K_M(u)^2 du = sum_{j,k} w_j w_k (1 - exp(-(x_j+x_k)s)) / (x_j + x_k)
    t3 = arb(0)
    for j in range(M):
        xj, wj = nodes[j], weights[j]
        for k in range(M):
            xk, wk = nodes[k], weights[k]
            sjk = xj + xk
            if sjk.is_zero():
                integ = s
            else:
                integ = (-(-(sjk * s)).expm1()) / sjk
            t3 += wj * wk * integ

    res = t1 + t2 + t3
    # Nonnegative by definition
    if res.lower() < arb(0):
        res = arb_interval(arb(0), res.upper())
    return res


def markov_kernel_l1_arb(
    H: arb,
    T: arb,
    nodes: Sequence[arb],
    weights: Sequence[arb],
    n_scan: int = 1200,
) -> arb:
    """Rigorous 176-bit Arb enclosure of L^1 kernel error int_0^T |K(u) - K_M(u)| du.

    Uses monotone box enclosures on a dense log-spaced partition of [u_0, T]:
    Since BOTH K(u) = sqrt(2H) u^{H-1/2} and K_M(u) = sum_j w_j exp(-x_j u) (for w_j >= 0, x_j >= 0)
    are strictly DECREASING functions of u > 0:
      On ANY interval [a, b] with 0 < a < b:
        K(u)   in [K(b), K(a)]
        K_M(u) in [K_M(b), K_M(a)]
      Therefore f(u) = K(u) - K_M(u) lies rigorously in [K(b) - K_M(a), K(a) - K_M(b)].
      - Case 1 (Certified positive: K(b) - K_M(a) >= 0):
          int_a^b |f(u)| du == int_a^b (K(u) - K_M(u)) du   (EXACT closed-form antiderivative in Arb!)
      - Case 2 (Certified negative: K(a) - K_M(b) <= 0):
          int_a^b |f(u)| du == int_a^b (K_M(u) - K(u)) du   (EXACT closed-form antiderivative in Arb!)
      - Case 3 (Indeterminate / root bracket):
          Recursively bisect [a, b] (up to depth 16) to isolate each root cleanly;
          on the tiny residual root box [a, b], enclose |f(u)| rigorously via:
            0 <= int_a^b |f(u)| du <= int_a^b max(|K(b) - K_M(a)|, |K(a) - K_M(b)|) du.
    Plus on the singular corner [0, u_0] (choosing u_0 so K(u_0) > K_M(0) = sum w_j, where K(u) > K_M(u)
    IDENTICALLY on (0, u_0]!):
      Because K(u) -> +inf and K_M(u) <= sum_j w_j, for u_0 = min(T, (sqrt(2H)/(sum w_j))^{1/(1/2-H)})
      (if sum w_j > 0), we have f(u) >= 0 IDENTICALLY on (0, u_0], so the corner integral is ALSO
      EXACT closed-form int_0^{u_0} (K(u) - K_M(u)) du with ZERO corner slack!.

    rigorous 176-bit Arb enclosure with ZERO quadrature approximation error.
    """
    set_dps()
    _check_rough_H_arb(H)
    for wj, xj in zip(weights, nodes):
        if wj < arb(0) or xj < arb(0):
            raise ValueError("markov_kernel_l1_arb requires nonnegative weights and nodes.")

    sqrt2H = (arb(2) * H).sqrt()
    p = H + arb(1) / arb(2)
    hm = H - arb(1) / arb(2)

    def K_val(u: arb) -> arb:
        return sqrt2H * (u ** hm)

    def KM_val(u: arb) -> arb:
        acc = arb(0)
        for xj, wj in zip(nodes, weights):
            acc += wj * (-(xj * u)).exp()
        return acc

    def antideriv_diff(a: arb, b: arb) -> arb:
        """Exact int_a^b (K(u) - K_M(u)) du in 176-bit Arb."""
        k_int = (sqrt2H / p) * (b ** p - a ** p)
        km_int = arb(0)
        for xj, wj in zip(nodes, weights):
            if xj.is_zero():
                km_int += wj * (b - a)
            else:
                # int_a^b e^{-x u} du = e^{-x a} * (1 - e^{-x (b-a)}) / x
                km_int += wj * (-(xj * a)).exp() * (-(-(xj * (b - a))).expm1()) / xj
        return k_int - km_int

    # 1. Exact positive corner [0, u_0] where K(u) >= K_M(0) = sum w_j
    w_sum = sum(weights, arb(0))
    if w_sum.is_zero():
        return (sqrt2H / p) * (T ** p)
    if H < arb(1) / arb(2):
        # Solve sqrt(2H) u_0^{H-1/2} = 1.05 * w_sum -> strictly positive on (0, u_0].
        u0_cand = (sqrt2H / (arb("1.05") * w_sum)) ** (arb(1) / (arb(1) / arb(2) - H))
        u0 = u0_cand if u0_cand < T else T
        # Guard underflow to reasonable minimum
        u_floor = T * arb("1e-35")
        if u0 < u_floor:
            u0 = u_floor
            # On [0, u_floor] use conservative bound [|antideriv|, int(K + K_M)]
            k_int = (sqrt2H / p) * (u0 ** p)
            km_int = arb(0)
            for xj, wj in zip(nodes, weights):
                km_int += wj * (u0 if xj.is_zero() else (-(-(xj * u0)).expm1()) / xj)
            total = arb_interval(abs(k_int - km_int), k_int + km_int)
        else:
            # Verify K(u0) >= K_M(0) strictly so K(u) - K_M(u) >= 0 on ALL of (0, u0].
            if K_val(u0) >= w_sum:
                total = antideriv_diff(arb(0), u0)
            else:
                k_int = (sqrt2H / p) * (u0 ** p)
                km_int = arb(0)
                for xj, wj in zip(nodes, weights):
                    km_int += wj * (u0 if xj.is_zero() else (-(-(xj * u0)).expm1()) / xj)
                total = arb_interval(abs(k_int - km_int), k_int + km_int)
    else:
        u0 = arb(0)
        total = arb(0)

    if u0 >= T:
        return total

    # 2. Rigorous adaptive interval-certified integration on [u0, T]
    # Build log-spaced scan grid from u0 to T
    T_f = float(T.mid())
    u0_f = max(float(u0.mid()), 1e-35)
    grid_f = np.geomspace(u0_f, T_f, n_scan + 1)
    grid_a = [u0] + [arb(str(float(v))) for v in grid_f[1:-1]] + [T]

    def integrate_segment(a: arb, b: arb, ka: arb, kb: arb, ma: arb, mb: arb, depth: int) -> arb:
        # Since K and K_M are decreasing: f(u) in [kb - ma, ka - mb] on [a, b]
        f_lo = kb - ma
        f_hi = ka - mb
        if f_lo >= arb(0):
            return antideriv_diff(a, b)
        if f_hi <= arb(0):
            return -antideriv_diff(a, b)
        if depth >= 16:
            # Tiny root bracket: rigorous conservative enclosure using sound Arb ball .max()
            M_abs = abs(f_lo).max(abs(f_hi))
            ad = abs(antideriv_diff(a, b))
            return arb_interval(ad, (b - a) * M_abs)
        mid = (a + b) / arb(2)
        km = K_val(mid)
        mm = KM_val(mid)
        return (
            integrate_segment(a, mid, ka, km, ma, mm, depth + 1)
            + integrate_segment(mid, b, km, kb, mm, mb, depth + 1)
        )

    K_pts = [K_val(u) for u in grid_a]
    M_pts = [KM_val(u) for u in grid_a]
    for i in range(len(grid_a) - 1):
        total += integrate_segment(grid_a[i], grid_a[i + 1], K_pts[i], K_pts[i + 1], M_pts[i], M_pts[i + 1], 0)
    return total


# =============================================================================
# Exact 176-bit Arb L^2 kernel gap for BLP Hybrid scheme (kappa >= 1)
# =============================================================================

def blp_optimal_b_arb(k: int, H: arb) -> arb:
    """Optimal BLP (2017) evaluation point b_k^* for cell [k-1, k] (1-indexed k > kappa).

    For K(u) = sqrt(2H) u^{H-1/2}, the L^2-optimal constant approximation on [(k-1)dt, k*dt] is:
      c_k^* = (1/dt) int_{(k-1)dt}^{k*dt} K(u) du
            = sqrt(2H) * dt^{H-1/2} * (k^{H+1/2} - (k-1)^{H+1/2}) / (H + 1/2),
    which equals K(b_k^* * dt) with b_k^* = ((k^{H+1/2} - (k-1)^{H+1/2}) / (H+1/2))^{1/(H-1/2)}.
    """
    p = H + arb(1) / arb(2)
    alpha = H - arb(1) / arb(2)
    avg = (arb(k) ** p - arb(k - 1) ** p) / p
    return avg ** (arb(1) / alpha)


def blp_hybrid_kernel_l2_sq_arb(H: arb, T: arb, n: int, kappa: int, s_idx: int) -> arb:
    """Exact int_0^{s} (K(u) - K_{kappa,n}(u))^2 du at s = s_idx * (T/n) in 176-bit Arb.

    On [0, min(s_idx, kappa)*dt], K_{kappa,n} = K identically (zero error!).
    On each cell [(k-1)*dt, k*dt] for k = kappa+1, ..., s_idx:
      K_{kappa,n}(u) = c_k^* = mean of K on that cell (L^2-optimal BLP weight).
    Therefore the exact cell L^2 error is:
      int_{(k-1)dt}^{k dt} K(u)^2 du - dt * (c_k^*)^2
      = dt^{2H} * [ (k^{2H} - (k-1)^{2H}) - 2H * ((k^{H+1/2} - (k-1)^{H+1/2}) / (H+1/2))^2 ].
    """
    set_dps()
    if s_idx <= kappa:
        return arb(0)
    dt = T / arb(n)
    dt_2H = dt ** (arb(2) * H)
    p = H + arb(1) / arb(2)
    twoH = arb(2) * H
    acc = arb(0)
    for k in range(kappa + 1, s_idx + 1):
        t_sq = arb(k) ** twoH - arb(k - 1) ** twoH
        avg = (arb(k) ** p - arb(k - 1) ** p) / p
        cell_err = t_sq - twoH * (avg * avg)
        if cell_err.lower() < arb(0):
            cell_err = arb_interval(arb(0), cell_err.upper())
        acc += cell_err
    return dt_2H * acc


def blp_hybrid_EJ_arb(
    H: arb,
    T: arb,
    eta: arb,
    xi0_flat: arb,
    n: int,
    kappa: int,
) -> arb:
    """Certified 176-bit Arb upper enclosure of E_J(K_{kappa,n}) for continuous-time BLP hybrid kernel.

    Uses monotonicity: D(s) = int_0^s (K(u) - K_{kappa,n}(u))^2 du is nondecreasing in s.
    On each cell [t_{i}, t_{i+1}], D(s) in [D(t_i), D(t_{i+1})], giving a rigorous
    monotone-box Arb enclosure of:
      E_J^2 = int_0^T xi0 * (2 - 2 exp(-eta^2 / 8 * D(s))) ds.
    """
    set_dps()
    dt = T / arb(n)
    c_exp = -(eta * eta) / arb(8)
    # Precompute D(t_i) for i = 0..n incrementally in O(n)
    D_vals = [arb(0)] * (n + 1)
    dt_2H = dt ** (arb(2) * H)
    p = H + arb(1) / arb(2)
    twoH = arb(2) * H
    acc = arb(0)
    for k in range(1, n + 1):
        if k > kappa:
            t_sq = arb(k) ** twoH - arb(k - 1) ** twoH
            avg = (arb(k) ** p - arb(k - 1) ** p) / p
            cell_err = t_sq - twoH * (avg * avg)
            if cell_err.lower() < arb(0):
                cell_err = arb_interval(arb(0), cell_err.upper())
            acc += cell_err
        D_vals[k] = dt_2H * acc

    ej_sq_lo = arb(0)
    ej_sq_hi = arb(0)
    for i in range(n):
        f_lo = -arb(2) * (c_exp * D_vals[i]).expm1()
        f_hi = -arb(2) * (c_exp * D_vals[i + 1]).expm1()
        ej_sq_lo += xi0_flat * dt * f_lo
        ej_sq_hi += xi0_flat * dt * f_hi

    ej_sq = arb_interval(ej_sq_lo, ej_sq_hi)
    return ej_sq.sqrt()


def markov_EJ_arb(
    H: arb,
    T: arb,
    eta: arb,
    xi0_flat: arb,
    nodes: Sequence[arb],
    weights: Sequence[arb],
    n_boxes: int = 80,
) -> Tuple[arb, arb]:
    """Certified 176-bit Arb enclosure of E_J(K_M) and terminal L^2 kernel error ||K - K_M||_{L^2(0,T)}.

    Since D(s) = int_0^s (K(u) - K_M(u))^2 du is strictly nondecreasing in s, partitioning [0, T]
    into n_boxes graded intervals s_m = T * (m/n_boxes)^2 encloses the integral rigorously
    with zero quadrature error.
    """
    set_dps()
    c_exp = -(eta * eta) / arb(8)
    grid = [T * (arb(m) / arb(n_boxes)) ** 2 for m in range(n_boxes + 1)]
    D_grid = [markov_kernel_l2_sq_arb(H, s, nodes, weights) for s in grid]

    ej_sq_lo = arb(0)
    ej_sq_hi = arb(0)
    for m in range(n_boxes):
        ds = grid[m + 1] - grid[m]
        f_lo = -arb(2) * (c_exp * D_grid[m]).expm1()
        f_hi = -arb(2) * (c_exp * D_grid[m + 1]).expm1()
        ej_sq_lo += xi0_flat * ds * f_lo
        ej_sq_hi += xi0_flat * ds * f_hi

    ej_sq = arb_interval(ej_sq_lo, ej_sq_hi)
    return ej_sq.sqrt(), D_grid[-1].sqrt()


# =============================================================================
# Exact 176-bit Arb Time-Discretisation E_J for Left-Point and CE-Adapted
# =============================================================================

def time_EJ_ce_ideal_arb(
    H: arb,
    T: arb,
    eta: arb,
    xi0_flat: arb,
    n: int,
) -> arb:
    """Exact 176-bit Arb E_J for ideal continuous-past CE-adapted sigma_tilde_s = E[sqrt(v_s) | F_{t_i}].

    On each cell [t_i, t_{i+1}], D_i(s,s) = Var(Y_s | F_{t_i}) = (s - t_i)^{2H}, so:
      E[(sqrt(v_s) - sigma_tilde_s)^2] = xi0 * (1 - exp(-(eta^2 / 4) * (s - t_i)^{2H})).
    Integrated over n uniform cells of width dt = T/n via monotone box subdivision (rigorous Arb).
    """
    set_dps()
    dt = T / arb(n)
    c_exp = -(eta * eta) / arb(4)
    # Integral is n * xi0 * int_0^{dt} (1 - exp(c_exp * u^{2H})) du.
    # Integrand is strictly increasing in u on [0, dt]; graded monotone boxes give tight Arb enclosure.
    m_boxes = 128
    lo_sum = arb(0)
    hi_sum = arb(0)
    for j in range(m_boxes):
        u0 = dt * (arb(j) / arb(m_boxes)) ** 3
        u1 = dt * (arb(j + 1) / arb(m_boxes)) ** 3
        du = u1 - u0
        f0 = -(c_exp * (u0 ** (arb(2) * H))).expm1()
        f1 = -(c_exp * (u1 ** (arb(2) * H))).expm1()
        lo_sum += du * f0
        hi_sum += du * f1
    ej_sq = arb(n) * xi0_flat * arb_interval(lo_sum, hi_sum)
    return ej_sq.sqrt()


def time_EJ_leftpoint_arb(
    H: arb,
    T: arb,
    eta: arb,
    xi0_flat: arb,
    n: int,
    sub_boxes: int = 8,
) -> arb:
    """Certified 176-bit Arb E_J for left-point sigma_hat_s = sqrt(v_{t_i}) on [t_i, t_{i+1}).

    Exact integrand:
      E[(sqrt(v_s) - sqrt(v_{t_i}))^2] = xi0 * (2 - 2 exp(-(eta^2 / 8) * V_inc(t_i, s))),
    where V_inc(t_i, s) = Var(Y_s - Y_{t_i}) = s^{2H} + t_i^{2H} - 2 R_RL(t_i, s).
    Note: for 0 < H < 1/2, V_inc(t_i, s) is strictly increasing in s >= t_i:
      V_inc(t_i, s) = (s - t_i)^{2H} + 2H int_0^{t_i} ((s-r)^{H-1/2} - (t_i-r)^{H-1/2})^2 dr,
    and both terms increase monotonically with s! Hence box bounds on [s_j, s_{j+1}] are rigorous.
    """
    set_dps()
    dt = T / arb(n)
    c_exp = -(eta * eta) / arb(8)
    lo_total = arb(0)
    hi_total = arb(0)
    for i in range(n):
        ti = arb(i) * dt
        # Graded sub-boxes inside [t_i, t_{i+1}]
        pts = [ti + dt * (arb(j) / arb(sub_boxes)) ** 2 for j in range(sub_boxes + 1)]
        v_vals = []
        for s in pts:
            if s == ti:
                v_vals.append(arb(0))
            elif ti.is_zero():
                v_vals.append(s ** (arb(2) * H))
            else:
                v = s ** (arb(2) * H) + ti ** (arb(2) * H) - arb(2) * rl_cov_arb(H, ti, s)
                if v.lower() < arb(0):
                    v = arb_interval(arb(0), v.upper())
                v_vals.append(v)
        for j in range(sub_boxes):
            ds = pts[j + 1] - pts[j]
            f0 = -arb(2) * (c_exp * v_vals[j]).expm1()
            f1 = -arb(2) * (c_exp * v_vals[j + 1]).expm1()
            lo_total += xi0_flat * ds * f0
            hi_total += xi0_flat * ds * f1
    return arb_interval(lo_total, hi_total).sqrt()


# =============================================================================
# Exact Integrated-Variance Second Moment & Variance E[I_T^2], Var(I_T)
# =============================================================================

def exact_IT_moments_arb(
    H: arb,
    T: arb,
    eta: arb,
    xi0_flat: arb,
    n_boxes: int = 32,
) -> Tuple[arb, arb]:
    """Certified 176-bit Arb mean E[I_T] and variance Var(I_T) for flat xi0.

    E[I_T] = xi0 * T.
    Var(I_T) = xi0^2 * int_0^T int_0^T expm1(eta^2 R_RL(s, u)) ds du
             = 2 * xi0^2 * int_0^T int_0^u expm1(eta^2 R_RL(s, u)) ds du.
    Note: R_RL(s, u) increases in s on [0, u] and decreases in (u - s) >= 0,
    and for s <= u, R_RL(s, u) is strictly increasing in s and bounded above by s^{2H} (actually (s u)^H).
    Even better: on any box [s0, s1] x [u0, u1] with s1 <= u0:
      R_RL(s, u) is INCREASING in s and DECREASING in u.
    Proof: d/du R_RL(s, u) = 2H (H - 1/2) int_0^s (s-r)^{H-1/2} (u-r)^{H-3/2} dr < 0 for H < 1/2.
    And d/ds R_RL(s, u) > 0 because R_RL(s, u) = 2H int_0^s v^{H-1/2} (u - s + v)^{H-1/2} dv;
    differentiating in s gives positive boundary term + positive integrand term (since -(H-1/2) > 0).
    Thus on off-diagonal boxes s1 <= u0, min is at (s0, u1) and max is at (s1, u0) -- rigorous.
    On diagonal triangles s0 <= s <= u <= s1, 0 <= R_RL(s, u) <= u^{2H} <= s1^{2H}, and lower >= R_RL(s0, s1).
    """
    set_dps()
    _check_rough_H_arb(H)
    mean_IT = xi0_flat * T
    eta_sq = eta * eta
    grid = [T * arb(j) / arb(n_boxes) for j in range(n_boxes + 1)]
    lo_half = arb(0)
    hi_half = arb(0)
    for j in range(n_boxes):
        u0, u1 = grid[j], grid[j + 1]
        du = u1 - u0
        # Diagonal triangle: u0 <= s <= u <= u1, area = du^2 / 2
        area_tri = du * du / arb(2)
        r_tri_lo = rl_cov_arb(H, u0, u1) if not u0.is_zero() else arb(0)
        r_tri_hi = u1 ** (arb(2) * H)
        lo_half += area_tri * (eta_sq * r_tri_lo).expm1()
        hi_half += area_tri * (eta_sq * r_tri_hi).expm1()
        # Off-diagonal rectangles: 0 <= i < j, so s in [s0, s1] with s1 <= u0
        for i in range(j):
            s0, s1 = grid[i], grid[i + 1]
            area = (s1 - s0) * du
            r_lo = rl_cov_arb(H, s0, u1) if not s0.is_zero() else arb(0)
            r_hi = rl_cov_arb(H, s1, u0)
            lo_half += area * (eta_sq * r_lo).expm1()
            hi_half += area * (eta_sq * r_hi).expm1()

    var_IT = arb(2) * (xi0_flat * xi0_flat) * arb_interval(lo_half, hi_half)
    return mean_IT, var_IT
