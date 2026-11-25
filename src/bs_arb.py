"""Exact 176-bit Arb (python-flint) and vectorized Black-Scholes engine.

Provides:
1. Strict outward-rounded float conversion from Arb balls (FLOOR / CEILING).
2. 176-bit Arb Black-Scholes Call and Put pricing in total variance V = sigma^2 * T.
3. 176-bit Arb verified implied volatility inversion propagating price intervals [P_L, P_U]
   outward into certified IV intervals [IV_L, IV_U] (and ATM skew enclosures).
4. Fast vectorized NumPy Black-Scholes Call/Put and Vega for Romano-Touzi conditional MC.

Mathematical guarantees:
- For every F > 0, K > 0, V >= 0, P_BS(F, K, V) = K * Phi(-d_-) - F * Phi(-d_+) is
  strictly increasing in V > 0 and satisfies:
      |d/du P_BS(e^u, K, Sigma^2)| = e^u Phi(-d_+) <= K Phi(-d_-) <= K,
      |d/dSigma P_BS(e^u, K, Sigma^2)| = K phi(d_-) <= K / sqrt(2 pi).
- All Arb scalar conversions step strictly outward by 2 ULPs and assert exact Arb ordering.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.special import ndtr
from flint import arb, ctx

ARB_DPS = 52  # 176 bits (> 169 bits)


def set_dps(dps: int = ARB_DPS) -> None:
    """Set global Arb working precision (at least 52 decimal digits = 176 bits)."""
    ctx.dps = max(ARB_DPS, dps)


set_dps()


def arb_interval(lo: arb, hi: arb) -> arb:
    """Return a single Arb ball guaranteed to contain [lo.lower(), hi.upper()]."""
    return arb(lo.lower()).union(arb(hi.upper()))


def arb_lo_float(x: arb) -> float:
    """Strict outward lower binary64 bound (guaranteed <= x.lower()).

    Exact zero is preserved as 0.0 (avoiding negative subnormal -1e-323).
    """
    lo_arb = arb(x.lower())
    if lo_arb.is_zero():
        return 0.0
    val = float(lo_arb)
    if not math.isfinite(val):
        raise ArithmeticError(f"Non-finite Arb lower endpoint: {x}")
    cand = math.nextafter(math.nextafter(val, -math.inf), -math.inf)
    if not (math.isfinite(cand) and arb(cand) <= lo_arb):
        raise ArithmeticError(f"Failed outward lower rounding for {x}")
    return cand


def arb_hi_float(x: arb) -> float:
    """Strict outward upper binary64 bound (guaranteed >= x.upper()).

    Exact zero is preserved as 0.0.
    """
    hi_arb = arb(x.upper())
    if hi_arb.is_zero():
        return 0.0
    val = float(hi_arb)
    if not math.isfinite(val):
        raise ArithmeticError(f"Non-finite Arb upper endpoint: {x}")
    cand = math.nextafter(math.nextafter(val, math.inf), math.inf)
    if not (math.isfinite(cand) and arb(cand) >= hi_arb):
        raise ArithmeticError(f"Failed outward upper rounding for {x}")
    return cand


def arb_to_bounds(x: arb) -> Tuple[float, float]:
    """Return strict outward-rounded (lo, hi) binary64 pair enclosing x."""
    return arb_lo_float(x), arb_hi_float(x)


def arb_norm_cdf(x: arb) -> arb:
    """Standard normal CDF Phi(x) = 0.5 * erfc(-x / sqrt(2)) in Arb ball arithmetic."""
    inv_sqrt2 = arb(1) / arb(2).sqrt()
    return (arb(1) / arb(2)) * ((-x) * inv_sqrt2).erfc()


def arb_norm_pdf(x: arb) -> arb:
    """Standard normal PDF phi(x) = exp(-x^2 / 2) / sqrt(2 pi) in Arb ball arithmetic."""
    two_pi = arb(2) * arb.pi()
    return (-(x * x) / arb(2)).exp() / two_pi.sqrt()


def bs_call_arb(F: arb, K: arb, V: arb) -> arb:
    """Black-Scholes undiscounted Call price C_BS(F, K, V) in 176-bit Arb arithmetic.

    Here F > 0 is forward, K > 0 is strike, and V >= 0 is total variance sigma^2 * T.
    """
    if V <= arb(0):
        return (F - K).max(arb(0))
    if V.lower() <= arb(0):
        intrinsic = (F - K).max(arb(0))
        hi_val = bs_call_arb(F, K, arb(V.upper()))
        return arb_interval(intrinsic, hi_val)
    sig = V.sqrt()
    log_fk = (F / K).log()
    d1 = log_fk / sig + sig / arb(2)
    d2 = d1 - sig
    return F * arb_norm_cdf(d1) - K * arb_norm_cdf(d2)


def bs_put_arb(F: arb, K: arb, V: arb) -> arb:
    """Black-Scholes undiscounted Put price P_BS(F, K, V) in 176-bit Arb arithmetic."""
    if V <= arb(0):
        return (K - F).max(arb(0))
    if V.lower() <= arb(0):
        intrinsic = (K - F).max(arb(0))
        hi_val = bs_put_arb(F, K, arb(V.upper()))
        return arb_interval(intrinsic, hi_val)
    sig = V.sqrt()
    log_fk = (F / K).log()
    d1 = log_fk / sig + sig / arb(2)
    d2 = d1 - sig
    return K * arb_norm_cdf(-d2) - F * arb_norm_cdf(-d1)


def bs_iv_from_call_arb(
    call_price: arb,
    K: arb,
    T: arb,
    F: arb | None = None,
    sigma_lo_init: float = 1e-6,
    sigma_hi_init: float = 5.0,
    bisect_iters: int = 80,
) -> Tuple[arb, arb]:
    """Certify Black-Scholes implied volatility enclosure [iv_lo, iv_hi] for a Call price ball.

    Uses monotonicity of C_BS(F, K, sigma^2 T) in sigma > 0 and strict Arb bisection:
      - iv_lo satisfies C_BS(F, K, iv_lo^2 T).upper() <= call_price.lower()  (or 0 if intrinsic)
      - iv_hi satisfies C_BS(F, K, iv_hi^2 T).lower() >= call_price.upper()
    """
    set_dps()
    if F is None:
        F = arb(1)
    if not (T > arb(0) and K > arb(0) and F > arb(0)):
        raise ValueError("Require strictly positive F, K, T for IV inversion.")

    intrinsic = F - K
    if intrinsic < arb(0):
        intrinsic = arb(0)

    c_lo = arb(call_price.lower())
    c_hi = arb(call_price.upper())

    # Solve lower IV bound for c_lo
    if c_lo <= intrinsic:
        iv_lo = arb(0)
    elif c_lo >= F:
        return arb("inf"), arb("inf")
    else:
        lo = arb(str(sigma_lo_init))
        while bs_call_arb(F, K, lo * lo * T) >= c_lo and lo > arb("1e-15"):
            lo = lo / arb(2)
        hi = arb(str(sigma_hi_init))
        while bs_call_arb(F, K, hi * hi * T) <= c_lo:
            hi = hi * arb(2)
            if hi > arb(100):
                raise ArithmeticError("IV lower-endpoint upper bracket exceeded 100.")
        for _ in range(bisect_iters):
            mid = (lo + hi) / arb(2)
            val = bs_call_arb(F, K, mid * mid * T)
            if val.upper() <= c_lo:
                lo = mid
            elif val.lower() >= c_lo:
                hi = mid
            else:
                break
        if not (bs_call_arb(F, K, lo * lo * T) <= c_lo):
            raise ArithmeticError("Failed certified IV lower bound verification.")
        iv_lo = lo

    # Solve upper IV bound for c_hi (rigorous +inf when c_hi >= F; NEVER artificially clip!)
    if c_hi <= intrinsic:
        iv_hi = arb(0)
    elif c_hi >= F:
        iv_hi = arb("inf")
    else:
        lo = arb(str(sigma_lo_init))
        while bs_call_arb(F, K, lo * lo * T) >= c_hi and lo > arb("1e-15"):
            lo = lo / arb(2)
        hi = arb(str(sigma_hi_init))
        while bs_call_arb(F, K, hi * hi * T) <= c_hi:
            hi = hi * arb(2)
            if hi > arb(100):
                raise ArithmeticError("IV upper-endpoint upper bracket exceeded 100.")
        for _ in range(bisect_iters):
            mid = (lo + hi) / arb(2)
            val = bs_call_arb(F, K, mid * mid * T)
            if val.lower() >= c_hi:
                hi = mid
            elif val.upper() <= c_hi:
                lo = mid
            else:
                break
        if not (bs_call_arb(F, K, hi * hi * T) >= c_hi):
            raise ArithmeticError("Failed certified IV upper bound verification.")
        iv_hi = hi

    return iv_lo, iv_hi


def bs_iv_interval_from_price_bounds(
    price_lo: float,
    price_hi: float,
    K: float,
    T: float,
    F: float = 1.0,
    is_put: bool = False,
) -> Tuple[float, float]:
    """Convert price interval [price_lo, price_hi] into outward-rounded IV bounds [iv_lo, iv_hi].

    If is_put=True, converts Put interval to Call interval via exact Put-Call parity
    C = P + F - K in Arb arithmetic.
    """
    set_dps()
    F_a = arb(str(F))
    K_a = arb(str(K))
    T_a = arb(str(T))
    p_lo_a = arb(price_lo)
    p_hi_a = arb(price_hi)
    if p_lo_a > p_hi_a:
        raise ValueError(f"Invalid price interval [{price_lo}, {price_hi}]")
    if is_put:
        c_lo_a = p_lo_a + F_a - K_a
        c_hi_a = p_hi_a + F_a - K_a
    else:
        c_lo_a = p_lo_a
        c_hi_a = p_hi_a
    ball = arb_interval(c_lo_a, c_hi_a)
    iv_lo_a, iv_hi_a = bs_iv_from_call_arb(ball, K_a, T_a, F=F_a)
    lo_f = math.inf if not iv_lo_a.is_finite() else arb_lo_float(iv_lo_a)
    hi_f = math.inf if not iv_hi_a.is_finite() else arb_hi_float(iv_hi_a)
    return lo_f, hi_f


def bs_iv_point_arb(price: float, K: float, T: float, F: float = 1.0, is_put: bool = False) -> float:
    """High-precision scalar IV midpoint verified via Arb bisection."""
    lo, hi = bs_iv_interval_from_price_bounds(price, price, K, T, F=F, is_put=is_put)
    return 0.5 * (lo + hi)


def atm_skew_interval_arb(
    iv_minus: Tuple[float, float],
    iv_plus: Tuple[float, float],
    dk: float,
) -> Tuple[float, float]:
    """Certified central-difference ATM skew S_ATM(dk) = (IV(+dk) - IV(-dk)) / (2 dk).

    Propagates outward-rounded IV intervals rigorously:
      lo = (iv_plus_lo - iv_minus_hi) / (2 dk)
      hi = (iv_plus_hi - iv_minus_lo) / (2 dk)
    """
    set_dps()
    if not math.isfinite(dk) or dk <= 0:
        raise ValueError("Skew spacing must be finite and strictly positive")
    if any(math.isnan(x) for pair in (iv_minus, iv_plus) for x in pair) or any(pair[0] > pair[1] for pair in (iv_minus, iv_plus)):
        raise ValueError("Skew requires ordered IV intervals")
    dk_a = arb(str(dk))
    denom = arb(2) * dk_a
    lo_a = (arb(iv_plus[0]) - arb(iv_minus[1])) / denom
    hi_a = (arb(iv_plus[1]) - arb(iv_minus[0])) / denom
    return arb_lo_float(lo_a), arb_hi_float(hi_a)


# =============================================================================
# Fast vectorized NumPy Black-Scholes for Romano-Touzi Conditional Monte Carlo
# =============================================================================

def bs_call_vec(F: np.ndarray | float, K: np.ndarray | float, V: np.ndarray | float) -> np.ndarray:
    """Vectorized Black-Scholes Call price C_BS(F, K, V) where V = sigma^2 * T >= 0.

    Handles V = 0 cleanly without NaNs and supports scalar or array broadcasting cleanly.
    """
    F_b, K_b, V_b = np.broadcast_arrays(
        np.asarray(F, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.maximum(np.asarray(V, dtype=np.float64), 0.0),
    )
    is_scalar = F_b.ndim == 0
    F_1 = np.atleast_1d(F_b)
    K_1 = np.atleast_1d(K_b)
    V_1 = np.atleast_1d(V_b)
    out = np.array(np.maximum(F_1 - K_1, 0.0), dtype=np.float64, copy=True)
    mask = V_1 > 1e-30
    if np.any(mask):
        sig = np.sqrt(V_1[mask])
        d1 = np.log(F_1[mask] / K_1[mask]) / sig + 0.5 * sig
        d2 = d1 - sig
        out[mask] = F_1[mask] * ndtr(d1) - K_1[mask] * ndtr(d2)
    return float(out[0]) if is_scalar else out.reshape(F_b.shape)


def bs_put_vec(F: np.ndarray | float, K: np.ndarray | float, V: np.ndarray | float) -> np.ndarray:
    """Vectorized Black-Scholes Put price P_BS(F, K, V) where V = sigma^2 * T >= 0."""
    F_b, K_b, V_b = np.broadcast_arrays(
        np.asarray(F, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.maximum(np.asarray(V, dtype=np.float64), 0.0),
    )
    is_scalar = F_b.ndim == 0
    F_1 = np.atleast_1d(F_b)
    K_1 = np.atleast_1d(K_b)
    V_1 = np.atleast_1d(V_b)
    out = np.array(np.maximum(K_1 - F_1, 0.0), dtype=np.float64, copy=True)
    mask = V_1 > 1e-30
    if np.any(mask):
        sig = np.sqrt(V_1[mask])
        d1 = np.log(F_1[mask] / K_1[mask]) / sig + 0.5 * sig
        d2 = d1 - sig
        out[mask] = K_1[mask] * ndtr(-d2) - F_1[mask] * ndtr(-d1)
    return float(out[0]) if is_scalar else out.reshape(F_b.shape)


def bs_call_iv_vec(prices: np.ndarray, K: np.ndarray, T: float, F: float = 1.0) -> np.ndarray:
    """Fast scalar-verified vectorized IV solver (uses Arb for final check when desired)."""
    prices = np.asarray(prices, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    out = np.empty_like(prices, dtype=np.float64)
    for idx in np.ndindex(prices.shape):
        k_val = float(K[idx]) if K.ndim > 0 else float(K)
        out[idx] = bs_iv_point_arb(float(prices[idx]), k_val, T, F=F, is_put=False)
    return out
