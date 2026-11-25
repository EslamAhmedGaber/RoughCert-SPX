"""RoughCert-SPX-Det: Fully deterministic 176-bit Arb SPX pricing and kernel-gap certificates.

Implements:
1. Theorem 2 (Zero-Moment Global Put-Lipschitz Price Bound):
     |C_cont(K) - C_hat(K)| <= K * (|rho| + rho_bar / sqrt(2 pi)) * E_J
                               + (K * rho^2 / 2) * min(E_I, E_J * (sqrt(E[I_T]) + sqrt(E[I_hat_T])))
   Evaluated strictly in 176-bit Arb with outward rounding (FLOOR / CEILING).
2. Proposition 1 and Theorem 2 (Deterministic Certifiability Bounds for SPX options):
   (a) Uncorrelated case (rho = 0):
       C_cont(K) = E[phi_K(I_T)], where phi_K(V) = C_BS(1, K, V).
       - At ATM (K = 1), phi_1(V) is strictly concave on (0, inf) with exact Jensen upper bound:
             C_cont(1) <= phi_1(E[I_T])
         and sharp lower bound via 1-factor Gaussian conditioning + conditional variance / Lo-type
         concavity bracket in 176-bit Arb.
       - For general K > 0, uses exact one-factor conditional projection I_hat(Z) = E[I_T | Z]
         with exact Arb residual variance P_I = Var(I_T) - Var(I_hat(Z)) and global Lipschitz bound
         in sqrt(V): |phi_K(V) - phi_K(V_hat)| <= (K / sqrt(2 pi)) * |sqrt(V) - sqrt(V_hat)|,
         plus sharper second-order concave/secant bracket at ATM.
   (b) Correlated case (rho != 0):
       Deterministic sampling-free 176-bit Arb enclosure anchored at Black-Scholes (V0 = E[I_T])
       or 1-factor projected forward-variance law, with explicit residual E_J(k=0 or k=1) bound.
       Proves quantitatively where full deterministic Arb pricing succeeds (< 25 bps IV) vs.
       where roughness of J_T necessitates certified-bias Monte Carlo.
3. Arb Kernel-Gap Certificate for Markovian (Bayer-Breneis) and BLP Hybrid kernels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from flint import arb

from .bs_arb import (
    set_dps,
    arb_interval,
    arb_lo_float,
    arb_hi_float,
    bs_call_arb,
    bs_iv_from_call_arb,
)
from .volterra_arb import (
    rl_cov_arb,
    markov_EJ_arb,
    blp_hybrid_EJ_arb,
    exact_IT_moments_arb,
)

set_dps()


def _check_domain(H: float, T: float, eta: float, rho: float, xi0: float, K: float) -> None:
    """Enforce Assumption 1 (Gassiat 2019 true-martingale domain rho in [-1, 0] and rough RL kernel)."""
    if not all(math.isfinite(x) for x in (H, T, eta, rho, xi0, K)):
        raise ValueError("Certification requires finite model parameters")
    if not (0.0 < H <= 0.5):
        raise ValueError(f"Rough volatility Arb bounds require 0 < H <= 0.5; got H={H}")
    if not (T > 0.0 and xi0 > 0.0 and K > 0.0 and eta >= 0.0):
        raise ValueError(f"Require T>0, xi0>0, K>0, eta>=0; got T={T}, xi0={xi0}, K={K}, eta={eta}")
    if not (-1.0 <= rho <= 0.0):
        raise ValueError(
            f"Theorem 2 (Put-Call parity / true martingale via Gassiat 2019) requires rho in [-1, 0]; got rho={rho}"
        )


def put_lipschitz_price_gap_arb(
    K: arb,
    rho: arb,
    E_J: arb,
    mean_IT: arb,
    mean_Ihat: arb | None = None,
    E_I: arb | None = None,
) -> arb:
    """Theorem 2: Certified 176-bit Arb upper bound on |C_cont(K) - C_hat(K)|.

    Assumes rho in [-1, 0] and both processes are true martingales (E[F_T] = E[F_hat_T] = 1).
    Uses:
      |C - C_hat| <= K * (|rho| + rho_bar / sqrt(2 pi)) * E_J
                     + (K * rho^2 / 2) * min(E_I, E_J * (sqrt(E[I_T]) + sqrt(E[I_hat_T])))
    """
    set_dps()
    if rho > arb(0) or rho < arb(-1):
        raise ValueError(f"Theorem 2 requires rho in [-1, 0]; got rho={rho}")
    if mean_Ihat is None:
        mean_Ihat = mean_IT
    rho_abs = abs(rho)
    one_m_r2 = arb(1) - rho * rho
    if one_m_r2.lower() < arb(0):
        one_m_r2 = arb_interval(arb(0), one_m_r2.upper())
    rho_bar = one_m_r2.sqrt()
    inv_sqrt2pi = arb(1) / (arb(2) * arb.pi()).sqrt()

    term_linear = K * (rho_abs + rho_bar * inv_sqrt2pi) * E_J
    i_bound = E_J * (mean_IT.sqrt() + mean_Ihat.sqrt())
    if E_I is not None and E_I.upper() < i_bound.lower():
        i_bound = E_I
    elif E_I is not None and E_I.upper() < i_bound.upper():
        i_bound = arb_interval(arb(0), E_I.upper().min(i_bound.upper()))
    term_quad = (K * rho * rho / arb(2)) * i_bound
    return term_linear + term_quad


def baseline_EJ_constant_vol_arb(H: arb, T: arb, eta: arb, xi0_flat: arb) -> arb:
    """Exact 176-bit Arb E_J when approximating sqrt(v_s) by deterministic baseline sigma_0(s) = E[sqrt(v_s)].

    Since E[sqrt(v_s)] = sqrt(xi0) * exp(-eta^2 / 8 * s^{2H}), we have:
      E[(sqrt(v_s) - E[sqrt(v_s)])^2] = xi0 * (1 - exp(-(eta^2 / 4) * s^{2H})).
    This is the exact 0-factor (k=0) L^2-optimal deterministic integrand for J_T.
    """
    set_dps()
    c_exp = -(eta * eta) / arb(4)
    m_boxes = 128
    lo_sum = arb(0)
    hi_sum = arb(0)
    for j in range(m_boxes):
        s0 = T * (arb(j) / arb(m_boxes)) ** 3
        s1 = T * (arb(j + 1) / arb(m_boxes)) ** 3
        ds = s1 - s0
        f0 = -(c_exp * (s0 ** (arb(2) * H))).expm1()
        f1 = -(c_exp * (s1 ** (arb(2) * H))).expm1()
        lo_sum += ds * f0
        hi_sum += ds * f1
    return (xi0_flat * arb_interval(lo_sum, hi_sum)).sqrt()


def deterministic_spx_bracket_k0_arb(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_flat: float,
    K: float,
) -> Dict[str, float]:
    """Fully deterministic 176-bit Arb SPX price and IV bracket anchored at the exact k=0
    deterministic volatility surrogate sigma_det(s) = E[sqrt(v_s)].

    Note: for sigma_det(s) deterministic, the surrogate SPX model is exact Black-Scholes with
    total variance I_det = int_0^T E[sqrt(v_s)]^2 ds = E[I_T] - E_J(k=0)^2.
    (Wait: check! E[sqrt(v_s)]^2 = xi0 * exp(-eta^2/4 * s^{2H}) = xi0 - xi0*(1 - exp(-eta^2/4 * s^{2H})),
     so I_det = E[I_T] - E_J(k=0)^2 EXACTLY!)
    Then Theorem 2 gives a rigorous, sampling-free 176-bit Arb enclosure of C_cont(K).
    For rho = 0 at ATM (K = 1), we ALSO tighten the upper bound by exact Jensen concavity
    C_cont(1) <= C_BS(1, 1, E[I_T]).
    """
    set_dps()
    _check_domain(H, T, eta, rho, xi0_flat, K)
    H_a = arb(str(H))
    T_a = arb(str(T))
    eta_a = arb(str(eta))
    rho_a = arb(str(rho))
    xi0_a = arb(str(xi0_flat))
    K_a = arb(str(K))

    mean_IT = xi0_a * T_a
    E_J0 = baseline_EJ_constant_vol_arb(H_a, T_a, eta_a, xi0_a)
    I_det = mean_IT - E_J0 * E_J0
    if I_det.lower() < arb(0):
        I_det = arb_interval(arb(0), I_det.upper())

    # Surrogate price is exact Black-Scholes with variance I_det
    C_det = bs_call_arb(arb(1), K_a, I_det)

    # Gap via Theorem 2
    gap = put_lipschitz_price_gap_arb(
        K=K_a, rho=rho_a, E_J=E_J0, mean_IT=mean_IT, mean_Ihat=I_det
    )

    # For rho = 0, we also have the pure variance-channel Lipschitz bound in sqrt(V):
    # |E[C_BS(1,K,I_T)] - C_BS(1,K,I_det)| <= (K/sqrt(2pi)) * E_J0  (already equals gap when rho=0!).
    # Moreover, at rho = 0, K = 1 (ATM), phi_1(V) is strictly concave: C_cont(1) <= C_BS(1, 1, E[I_T]).
    c_lo = C_det - gap
    c_hi = C_det + gap

    if abs(rho) == 0.0 and K == 1.0:
        c_jensen_hi = bs_call_arb(arb(1), K_a, mean_IT)
        if c_jensen_hi.upper() < c_hi.upper():
            c_hi = c_jensen_hi

    # Clip lower bound to no-arbitrage intrinsic max(1 - K, 0)
    intrinsic = arb(1) - K_a
    if intrinsic < arb(0):
        intrinsic = arb(0)
    if c_lo.lower() < intrinsic.lower():
        c_lo = intrinsic

    # Rigorous no-arbitrage call cap: C_cont(K) <= 1 (gives iv_hi = +inf cleanly if c_hi >= 1; NEVER fake-clip!)
    if c_hi.upper() >= arb(1):
        c_hi = arb(1)

    price_ball = arb_interval(c_lo, c_hi)
    iv_lo_a, iv_hi_a = bs_iv_from_call_arb(price_ball, K_a, T_a, F=arb(1))
    iv_lo_f = arb_lo_float(iv_lo_a)
    iv_hi_f = math.inf if not iv_hi_a.is_finite() else arb_hi_float(iv_hi_a)
    p_lo_f = arb_lo_float(c_lo)
    p_hi_f = arb_hi_float(c_hi)

    return {
        "status": "Arb-certified",
        "method": "RoughCert-SPX-Det (k=0)",
        "H": H,
        "T": T,
        "eta": eta,
        "rho": rho,
        "xi0": xi0_flat,
        "K": K,
        "E_J_lo": arb_lo_float(E_J0),
        "E_J_hi": arb_hi_float(E_J0),
        "price_lo": p_lo_f,
        "price_hi": p_hi_f,
        "price_width": arb_hi_float(arb(p_hi_f) - arb(p_lo_f)),
        "iv_lo": iv_lo_f,
        "iv_hi": iv_hi_f,
        "iv_width_bps": math.inf if not math.isfinite(iv_hi_f) else arb_hi_float(arb(10000) * (arb(iv_hi_f) - arb(iv_lo_f))),
    }


def deterministic_spx_rho0_cond_arb(
    H: float,
    T: float,
    eta: float,
    xi0_flat: float,
    K: float,
    n_z: int = 200,
    z_max: float = 8.0,
) -> Dict[str, float]:
    """Sharper 1-factor deterministic 176-bit Arb bracket for rho = 0 using rank-1 Gaussian
    projection onto Z = (1 / ||h||) * int_0^T Y_s ds (mean-direction projection of I_T).

    Exact 1-factor projection:
      Y_hat_s = c(s) * Z,   c(s) = Cov(Y_s, Z),   D_1(s, s) = s^{2H} - c(s)^2 >= 0.
      I_hat(Z) = E[I_T | Z] = int_0^T xi0 * exp(eta c(s) Z - 0.5 eta^2 c(s)^2) ds.
    Residual variance P_I = Var(I_T) - Var(I_hat(Z)) is bounded rigorously in Arb via:
      E[|sqrt(I_T) - sqrt(I_hat)|] <= E[sqrt(I_T) - sqrt(I_hat)]? No: by reverse Minkowski
      or Cauchy-Schwarz on I_T - I_hat:
        |sqrt(a) - sqrt(b)| <= |a - b| / (sqrt(a) + sqrt(b)) <= sqrt(|a - b|),
      AND even sharper at ATM (K=1): phi_1 is strictly concave and increasing, so:
        E[phi_1(I_T)] <= E[phi_1(I_hat(Z))]   (exact conditional Jensen upper bound!!).
      For the lower bound (and general K):
        Let sigma_tilde_s(Z) = E[sqrt(v_s) | Z] = sqrt(xi0) exp(0.5 eta c(s) Z - 0.25 eta^2 c(s)^2 - 0.125 eta^2 D_1(s,s)).
        Wait: E[(sqrt(v_s) - sigma_tilde_s(Z))^2] = xi0 * (1 - exp(-0.25 eta^2 D_1(s,s))).
        And I_tilde(Z) = int_0^T sigma_tilde_s(Z)^2 ds satisfies:
          ||sqrt(I_T) - sqrt(I_tilde(Z))||_2 <= E_J1 = (int_0^T xi0 (1 - exp(-0.25 eta^2 D_1(s,s))) ds)^{1/2}.
        Therefore for EVERY K > 0:
          |E[phi_K(I_T)] - E[phi_K(I_tilde(Z))]| <= (K / sqrt(2 pi)) * E_J1.
      And at ATM (K = 1), we also intersect with the exact conditional Jensen upper bound
      E[phi_1(I_T)] <= E[phi_1(I_hat(Z))].
    """
    set_dps()
    _check_domain(H, T, eta, 0.0, xi0_flat, K)
    H_a = arb(str(H))
    T_a = arb(str(T))
    eta_a = arb(str(eta))
    xi0_a = arb(str(xi0_flat))
    K_a = arb(str(K))

    # Direction h = int_0^T e_s ds has norm^2 = int_0^T int_0^T R_RL(s, u) ds du
    # Closed form! Let's compute <e_s, h> = int_0^T R_RL(s, u) du.
    # Even simpler and 100% exact closed form: choose the rank-1 projection Z = Y_T / T^H (terminal value).
    # Wait: for RL fBm, Cov(Y_s, Y_T) = R_RL(s, T) is CLOSED FORM via rl_cov_arb(H, s, T).
    # Check: c(s) = R_RL(s, T) / T^H is strictly increasing in s on [0, T], with c(0)=0, c(T)=T^H.
    # Let's verify monotonicity of c(s): since R_RL(s, T) increases in s (proved in volterra_arb.py),
    # c(s) is strictly increasing on [0, T].
    # And D_1(s, s) = s^{2H} - c(s)^2 is computable in closed form at every s with 176-bit Arb.
    TH = T_a ** H_a
    n_s = 48
    s_grid = [T_a * (arb(i) / arb(n_s)) ** 2 for i in range(n_s + 1)]
    c_grid = [arb(0)] + [rl_cov_arb(H_a, s, T_a) / TH for s in s_grid[1:]]
    # Rigorous box enclosure of D_1(s, s) = s^{2H} - c(s)^2 on each subinterval [s_i, s_{i+1}]:
    # Since both s^{2H} and c(s)^2 are nondecreasing, D_1(s,s) in [s_i^{2H} - c_{i+1}^2, s_{i+1}^{2H} - c_i^2].
    D_box_hi = []
    D_box_lo = []
    for i in range(n_s):
        d_lo = s_grid[i] ** (arb(2) * H_a) - c_grid[i + 1] ** 2
        if d_lo.lower() < arb(0):
            d_lo = arb(0)
        d_hi = s_grid[i + 1] ** (arb(2) * H_a) - c_grid[i] ** 2
        if d_hi.lower() < arb(0):
            d_hi = arb(0)
        D_box_lo.append(d_lo)
        D_box_hi.append(d_hi)

    # Certified E_J1 = (int_0^T xi0 * (1 - exp(-0.25 eta^2 D_1(s,s))) ds)^{1/2}
    c_exp = -(eta_a * eta_a) / arb(4)
    ej1_sq_lo = arb(0)
    ej1_sq_hi = arb(0)
    for i in range(n_s):
        ds = s_grid[i + 1] - s_grid[i]
        ej1_sq_lo += xi0_a * ds * (-(c_exp * D_box_lo[i]).expm1())
        ej1_sq_hi += xi0_a * ds * (-(c_exp * D_box_hi[i]).expm1())
    E_J1 = arb_interval(ej1_sq_lo, ej1_sq_hi).sqrt()

    # Integrate E[phi_K(I_tilde(Z))] and (at ATM) E[phi_1(I_hat(Z))] over Z ~ N(0,1) rigorously in Arb.
    # Monotonicity in Z:
    # Since c(s) >= 0 for all s, both I_tilde(Z) and I_hat(Z) are STRICTLY INCREASING in Z.
    # And phi_K(V) = C_BS(1, K, V) is STRICTLY INCREASING in V.
    # Therefore g(Z) = phi_K(I_tilde(Z)) is STRICTLY INCREASING in Z on (-inf, +inf) and bounded in [intrinsic, 1].
    # That makes 1D Gaussian integration in Arb rigorous via monotone panels + exact tails.
    z_grid = [arb(-z_max) + arb(2 * z_max) * arb(j) / arb(n_z) for j in range(n_z + 1)]
    inv_sqrt2 = arb(1) / arb(2).sqrt()
    cdf_grid = [(arb(1) / arb(2)) * ((-z) * inv_sqrt2).erfc() for z in z_grid]

    def eval_I_tilde_box(z_val: arb) -> arb:
        """Rigorous Arb enclosure of I_tilde(z) = int_0^T xi0 exp(eta c(s) z - 0.5 eta^2 (s^{2H} - 0.5 D_1(s,s))) ds.
        Notice: 0.5 eta^2 (c(s)^2 + 0.5 D_1(s,s)) = 0.5 eta^2 (0.5 s^{2H} + 0.5 c(s)^2).
        Check algebra:
          sigma_tilde(s)^2 = xi0 * exp(eta c(s) z - 0.5 eta^2 s^{2H} + 0.25 eta^2 D_1(s,s))
                           = xi0 * exp(eta c(s) z - 0.25 eta^2 s^{2H} - 0.25 eta^2 c(s)^2).
        Both c(s) and s^{2H} are nondecreasing on [s_i, s_{i+1}], so box bounds are exact.
        """
        acc_lo = arb(0)
        acc_hi = arb(0)
        for i in range(n_s):
            ds = s_grid[i + 1] - s_grid[i]
            c_box = arb_interval(c_grid[i], c_grid[i + 1])
            s2h_box = arb_interval(s_grid[i] ** (arb(2) * H_a), s_grid[i + 1] ** (arb(2) * H_a))
            ex = eta_a * c_box * z_val - (eta_a * eta_a / arb(4)) * (s2h_box + c_box * c_box)
            val = xi0_a * ds * ex.exp()
            acc_lo += arb(val.lower())
            acc_hi += arb(val.upper())
        return arb_interval(acc_lo, acc_hi)

    def eval_I_hat_box(z_val: arb) -> arb:
        """Rigorous Arb enclosure of I_hat(z) = E[I_T | Z=z] = int_0^T xi0 exp(eta c(s) z - 0.5 eta^2 c(s)^2) ds."""
        acc_lo = arb(0)
        acc_hi = arb(0)
        for i in range(n_s):
            ds = s_grid[i + 1] - s_grid[i]
            c_box = arb_interval(c_grid[i], c_grid[i + 1])
            ex = eta_a * c_box * z_val - (eta_a * eta_a / arb(2)) * (c_box * c_box)
            val = xi0_a * ds * ex.exp()
            acc_lo += arb(val.lower())
            acc_hi += arb(val.upper())
        return arb_interval(acc_lo, acc_hi)

    intrinsic = arb(1) - K_a
    if intrinsic < arb(0):
        intrinsic = arb(0)

    g_vals = [bs_call_arb(arb(1), K_a, eval_I_tilde_box(z)) for z in z_grid]
    # Left tail (-inf, -z_max]: g in [intrinsic, g_vals[0]]
    p_left = cdf_grid[0]
    p_right = arb(1) - cdf_grid[-1]
    e_lo = p_left * intrinsic + p_right * arb(g_vals[-1].lower())
    e_hi = p_left * arb(g_vals[0].upper()) + p_right * arb(1)
    for j in range(n_z):
        dp = cdf_grid[j + 1] - cdf_grid[j]
        e_lo += dp * arb(g_vals[j].lower())
        e_hi += dp * arb(g_vals[j + 1].upper())

    E_g = arb_interval(e_lo, e_hi)
    gap = (K_a / (arb(2) * arb.pi()).sqrt()) * E_J1
    c_lo = arb(E_g.lower()) - arb(gap.upper())
    c_hi = arb(E_g.upper()) + arb(gap.upper())

    # Tighten ATM upper bound using exact conditional Jensen E[phi_1(I_T)] <= E[phi_1(I_hat(Z))]
    if K == 1.0:
        gh_vals = [bs_call_arb(arb(1), K_a, eval_I_hat_box(z)) for z in z_grid]
        j_hi = p_left * arb(gh_vals[0].upper()) + p_right * arb(1)
        for j in range(n_z):
            dp = cdf_grid[j + 1] - cdf_grid[j]
            j_hi += dp * arb(gh_vals[j + 1].upper())
        if j_hi.upper() < c_hi.upper():
            c_hi = j_hi

    if c_lo.lower() < intrinsic.lower():
        c_lo = intrinsic
    if c_hi.upper() >= arb(1):
        c_hi = arb(1)

    price_ball = arb_interval(c_lo, c_hi)
    iv_lo_a, iv_hi_a = bs_iv_from_call_arb(price_ball, K_a, T_a, F=arb(1))
    iv_lo_f = arb_lo_float(iv_lo_a)
    iv_hi_f = math.inf if not iv_hi_a.is_finite() else arb_hi_float(iv_hi_a)
    p_lo_f = arb_lo_float(c_lo)
    p_hi_f = arb_hi_float(c_hi)

    return {
        "status": "Arb-certified",
        "method": "RoughCert-SPX-Det (rho=0, k=1)",
        "H": H,
        "T": T,
        "eta": eta,
        "rho": 0.0,
        "xi0": xi0_flat,
        "K": K,
        "E_J1_lo": arb_lo_float(E_J1),
        "E_J1_hi": arb_hi_float(E_J1),
        "price_lo": p_lo_f,
        "price_hi": p_hi_f,
        "price_width": arb_hi_float(arb(p_hi_f) - arb(p_lo_f)),
        "iv_lo": iv_lo_f,
        "iv_hi": iv_hi_f,
        "iv_width_bps": math.inf if not math.isfinite(iv_hi_f) else arb_hi_float(arb(10000) * (arb(iv_hi_f) - arb(iv_lo_f))),
    }


def deterministic_spx_rho0_atm_twomoment_arb(
    H: float,
    T: float,
    eta: float,
    xi0_flat: float,
    n_boxes: int = 24,
) -> Dict[str, float]:
    """Sharp two-moment 176-bit Arb deterministic ATM Call bracket for rho = 0 (Proposition 1).

    Since phi_1(V) = C_BS(1, 1, V) satisfies phi_1(0) = 0, phi_1' > 0, phi_1'' < 0, phi_1''' > 0
    on (0, inf), for any nonnegative random variable I_T >= 0 with mean m and variance v:
        (m^2 / (m^2 + v)) * phi_1(m + v / m)  <=  E[phi_1(I_T)]  <=  phi_1(m).
    Because the lower bound is strictly decreasing in v >= 0 (by concavity x phi_1'(x) < phi_1(x)),
    plugging in the Arb upper bound v_hi >= Var(I_T) from exact_IT_moments_arb gives a rigorous,
    sampling-free O(eta^2 T^{2H}) width enclosure with ZERO Gaussian integration.
    """
    set_dps()
    _check_domain(H, T, eta, 0.0, xi0_flat, 1.0)
    H_a = arb(str(H))
    T_a = arb(str(T))
    eta_a = arb(str(eta))
    xi0_a = arb(str(xi0_flat))

    m_a, var_a = exact_IT_moments_arb(H_a, T_a, eta_a, xi0_a, n_boxes=n_boxes)
    v_hi = arb(var_a.upper())

    c_hi = bs_call_arb(arb(1), arb(1), m_a)
    x_star = m_a + v_hi / m_a
    c_lo = (m_a * m_a / (m_a * m_a + v_hi)) * bs_call_arb(arb(1), arb(1), x_star)

    price_ball = arb_interval(c_lo, c_hi)
    iv_lo_a, iv_hi_a = bs_iv_from_call_arb(price_ball, arb(1), T_a, F=arb(1))
    p_lo_f = arb_lo_float(c_lo)
    p_hi_f = arb_hi_float(c_hi)
    return {
        "status": "Arb-certified",
        "method": "RoughCert-SPX-Det (rho=0, 2-moment sharp)",
        "H": H,
        "T": T,
        "eta": eta,
        "rho": 0.0,
        "xi0": xi0_flat,
        "K": 1.0,
        "var_IT_lo": arb_lo_float(var_a),
        "var_IT_hi": arb_hi_float(var_a),
        "price_lo": p_lo_f,
        "price_hi": p_hi_f,
        "price_width": arb_hi_float(arb(p_hi_f) - arb(p_lo_f)),
        "iv_lo": arb_lo_float(iv_lo_a),
        "iv_hi": arb_hi_float(iv_hi_a),
        "iv_width_bps": arb_hi_float(arb(10000) * (arb(arb_hi_float(iv_hi_a)) - arb(arb_lo_float(iv_lo_a)))),
    }


def certify_kernel_gap_markov_arb(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_flat: float,
    K: float,
    nodes: Sequence[float],
    weights: Sequence[float],
    n_boxes: int = 64,
) -> Dict[str, float]:
    """Certify 176-bit Arb upper bound on |C_cont(rBergomi) - C_cont(Markovian M-factor)|."""
    set_dps()
    _check_domain(H, T, eta, rho, xi0_flat, K)
    H_a = arb(str(H))
    T_a = arb(str(T))
    eta_a = arb(str(eta))
    rho_a = arb(str(rho))
    xi0_a = arb(str(xi0_flat))
    K_a = arb(str(K))
    nodes_a = [arb(str(float(x))) for x in nodes]
    weights_a = [arb(str(float(w))) for w in weights]

    E_J_a, kern_l2_a = markov_EJ_arb(H_a, T_a, eta_a, xi0_a, nodes_a, weights_a, n_boxes=n_boxes)
    mean_IT = xi0_a * T_a
    gap_a = put_lipschitz_price_gap_arb(K=K_a, rho=rho_a, E_J=E_J_a, mean_IT=mean_IT)
    return {
        "status": "Arb-certified",
        "M_factors": len(nodes),
        "kernel_L2_err_lo": arb_lo_float(kern_l2_a),
        "kernel_L2_err_hi": arb_hi_float(kern_l2_a),
        "E_J_lo": arb_lo_float(E_J_a),
        "E_J_hi": arb_hi_float(E_J_a),
        "price_gap_upper": arb_hi_float(gap_a),
    }


def certify_kernel_gap_blp_hybrid_arb(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_flat: float,
    K: float,
    n: int,
    kappa: int,
) -> Dict[str, float]:
    """Certify 176-bit Arb upper bound on |C_cont(rBergomi) - C_cont(BLP Hybrid kernel kappa, n)|."""
    set_dps()
    _check_domain(H, T, eta, rho, xi0_flat, K)
    H_a = arb(str(H))
    T_a = arb(str(T))
    eta_a = arb(str(eta))
    rho_a = arb(str(rho))
    xi0_a = arb(str(xi0_flat))
    K_a = arb(str(K))

    E_J_a = blp_hybrid_EJ_arb(H_a, T_a, eta_a, xi0_a, n=n, kappa=kappa)
    mean_IT = xi0_a * T_a
    gap_a = put_lipschitz_price_gap_arb(K=K_a, rho=rho_a, E_J=E_J_a, mean_IT=mean_IT)
    return {
        "status": "Arb-certified",
        "n": n,
        "kappa": kappa,
        "E_J_lo": arb_lo_float(E_J_a),
        "E_J_hi": arb_hi_float(E_J_a),
        "price_gap_upper": arb_hi_float(gap_a),
    }
