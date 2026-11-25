"""RoughCert-SPX-CE and RoughCert-SPX-Bridge: Exact-Bridge Coupled RQMC Engine.

Implements:
1. Exact joint Gaussian path generator for (Delta W^(1), Y) on nested dyadic grids
   n_0 < 2 n_0 < ... < n_max via exact Riemann-Liouville covariance and Schur complement,
   driven by scrambled Sobol (LMS + digital shift) RQMC with Brownian-bridge ordering + antithetic coupling.
2. Exact zero-mean six-control bank fitted strictly on independent pilot shifts (R_pilot = 2):
   - Three multi-budget discrete stopped-timer option controls (Lemma 1):
       E[C_BS(F_tau, K, Q_m - rho^2 * I_tau)] == C_BS(1, K, Q_m)  EXACTLY on any finite grid n.
   - Exact discrete martingale & variance controls: F_T - 1, I_left - E[I_left], and J_left.
   Separate level weights W_l and increment-optimal weights W_{Delta,l} are held fixed across
   the R independent main shifts, preserving exact unbiasedness and strict i.i.d. Student-t validity.
3. RoughCert-SPX-CE (primary method='ce_i_only'):
   - Channel I_T: I_hat^CE = E[I_T | G_n] evaluated by 2-point Gauss-Legendre inside each cell
     (exact E[I_hat^CE] = int_0^T xi0(s) ds up to deterministic quadrature of xi0).
   - Channel J_T: energy-conserving left-point leverage integral preserving exact local
     quadratic variation E[v_{t_i}] = xi0(t_i) and the full FSW weak rate min(3H+1/2, 1).
   - Also retains 'ce_adapted' (conditional-Jensen J_T smoothing) strictly as an ablation study
     confirming Proposition 2's Theta(n^{-2H}) correlation-attenuation pitfall.
4. RoughCert-SPX-Bridge:
   Evaluates nested levels n_0, 2 n_0, ..., n_max on the EXACT SAME Brownian sample path W^(1)
   (zero kernel error, exact telescoping increments Delta_l = C_hat(n_l) - C_hat(n_{l-1})),
   separating RQMC sampling CI (Student-t over R independent shifts) from coupled discretisation
   bias envelope via 3-sigma contraction gates, and mapping price brackets into 176-bit Arb IV & skew.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import ndtri
from scipy.stats import qmc, t as student_t

from .bs_arb import (
    bs_call_vec,
    bs_put_vec,
    bs_iv_interval_from_price_bounds,
    bs_iv_point_arb,
    atm_skew_interval_arb,
)
from .volterra_arb import rl_cov_float, rl_cross_cov_float


# =============================================================================
# Brownian-Bridge transform matrix for power-of-2 grid n
# =============================================================================

def brownian_bridge_matrix(n: int, T: float) -> np.ndarray:
    """Construct n x n matrix B such that Delta_W = B @ Z (Z ~ N(0, I_n)) produces exact
    Brownian increments Delta W_i on [i*dt, (i+1)*dt], ordering the leading Sobol coordinates
    to the coarsest temporal scales (endpoint, midpoint, quarters, ...).

    Supports any positive integer n (exact power-of-2 dyadic bridge when n = 2^p,
    general recursive midpoint bridge otherwise).
    """
    dt = T / n
    # Build bridge for W_{t_1}, ..., W_{t_n} where W_0 = 0
    # Each row j of W_mat gives W_{t_{j+1}} as a linear combination of Z_0, ..., Z_{n-1}
    W_mat = np.zeros((n + 1, n), dtype=np.float64)
    # Column 0 generates W_T = sqrt(T) * Z_0
    W_mat[n, 0] = math.sqrt(T)
    col = 1
    queue: List[Tuple[int, int]] = [(0, n)]
    while queue:
        next_queue: List[Tuple[int, int]] = []
        for lo, hi in queue:
            if hi - lo <= 1:
                continue
            mid = (lo + hi) // 2
            t_lo = lo * dt
            t_mid = mid * dt
            t_hi = hi * dt
            w_lo = (t_hi - t_mid) / (t_hi - t_lo)
            w_hi = (t_mid - t_lo) / (t_hi - t_lo)
            std_mid = math.sqrt((t_mid - t_lo) * (t_hi - t_mid) / (t_hi - t_lo))
            W_mat[mid, :] = w_lo * W_mat[lo, :] + w_hi * W_mat[hi, :]
            W_mat[mid, col] = std_mid
            col += 1
            next_queue.append((lo, mid))
            next_queue.append((mid, hi))
        queue = next_queue
    assert col == n, f"Bridge column mismatch: {col} vs {n}"
    # Increments Delta W_i = W_{t_{i+1}} - W_{t_i}
    B = W_mat[1:, :] - W_mat[:-1, :]
    return B


def _safe_psd_factor(cov: np.ndarray, use_pca: bool = True) -> np.ndarray:
    """Factor symmetric PSD covariance matrix C = L @ L.T.

    If use_pca=True, uses eigendecomposition sorted in descending eigenvalue order so the
    leading RQMC Sobol dimensions carry the highest variance modes.
    """
    sym = 0.5 * (cov + cov.T)
    if use_pca:
        eigvals, eigvecs = np.linalg.eigh(sym)
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 0.0)
        eigvecs = eigvecs[:, idx]
        return eigvecs * np.sqrt(eigvals)[None, :]
    jitter = 1e-14 * max(float(np.mean(np.diag(sym))), 1.0)
    return cholesky(sym + jitter * np.eye(sym.shape[0]), lower=True)


# =============================================================================
# Exact Joint Gaussian Volterra Grid + Causal/Full Conditional Projection Setup
# =============================================================================

@dataclass
class VolterraGridCache:
    """Precomputed exact Gaussian matrices for grid size n on [0, T]."""
    H: float
    T: float
    n: int
    dt: float
    t_grid: np.ndarray              # shape (n+1,)
    B_bridge: np.ndarray            # (n, n): dW = Z_W @ B_bridge.T
    A_yw: np.ndarray                # (n, n): E[Y | dW] = dW @ A_yw.T
    L_y_given_w: np.ndarray         # (n, n): Y_perp = Z_Y @ L_y_given_w.T (PCA ordered!)
    # Full conditional projection for I_T^CE at 2-point Gauss-Legendre quadrature nodes:
    s_quad: np.ndarray              # (2*n,)
    w_quad: np.ndarray              # (2*n,)
    A_ce_full: np.ndarray           # (2*n, 2*n): Y_hat(s_quad) = G_n @ A_ce_full.T
    var_ce_full: np.ndarray         # (2*n,): Var(Y_hat(s_quad))
    # Strictly causal conditional projection for J_T^CE at 2-point GL nodes in each cell i:
    # For cell i (i >= 1), projects Y(s_{i,q}) onto G_i = (dW_0..dW_{i-1}, Y_{t_1}..Y_{t_i}).
    causal_weights_dw: List[np.ndarray]  # list length n: shape (2, i)
    causal_weights_y: List[np.ndarray]   # list length n: shape (2, i)
    var_ce_causal: np.ndarray            # (n, 2): Var(Y_hat(s_{i,q} | G_i))
    setup_seconds: float


def build_volterra_grid_cache(H: float, T: float, n: int, need_ce: bool = True) -> VolterraGridCache:
    """Assemble exact joint (dW, Y) Schur-complement factorisation and (optional) CE projections."""
    t0 = time.perf_counter()
    dt = T / n
    t_grid = np.linspace(0.0, T, n + 1)
    t_pos = t_grid[1:]  # t_1, ..., t_n

    # 1. Exact covariance blocks for G_n = [dW_0..dW_{n-1}, Y_{t_1}..Y_{t_n}]
    Sigma_YY = rl_cov_float(H, t_pos[:, None], t_pos[None, :])
    Sigma_YW = rl_cross_cov_float(H, t_pos[:, None], t_grid[:-1][None, :], t_grid[1:][None, :])

    A_yw = Sigma_YW / dt
    Sigma_Y_given_W = Sigma_YY - A_yw @ Sigma_YW.T
    Sigma_Y_given_W = 0.5 * (Sigma_Y_given_W + Sigma_Y_given_W.T)
    L_y_given_w = _safe_psd_factor(Sigma_Y_given_W, use_pca=True)
    B_bridge = brownian_bridge_matrix(n, T)

    # 2-point Gauss-Legendre nodes on [0, 1]
    gl_u = np.array([0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)], dtype=np.float64)
    gl_w = np.array([0.5, 0.5], dtype=np.float64)
    s_quad = (t_grid[:-1, None] + dt * gl_u[None, :]).reshape(-1)
    w_quad = np.repeat(dt * gl_w[None, :], n, axis=0).reshape(-1)

    if not need_ce:
        return VolterraGridCache(
            H=H, T=T, n=n, dt=dt, t_grid=t_grid,
            B_bridge=B_bridge, A_yw=A_yw, L_y_given_w=L_y_given_w,
            s_quad=s_quad, w_quad=w_quad,
            A_ce_full=np.empty((0, 0)), var_ce_full=np.empty(0),
            causal_weights_dw=[], causal_weights_y=[],
            var_ce_causal=np.empty((0, 0)),
            setup_seconds=time.perf_counter() - t0,
        )

    # 2. Full conditional expectation E[Y(s_quad) | G_n]
    # Use Schur complement of G_n = [dW, Y] to invert Sigma_G stably.
    # Wait: G_n = [dW, Y] is linearly equivalent to [dW, Y_perp] where dW ~ N(0, dt I) and Y_perp ~ N(0, Sigma_Y|W)
    # are INDEPENDENT.
    # Check: Cov(Y_s, dW) = C_{s, W}, Cov(Y_s, Y_perp) = C_{s, Y} - C_{s, W} @ A_yw.T.
    # Thus E[Y_s | G_n] = (1/dt) C_{s, W} dW + C_{s, Y_perp} Sigma_{Y|W}^{-1} Y_perp.
    # This avoids inverting a 2n x 2n matrix and has MUCH better conditioning.
    C_sW = rl_cross_cov_float(H, s_quad[:, None], t_grid[:-1][None, :], t_grid[1:][None, :])  # (2n, n)
    C_sY = rl_cov_float(H, s_quad[:, None], t_pos[None, :])  # (2n, n)
    C_sYperp = C_sY - (C_sW / dt) @ Sigma_YW.T

    # Solve Sigma_Y_given_W @ X = C_sYperp.T with tiny ridge for numerical safety
    diag_scale = max(float(np.mean(np.diag(Sigma_Y_given_W))), 1e-12)
    reg_S = Sigma_Y_given_W + (1e-13 * diag_scale) * np.eye(n)
    c_fac = cho_factor(reg_S, lower=True)
    W_yperp = cho_solve(c_fac, C_sYperp.T).T  # (2n, n)

    # Convert back to G_n = [dW, Y]: Y_perp = Y - A_yw @ dW
    W_y_full = W_yperp
    W_dw_full = (C_sW / dt) - W_yperp @ A_yw
    A_ce_full = np.hstack([W_dw_full, W_y_full])  # (2n, 2n)

    # Exact variance of projection: Var(Y_hat(s)) = (1/dt) ||C_{s,W}||^2 + sum(W_yperp * C_sYperp)
    var_ce_full = np.sum(C_sW * C_sW, axis=1) / dt + np.sum(W_yperp * C_sYperp, axis=1)
    # Clamp strictly to [0, s^{2H}]
    var_ce_full = np.clip(var_ce_full, 0.0, s_quad ** (2.0 * H))

    # 3. Strictly causal conditional expectation E[Y(s_{i,q}) | G_i] for each cell i = 0..n-1
    # Here G_i = [dW_0..dW_{i-1}, Y_{t_1}..Y_{t_i}] is strictly F_{t_i}-measurable.
    # Key speed trick: Sigma_{Y_i | dW_i} is simply the leading i x i principal submatrix of
    # Sigma_YY - (1/dt) * Sigma_YW_causal @ Sigma_YW_causal.T because Sigma_YW is LOWER TRIANGULAR.
    # Check: Cov(Y_{t_j}, dW_k) = 0 for all k >= j.
    # Therefore Y_{t_1..t_i} depends ONLY on dW_0..dW_{i-1} -- hence the leading i x i submatrix of
    # Sigma_Y_given_W IS EXACTLY Cov(Y_{1..i} | dW_{0..i-1})!.
    # Let's double-check that.
    # For j <= i and k >= i: Cov(Y_{t_j}, dW_k) = 0 (since t_k >= t_i >= t_j).
    # So (Sigma_YW @ Sigma_YW.T)_{j, m} for j, m <= i sums ONLY over k < min(j, m) <= i.
    # YES!! That means the Cholesky factor L_causal of Sigma_Y_given_W (lower triangular, unpermuted)
    # has leading i x i block L_causal[:i, :i] giving the EXACT Cholesky factor of Sigma_{Y_{1..i}|dW_{0..i-1}}!.
    L_causal = cholesky(reg_S, lower=True)  # (n, n) lower triangular

    causal_weights_dw: List[np.ndarray] = []
    causal_weights_y: List[np.ndarray] = []
    var_ce_causal = np.zeros((n, 2), dtype=np.float64)

    for i in range(n):
        if i == 0:
            causal_weights_dw.append(np.zeros((2, 0), dtype=np.float64))
            causal_weights_y.append(np.zeros((2, 0), dtype=np.float64))
            var_ce_causal[0, :] = 0.0
            continue
        s_i = s_quad[2 * i : 2 * i + 2]  # (2,)
        c_sw = C_sW[2 * i : 2 * i + 2, :i]  # (2, i)
        c_sy = C_sY[2 * i : 2 * i + 2, :i]  # (2, i)
        c_syperp = c_sy - (c_sw / dt) @ Sigma_YW[:i, :i].T  # (2, i)

        # Solve L_i L_i^T X = c_syperp.T via triangular solves
        Li = L_causal[:i, :i]
        # forward & backward solve
        tmp = np.linalg.solve(Li, c_syperp.T)
        w_yp = np.linalg.solve(Li.T, tmp).T  # (2, i)

        w_dw = (c_sw / dt) - w_yp @ A_yw[:i, :i]
        causal_weights_dw.append(w_dw)
        causal_weights_y.append(w_yp)

        v_c = np.sum(c_sw * c_sw, axis=1) / dt + np.sum(w_yp * c_syperp, axis=1)
        var_ce_causal[i, :] = np.clip(v_c, 0.0, s_i ** (2.0 * H))

    return VolterraGridCache(
        H=H, T=T, n=n, dt=dt, t_grid=t_grid,
        B_bridge=B_bridge, A_yw=A_yw, L_y_given_w=L_y_given_w,
        s_quad=s_quad, w_quad=w_quad,
        A_ce_full=A_ce_full, var_ce_full=var_ce_full,
        causal_weights_dw=causal_weights_dw,
        causal_weights_y=causal_weights_y,
        var_ce_causal=var_ce_causal,
        setup_seconds=time.perf_counter() - t0,
    )


def sample_rqmc_normals(
    dim: int,
    n_half: int,
    seed: int,
    antithetic: bool = True,
) -> np.ndarray:
    """Generate scrambled Sobol (LMS + digital shift) normal samples (shape (2*n_half, dim) if antithetic else (n_half, dim)).

    Requires n_half to be a power of 2 for exact Sobol balance properties.
    """
    m_pow = int(round(math.log2(n_half)))
    if (1 << m_pow) != n_half:
        raise ValueError(f"n_half={n_half} must be a power of 2 for Sobol RQMC.")
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    u = sampler.random_base2(m=m_pow)
    # Clip strictly away from 0 and 1 to guarantee finite inverse-normal values
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    z = ndtri(u)
    if antithetic:
        return np.vstack([z, -z])
    return z


def generate_exact_paths(
    cache: VolterraGridCache,
    Z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map standard normal matrix Z of shape (M, 2*n) into exact (dW, Y) of shape (M, n) each.

    Z[:, :n] drives Brownian bridge increments dW; Z[:, n:] drives PCA-ordered conditional residual Y_perp.
    """
    n = cache.n
    Z_w = Z[:, :n]
    Z_y = Z[:, n:]
    dW = Z_w @ cache.B_bridge.T
    Y = dW @ cache.A_yw.T + Z_y @ cache.L_y_given_w.T
    return dW, Y


def coarsen_paths(dW_fine: np.ndarray, Y_fine: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract EXACT coupled coarse-grid (dW, Y) from fine-grid paths on the same sample space."""
    M, n_fine = dW_fine.shape
    if n_fine % stride != 0:
        raise ValueError(f"n_fine={n_fine} not divisible by stride={stride}")
    n_coarse = n_fine // stride
    dW_c = dW_fine.reshape(M, n_coarse, stride).sum(axis=2)
    Y_c = Y_fine[:, stride - 1 :: stride]
    return dW_c, Y_c


# =============================================================================
# Pathwise Romano-Touzi Evaluators (Exact Zero-Bias Stopped Control Variate)
# =============================================================================

def _eval_stopped_timer_cv(
    sigma_steps: np.ndarray,
    dW: np.ndarray,
    dt: float,
    rho: float,
    strikes: np.ndarray,
    Q_var: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate exact zero-bias discrete-time stopped timer-option control variate Y_cv.

    For any strictly adapted step volatilities sigma_steps (shape (M, n)) and increments dW (M, n):
      F_{i+1} = F_i * exp(rho * sigma_i * dW_i - 0.5 * rho^2 * sigma_i^2 * dt).
    Stop updating F at the first step i where cumulative rho^2 * sum_{j<=i} sigma_j^2 * dt > Q_var.
    By exact discrete-time lognormal convolution,
      E[C_BS(F_tau, K, Q_var - rho^2 * I_tau)] == C_BS(1, K, Q_var)   EXACTLY.
    Returns (Y_cv_paths of shape (M, len(K)), exact_mean of shape (len(K),)).
    """
    M, n = sigma_steps.shape
    rho2 = rho * rho
    dv = rho2 * (sigma_steps * sigma_steps) * dt  # (M, n)
    cum_v = np.cumsum(dv, axis=1)  # (M, n)
    active = cum_v <= Q_var        # (M, n) boolean: step i is taken iff cum_v <= Q_var

    # Guarantee monotonicity of stopping time (once exceeded, stay stopped!)
    # Since dv >= 0, cum_v is nondecreasing, so active is automatically monotone [True..True, False..False].
    step_logF = rho * sigma_steps * dW - 0.5 * dv
    logF_tau = np.sum(np.where(active, step_logF, 0.0), axis=1)  # (M,)
    I_tau_rho2 = np.sum(np.where(active, dv, 0.0), axis=1)       # (M,)
    V_rem = np.maximum(Q_var - I_tau_rho2, 0.0)                  # (M,)
    F_tau = np.exp(logF_tau)

    Y_cv = bs_call_vec(F_tau[:, None], strikes[None, :], V_rem[:, None])  # (M, K)
    exact_mean = bs_call_vec(np.ones(1), strikes, np.array([Q_var]))      # (K,)
    return Y_cv, exact_mean


def _build_zero_mean_cv_bank(
    sigma_steps: np.ndarray,
    dW: np.ndarray,
    dt: float,
    rho: float,
    strikes: np.ndarray,
    F_T: np.ndarray,
    I_centered: np.ndarray,
    J_T: np.ndarray,
    mean_v: float,
) -> np.ndarray:
    """Build exact zero-mean control-variate bank X of shape (M, K, P).

    Every column has 100% proven exact zero expectation in discrete time for any grid n:
      - If |rho| > 0 (P = 6):
          0: Y_cv(0.6 * mean_v) - C_BS(1, K, 0.6 * mean_v)   [Lemma 1]
          1: Y_cv(1.5 * mean_v) - C_BS(1, K, 1.5 * mean_v)   [Lemma 1]
          2: Y_cv(4.0 * mean_v) - C_BS(1, K, 4.0 * mean_v)   [Lemma 1]
          3: F_T - 1                                         [exact martingale, rho <= 0]
          4: I_centered = I - E[I]                           [exact known deterministic mean]
          5: J_T                                             [exact adapted Ito martingale]
      - If rho == 0 (P = 1):
          0: I_centered = I - E[I]
    """
    M = dW.shape[0]
    K = len(strikes)
    if abs(rho) <= 1e-12:
        X = np.empty((M, K, 1), dtype=np.float64)
        X[:, :, 0] = I_centered[:, None]
        return X

    X = np.empty((M, K, 6), dtype=np.float64)
    for j, mult in enumerate((0.6, 1.5, 4.0)):
        Yv, mv = _eval_stopped_timer_cv(sigma_steps, dW, dt, rho, strikes, mult * mean_v)
        X[:, :, j] = Yv - mv[None, :]
    X[:, :, 3] = (F_T - 1.0)[:, None]
    X[:, :, 4] = I_centered[:, None]
    X[:, :, 5] = J_T[:, None]
    return X


def _fit_multicv_pilot(C_pilot: np.ndarray, X_pilot: np.ndarray) -> np.ndarray:
    """Fit OLS multi-control weights W of shape (K, P) strictly on independent pilot paths."""
    _, K, P = X_pilot.shape
    W = np.zeros((K, P), dtype=np.float64)
    for k in range(K):
        Xk = X_pilot[:, k, :]
        Xk_c = Xk - Xk.mean(axis=0, keepdims=True)
        yk_c = C_pilot[:, k] - C_pilot[:, k].mean()
        # Column-scaled lstsq for numerical stability across scales (I ~ 0.01 vs F ~ 1.0)
        sc = np.linalg.norm(Xk_c, axis=0)
        good = sc > 1e-14
        if np.any(good):
            Xn = Xk_c[:, good] / sc[good][None, :]
            coef, *_ = np.linalg.lstsq(Xn, yk_c, rcond=1e-10)
            W[k, good] = coef / sc[good]
    return W


def evaluate_leftpoint_paths(
    dW: np.ndarray,
    Y: np.ndarray,
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_fn: Callable[[np.ndarray], np.ndarray],
    strikes: np.ndarray,
    Q_var: Optional[float] = None,
    build_cv_bank: bool = False,
) -> Dict[str, np.ndarray]:
    """Evaluate Exact-Grid Left-Point Romano-Touzi Call prices and exact timer CV on paths (dW, Y)."""
    M, n = dW.shape
    dt = T / n
    t_left = np.linspace(0.0, T - dt, n)
    xi0_left = np.asarray(xi0_fn(t_left), dtype=np.float64)
    mean_I_left = float(dt * np.sum(xi0_left))

    # Left-point Y: Y_{t_0} = 0, Y_{t_1}..Y_{t_{n-1}} = Y[:, :-1]
    Y_left = np.empty((M, n), dtype=np.float64)
    Y_left[:, 0] = 0.0
    if n > 1:
        Y_left[:, 1:] = Y[:, :-1]

    var_left = t_left ** (2.0 * H)
    v_left = xi0_left[None, :] * np.exp(eta * Y_left - 0.5 * (eta * eta) * var_left[None, :])
    sigma_left = np.sqrt(v_left)

    I_T = dt * np.sum(v_left, axis=1)
    J_T = np.sum(sigma_left * dW, axis=1)
    F_T = np.exp(rho * J_T - 0.5 * (rho * rho) * I_T)
    V_cond = (1.0 - rho * rho) * I_T

    C_paths = bs_call_vec(F_T[:, None], strikes[None, :], V_cond[:, None])
    out = {
        "C_paths": C_paths,
        "F_T": F_T,
        "I_T": I_T,
        "J_T": J_T,
    }
    if build_cv_bank:
        out["X_cv"] = _build_zero_mean_cv_bank(
            sigma_left, dW, dt, rho, strikes, F_T, I_T - mean_I_left, J_T, mean_I_left
        )
    elif Q_var is not None and abs(rho) > 1e-12:
        Y_cv, cv_mean = _eval_stopped_timer_cv(sigma_left, dW, dt, rho, strikes, Q_var)
        out["Y_cv"] = Y_cv
        out["cv_mean"] = cv_mean
    return out


def evaluate_ce_adapted_paths(
    cache: VolterraGridCache,
    dW: np.ndarray,
    Y: np.ndarray,
    eta: float,
    rho: float,
    xi0_fn: Callable[[np.ndarray], np.ndarray],
    strikes: np.ndarray,
    Q_var: Optional[float] = None,
    conserve_variance: bool = True,
    build_cv_bank: bool = False,
) -> Dict[str, np.ndarray]:
    """Jensen-J smoothing ablation (Proposition 2): causal E[sqrt(v_s)|G_i] inside J_T."""
    M, n = dW.shape
    dt = cache.dt
    H = cache.H

    # 1. Channel I_T: full CE projection at 2n Gauss-Legendre nodes
    G_n = np.hstack([dW, Y])  # (M, 2n)
    Y_hat_full = G_n @ cache.A_ce_full.T  # (M, 2n)
    xi0_quad = np.asarray(xi0_fn(cache.s_quad), dtype=np.float64)  # (2n,)
    mean_I_ce = float(xi0_quad @ cache.w_quad)
    v_ce = xi0_quad[None, :] * np.exp(eta * Y_hat_full - 0.5 * (eta * eta) * cache.var_ce_full[None, :])
    I_CE = v_ce @ cache.w_quad  # (M,)

    # 2. Channel J_T: strictly causal sigma_bar_i^CE (shape (M, n))
    sigma_ce = np.empty((M, n), dtype=np.float64)
    s_2d = cache.s_quad.reshape(n, 2)
    sqrt_xi0_2d = np.sqrt(xi0_quad.reshape(n, 2))
    s_2H_2d = s_2d ** (2.0 * H)

    sig0_q = sqrt_xi0_2d[0, :] * np.exp(-0.125 * (eta * eta) * s_2H_2d[0, :])
    sigma_ce[:, 0] = 0.5 * (sig0_q[0] + sig0_q[1])

    for i in range(1, n):
        y_c = dW[:, :i] @ cache.causal_weights_dw[i].T + Y[:, :i] @ cache.causal_weights_y[i].T
        shift = -0.125 * (eta * eta) * (s_2H_2d[i, :] + cache.var_ce_causal[i, :])
        sig_q = sqrt_xi0_2d[i, :][None, :] * np.exp(0.5 * eta * y_c + shift[None, :])
        sigma_ce[:, i] = 0.5 * (sig_q[:, 0] + sig_q[:, 1])

    I_ad = dt * np.sum(sigma_ce * sigma_ce, axis=1)
    J_CE = np.sum(sigma_ce * dW, axis=1)
    F_CE = np.exp(rho * J_CE - 0.5 * (rho * rho) * I_ad)

    if conserve_variance:
        V_cond = I_CE - (rho * rho) * np.minimum(I_CE, I_ad)
        V_cond = np.maximum(V_cond, 0.0)
    else:
        V_cond = (1.0 - rho * rho) * I_CE

    C_paths = bs_call_vec(F_CE[:, None], strikes[None, :], V_cond[:, None])
    out = {
        "C_paths": C_paths,
        "F_T": F_CE,
        "I_T": I_CE,
        "I_ad": I_ad,
        "J_T": J_CE,
    }
    if build_cv_bank:
        out["X_cv"] = _build_zero_mean_cv_bank(
            sigma_ce, dW, dt, rho, strikes, F_CE, I_CE - mean_I_ce, J_CE, mean_I_ce
        )
    elif Q_var is not None and abs(rho) > 1e-12:
        Y_cv, cv_mean = _eval_stopped_timer_cv(sigma_ce, dW, dt, rho, strikes, Q_var)
        out["Y_cv"] = Y_cv
        out["cv_mean"] = cv_mean
    return out


def evaluate_ce_i_only_paths(
    cache: VolterraGridCache,
    dW: np.ndarray,
    Y: np.ndarray,
    eta: float,
    rho: float,
    xi0_fn: Callable[[np.ndarray], np.ndarray],
    strikes: np.ndarray,
    Q_var: Optional[float] = None,
    build_cv_bank: bool = False,
) -> Dict[str, np.ndarray]:
    """Primary RoughCert-SPX-CE: exact Channel-I CE projection paired with energy-conserving J_T."""
    M, n = dW.shape
    dt = cache.dt
    H = cache.H
    T = cache.T

    t_left = np.linspace(0.0, T - dt, n)
    xi0_left = np.asarray(xi0_fn(t_left), dtype=np.float64)
    mean_I_left = float(dt * np.sum(xi0_left))
    Y_left = np.empty((M, n), dtype=np.float64)
    Y_left[:, 0] = 0.0
    if n > 1:
        Y_left[:, 1:] = Y[:, :-1]
    v_left = xi0_left[None, :] * np.exp(eta * Y_left - 0.5 * (eta * eta) * (t_left ** (2.0 * H))[None, :])
    sigma_left = np.sqrt(v_left)
    I_left = dt * np.sum(v_left, axis=1)
    J_left = np.sum(sigma_left * dW, axis=1)
    F_left = np.exp(rho * J_left - 0.5 * (rho * rho) * I_left)

    G_n = np.hstack([dW, Y])
    Y_hat_full = G_n @ cache.A_ce_full.T
    xi0_quad = np.asarray(xi0_fn(cache.s_quad), dtype=np.float64)
    v_ce = xi0_quad[None, :] * np.exp(eta * Y_hat_full - 0.5 * (eta * eta) * cache.var_ce_full[None, :])
    I_CE = v_ce @ cache.w_quad
    V_cond = (1.0 - rho * rho) * I_CE

    C_paths = bs_call_vec(F_left[:, None], strikes[None, :], V_cond[:, None])
    out = {"C_paths": C_paths, "F_T": F_left, "I_T": I_CE, "I_ad": I_left, "J_T": J_left}
    if build_cv_bank:
        out["X_cv"] = _build_zero_mean_cv_bank(
            sigma_left, dW, dt, rho, strikes, F_left, I_left - mean_I_left, J_left, mean_I_left
        )
    elif Q_var is not None and abs(rho) > 1e-12:
        Y_cv, cv_mean = _eval_stopped_timer_cv(sigma_left, dW, dt, rho, strikes, Q_var)
        out["Y_cv"] = Y_cv
        out["cv_mean"] = cv_mean
    return out


def _apply_crossfit_cv(
    C_shifts: np.ndarray,
    Y_cv_shifts: Optional[np.ndarray],
    cv_mean: Optional[np.ndarray],
    F_shifts: np.ndarray,
) -> np.ndarray:
    """Apply leave-one-shift-out cross-fitted dual control variate:
      1) Stopped timer option Y_cv - E[Y_cv]
      2) Martingale forward F_T - 1
    Input shapes:
      C_shifts:    (R, M, K)
      Y_cv_shifts: (R, M, K) or None
      F_shifts:    (R, M)
    Returns shift-level unbiased means of shape (R, K).
    Because control coefficients for shift r are fitted STRICTLY on shifts != r,
    each shift mean is 100% UNBIASED and the R shift means remain exchangeable/independent
    up to O(1/R) weight estimation (we can also fit weights on an independent pilot shift!).
    Here we use shift-wise means with leave-one-out OLS on path covariances from OTHER shifts.
    """
    R, M, K = C_shifts.shape
    shift_means = np.empty((R, K), dtype=np.float64)

    # Precompute per-shift means and within-shift second moments for fast LOO
    c_bar = C_shifts.mean(axis=1)     # (R, K)
    f_bar = F_shifts.mean(axis=1) - 1.0  # (R,)

    if Y_cv_shifts is None or cv_mean is None:
        # Use only F_T - 1 control variate
        dc = C_shifts - c_bar[:, None, :]
        df = F_shifts - F_shifts.mean(axis=1)[:, None]
        cov_cf = np.mean(dc * df[:, :, None], axis=1)   # (R, K)
        var_f = np.mean(df * df, axis=1)                 # (R,)
        for r in range(R):
            mask = np.arange(R) != r
            vf = float(np.mean(var_f[mask]))
            beta_f = np.mean(cov_cf[mask], axis=0) / vf if vf > 1e-18 else np.zeros(K)
            shift_means[r, :] = c_bar[r, :] - beta_f * f_bar[r]
        return shift_means

    y_bar = Y_cv_shifts.mean(axis=1) - cv_mean[None, :]  # (R, K)
    dc = C_shifts - C_shifts.mean(axis=1)[:, None, :]
    dy = Y_cv_shifts - Y_cv_shifts.mean(axis=1)[:, None, :]
    df = F_shifts - F_shifts.mean(axis=1)[:, None]

    # Per-shift sufficient statistics for 2-variable OLS [dy, df] -> dc
    s_yy = np.mean(dy * dy, axis=1)                # (R, K)
    s_ff = np.mean(df * df, axis=1)[:, None]       # (R, 1)
    s_yf = np.mean(dy * df[:, :, None], axis=1)    # (R, K)
    s_cy = np.mean(dc * dy, axis=1)                # (R, K)
    s_cf = np.mean(dc * df[:, :, None], axis=1)    # (R, K)

    for r in range(R):
        mask = np.arange(R) != r
        ayy = np.mean(s_yy[mask], axis=0)
        aff = float(np.mean(s_ff[mask]))
        ayf = np.mean(s_yf[mask], axis=0)
        acy = np.mean(s_cy[mask], axis=0)
        acf = np.mean(s_cf[mask], axis=0)

        det = ayy * aff - ayf * ayf
        beta_y = np.zeros(K, dtype=np.float64)
        beta_f = np.zeros(K, dtype=np.float64)
        good = det > 1e-20
        beta_y[good] = (aff * acy[good] - ayf[good] * acf[good]) / det[good]
        beta_f[good] = (ayy[good] * acf[good] - ayf[good] * acy[good]) / det[good]
        # Fallback to 1-var if collinear
        only_y = (~good) & (ayy > 1e-18)
        beta_y[only_y] = acy[only_y] / ayy[only_y]

        shift_means[r, :] = c_bar[r, :] - beta_y * y_bar[r, :] - beta_f * f_bar[r]

    return shift_means


def _fit_cv_weights_from_pilot(
    C_pilot: np.ndarray,
    Y_cv_pilot: Optional[np.ndarray],
    F_pilot: np.ndarray,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Fit OLS dual control-variate weights (beta_y, beta_f) strictly on independent pilot paths.

    Shapes: C_pilot (M_p, K), Y_cv_pilot (M_p, K) or None, F_pilot (M_p,).
    Because these weights are fixed BEFORE evaluating the R independent RQMC shifts,
    the R resulting shift estimators are STRICTLY i.i.d. and 100% unbiased (exact Student-t df=R-1).
    """
    K = C_pilot.shape[1]
    dc = C_pilot - C_pilot.mean(axis=0, keepdims=True)
    df = F_pilot - F_pilot.mean()
    aff = float(np.mean(df * df))
    acf = np.mean(dc * df[:, None], axis=0)

    if Y_cv_pilot is None:
        beta_f = acf / aff if aff > 1e-18 else np.zeros(K, dtype=np.float64)
        return None, beta_f

    dy = Y_cv_pilot - Y_cv_pilot.mean(axis=0, keepdims=True)
    ayy = np.mean(dy * dy, axis=0)
    ayf = np.mean(dy * df[:, None], axis=0)
    acy = np.mean(dc * dy, axis=0)

    det = ayy * aff - ayf * ayf
    beta_y = np.zeros(K, dtype=np.float64)
    beta_f = np.zeros(K, dtype=np.float64)
    good = det > 1e-20
    beta_y[good] = (aff * acy[good] - ayf[good] * acf[good]) / det[good]
    beta_f[good] = (ayy[good] * acf[good] - ayf[good] * acy[good]) / det[good]
    only_y = (~good) & (ayy > 1e-18)
    beta_y[only_y] = acy[only_y] / ayy[only_y]
    return beta_y, beta_f


def _apply_fixed_cv(
    C_shifts: np.ndarray,
    Y_cv_shifts: Optional[np.ndarray],
    cv_mean: Optional[np.ndarray],
    F_shifts: np.ndarray,
    beta_y: Optional[np.ndarray],
    beta_f: np.ndarray,
) -> np.ndarray:
    """Apply fixed pilot-fitted CV weights to R independent shifts -> strictly i.i.d. shift means (R, K)."""
    c_bar = C_shifts.mean(axis=1)           # (R, K)
    f_bar = F_shifts.mean(axis=1) - 1.0     # (R,)
    out = c_bar - f_bar[:, None] * beta_f[None, :]
    if Y_cv_shifts is not None and cv_mean is not None and beta_y is not None:
        y_bar = Y_cv_shifts.mean(axis=1) - cv_mean[None, :]
        out = out - y_bar * beta_y[None, :]
    return out


def compute_bridge_envelope(
    shift_means_lvl: np.ndarray,
    levels_list: List[int],
    H: float,
    T: float,
    strikes_arr: np.ndarray,
    alpha: float = 0.05,
    safety_factor: float = 1.5,
    incr_shifts_override: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Unified RoughCert-SPX-Bridge envelope calculator (Theorem 4; shared by E1, E2, E3, E4, E6).

    Implements Theorem 4's strict 3-sigma contraction gates:
      1. Zero upward rate clamping on certified asymptotic-envelope strikes (Cases (i)-(ii)).
      2. Explicit sign-flip detection between consecutive increments Delta_{L-1} and Delta_L.
      3. Strict contraction gate (ratio > 1.15, beta_pair >= 0.5*beta_fsw, beta_emp >= 0.5*beta_fsw);
         otherwise flags the strike explicitly as 'diagnostic' with positive display fallback rate.
      4. Distinct epistemic label 'asymptotic-envelope statistical(1-a)' (separate from
         deterministic 176-bit Arb-certified bounds of RoughCert-SPX-Det).
      5. Exact no-arbitrage clipping [(1-K)_+, 1] propagating upper >= 1 cleanly to +inf IV.
      6. Supports increment-optimal pilot-fitted zero-mean CV via incr_shifts_override.
    """
    L_count, n_shifts, K_len = shift_means_lvl.shape
    t_crit_joint = float(student_t.ppf(1.0 - alpha / 4.0, df=n_shifts - 1))
    t_crit_single = float(student_t.ppf(1.0 - alpha / 2.0, df=n_shifts - 1))

    level_means = shift_means_lvl.mean(axis=1)
    level_ses = shift_means_lvl.std(axis=1, ddof=1) / math.sqrt(n_shifts)

    incr_shifts = incr_shifts_override if incr_shifts_override is not None else (shift_means_lvl[1:] - shift_means_lvl[:-1])
    incr_means = incr_shifts.mean(axis=1)
    incr_ses = incr_shifts.std(axis=1, ddof=1) / math.sqrt(n_shifts)

    beta_fsw = min(3.0 * H + 0.5, 1.0)
    finest_mean = level_means[-1]
    finest_se = level_ses[-1]
    sampling_half = t_crit_joint * finest_se

    bias_envelope = np.empty(K_len, dtype=np.float64)
    status_per_strike: List[str] = []
    emp_rate_per_strike: List[Optional[float]] = []
    beta_used_per_strike: List[float] = []
    sign_flip_per_strike: List[bool] = []

    for k_idx in range(K_len):
        s_last = float(incr_means[-1, k_idx])
        s_prev = float(incr_means[-2, k_idx])
        d_last = abs(s_last)
        se_last = float(incr_ses[-1, k_idx])
        d_prev = abs(s_prev)
        se_prev = float(incr_ses[-2, k_idx])

        d_last_hi = d_last + t_crit_joint * se_last
        # Two independent increments at SNR 3 give a delta-method log2-ratio SE of about 0.68.
        z_res = 3.0
        resolved_last = d_last > z_res * se_last
        resolved_prev = d_prev > z_res * se_prev
        sign_flip = (s_last * s_prev < 0.0) and resolved_last and (d_prev > 2.0 * se_prev)
        sign_flip_per_strike.append(bool(sign_flip))

        if sign_flip and resolved_last:
            # Statistically resolved sign flip => non-asymptotic regime => diagnostic.
            beta_emp = math.log2(max(d_prev / max(d_last, 1e-15), 1e-6)) if resolved_prev else None
            beta_used = min(beta_fsw, max(0.20, 0.85 * beta_emp)) if (beta_emp is not None and beta_emp > 0.0) else 0.5 * beta_fsw
            d_eff_hi = max(d_last_hi, d_prev / (2.0 ** beta_used))
            status_k = "diagnostic"
            emp_rate_per_strike.append(beta_emp)
        elif resolved_last and resolved_prev:
            ratio = d_prev / max(d_last, 1e-15)
            beta_pair = math.log2(max(ratio, 1e-6))
            # If >= 3 increments exist and all resolved increments are same-sign & monotonically decaying,
            # use SNR-weighted log-log regression when the finest pair ALSO passes the strict contraction gate.
            beta_emp = beta_pair
            if L_count - 1 >= 3:
                d_all = np.abs(incr_means[:, k_idx])
                se_all = incr_ses[:, k_idx]
                s_all = incr_means[:, k_idx]
                res_mask = (d_all > z_res * se_all) & (d_all > 1e-15)
                if int(np.sum(res_mask)) >= 3:
                    idx_r = np.where(res_mask)[0]
                    same_sign = bool(np.all(s_all[idx_r] * s_last > 0.0))
                    monotone_dec = bool(np.all(d_all[idx_r[:-1]] > d_all[idx_r[1:]]))
                    if same_sign and monotone_dec and ratio > 1.15 and beta_pair >= 0.5 * beta_fsw:
                        x_r = np.log2(np.asarray(levels_list[:-1], dtype=np.float64)[idx_r])
                        y_r = np.log2(d_all[idx_r])
                        w_r = np.minimum(d_all[idx_r] / np.maximum(se_all[idx_r], 1e-15), 1e3)
                        beta_reg = float(-np.polyfit(x_r, y_r, deg=1, w=w_r)[0])
                        beta_emp = beta_reg
            emp_rate_per_strike.append(beta_emp)
            if ratio > 1.15 and beta_pair >= 0.5 * beta_fsw and beta_emp >= 0.5 * beta_fsw:
                # Genuine monotone contraction consistent with FSW asymptotic regime (no upward floor clamp)
                beta_used = min(beta_fsw, 0.85 * beta_emp)
                d_eff_hi = d_last_hi
                status_k = f"asymptotic-envelope statistical({1.0 - alpha:.2f})"
            else:
                # Slow contraction or non-contracting => flag diagnostic; numerical display rate keeps 2^beta - 1 > 0
                beta_used = min(beta_fsw, max(0.15, 0.85 * max(beta_pair, 0.0)))
                d_eff_hi = d_last_hi
                status_k = "diagnostic"
        elif not resolved_last:
            # Finest increment submerged below 3*SE sampling noise
            emp_rate_per_strike.append(None)
            beta_used = beta_fsw
            # Guard against zero-crossing pinch when previous increment was large
            d_eff_hi = max(d_last_hi, (d_prev + t_crit_joint * se_prev) / (2.0 ** beta_fsw)) if resolved_prev else d_last_hi
            if d_last <= d_prev + 2.0 * (se_last + se_prev):
                status_k = f"asymptotic-envelope statistical({1.0 - alpha:.2f})"
            else:
                status_k = "diagnostic"
        else:
            # resolved_last is True, resolved_prev is False (increment grew out of noise => diagnostic!)
            emp_rate_per_strike.append(None)
            beta_used = 0.5 * beta_fsw
            d_eff_hi = d_last_hi
            status_k = "diagnostic"

        b_tail = safety_factor * d_eff_hi / (2.0 ** beta_used - 1.0)
        bias_envelope[k_idx] = b_tail
        beta_used_per_strike.append(float(beta_used))
        status_per_strike.append(status_k)

    total_half = sampling_half + bias_envelope
    price_lo = np.maximum(finest_mean - total_half, np.maximum(1.0 - strikes_arr, 0.0))
    price_hi = np.minimum(finest_mean + total_half, 1.0)

    iv_mid, iv_total_lo, iv_total_hi, iv_samp_lo, iv_samp_hi = [], [], [], [], []
    for k_idx, K_val in enumerate(strikes_arr):
        iv_mid.append(bs_iv_point_arb(float(finest_mean[k_idx]), float(K_val), T, F=1.0))
        tl, th = bs_iv_interval_from_price_bounds(float(price_lo[k_idx]), float(price_hi[k_idx]), float(K_val), T)
        iv_total_lo.append(tl)
        iv_total_hi.append(th)
        sl_p = max(float(finest_mean[k_idx] - t_crit_single * finest_se[k_idx]), max(1.0 - float(K_val), 0.0))
        sh_p = min(float(finest_mean[k_idx] + t_crit_single * finest_se[k_idx]), 1.0)
        sl, sh = bs_iv_interval_from_price_bounds(sl_p, sh_p, float(K_val), T)
        iv_samp_lo.append(sl)
        iv_samp_hi.append(sh)

    return {
        "levels": levels_list,
        "alpha": alpha,
        "safety_factor": safety_factor,
        "beta_fsw_theory": beta_fsw,
        "level_means": level_means.tolist(),
        "level_ses": level_ses.tolist(),
        "incr_means": incr_means.tolist(),
        "incr_ses": incr_ses.tolist(),
        "finest_price_mean": finest_mean.tolist(),
        "sampling_half_price": sampling_half.tolist(),
        "bias_envelope_price": bias_envelope.tolist(),
        "continuum_price_lo": price_lo.tolist(),
        "continuum_price_hi": price_hi.tolist(),
        "iv_mid": iv_mid,
        "iv_sampling_lo": iv_samp_lo,
        "iv_sampling_hi": iv_samp_hi,
        "iv_continuum_lo": iv_total_lo,
        "iv_continuum_hi": iv_total_hi,
        "iv_sampling_width_bps": [math.inf if not math.isfinite(h) else 1e4 * (h - l) for l, h in zip(iv_samp_lo, iv_samp_hi)],
        "iv_continuum_width_bps": [math.inf if not math.isfinite(h) else 1e4 * (h - l) for l, h in zip(iv_total_lo, iv_total_hi)],
        "emp_rate_per_strike": emp_rate_per_strike,
        "beta_used_per_strike": beta_used_per_strike,
        "sign_flip_per_strike": sign_flip_per_strike,
        "status_per_strike": status_per_strike,
    }


def price_spx_single_level(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_fn: Callable[[np.ndarray], np.ndarray],
    strikes: Sequence[float],
    n: int,
    method: str = "ce_i_only",
    n_shifts: int = 16,
    n_half: int = 1024,
    base_seed: int = 202601,
    alpha: float = 0.05,
    cache: Optional[VolterraGridCache] = None,
    conserve_variance: bool = True,
    n_pilot_shifts: int = 2,
) -> Dict[str, object]:
    """Price SPX Call strip on a single exact grid n with scrambled Sobol (LMS + digital shift) RQMC + exact pilot-fitted CV.

    method in {'ce_i_only' (default RoughCert-SPX-CE), 'leftpoint', 'ce_adapted' (Jensen-J ablation)}.
    Uses n_pilot_shifts independent pilot shifts for CV weight estimation so the R main shifts
    are STRICTLY i.i.d. with exact Student-t df=R-1 coverage.
    Reports cold (setup) and amortised (solve) wall-clock times separately.
    """
    strikes_arr = np.asarray(strikes, dtype=np.float64)
    K_len = len(strikes_arr)

    if cache is None:
        cache = build_volterra_grid_cache(H, T, n, need_ce=(method in ("ce_adapted", "ce_i_only")))
    cold_setup_s = cache.setup_seconds

    t_solve0 = time.perf_counter()

    def _eval_one(Z_mat: np.ndarray) -> Dict[str, np.ndarray]:
        dW, Y = generate_exact_paths(cache, Z_mat)
        if method == "ce_i_only":
            return evaluate_ce_i_only_paths(cache, dW, Y, eta, rho, xi0_fn, strikes_arr, build_cv_bank=True)
        if method == "ce_adapted":
            return evaluate_ce_adapted_paths(
                cache, dW, Y, eta, rho, xi0_fn, strikes_arr,
                conserve_variance=conserve_variance, build_cv_bank=True,
            )
        if method == "leftpoint":
            return evaluate_leftpoint_paths(dW, Y, H, T, eta, rho, xi0_fn, strikes_arr, build_cv_bank=True)
        raise ValueError(f"Unknown method: {method}")

    M = 2 * n_half
    # Independent pilot shifts to fit exact zero-mean multi-CV weights strictly outside main R shifts
    C_p_list, X_p_list = [], []
    for p_idx in range(n_pilot_shifts):
        Zp = sample_rqmc_normals(dim=2 * n, n_half=max(256, n_half // 2), seed=base_seed + 900000 + p_idx, antithetic=True)
        rp = _eval_one(Zp)
        C_p_list.append(rp["C_paths"])
        X_p_list.append(rp["X_cv"])
    W_cv = _fit_multicv_pilot(np.vstack(C_p_list), np.vstack(X_p_list))

    shift_means = np.empty((n_shifts, K_len), dtype=np.float64)
    F_means = np.empty(n_shifts, dtype=np.float64)
    for r in range(n_shifts):
        Z = sample_rqmc_normals(dim=2 * n, n_half=n_half, seed=base_seed + r, antithetic=True)
        res = _eval_one(Z)
        c_bar = res["C_paths"].mean(axis=0)        # (K,)
        x_bar = res["X_cv"].mean(axis=0)           # (K, P)
        shift_means[r] = c_bar - np.sum(x_bar * W_cv, axis=1)
        F_means[r] = float(res["F_T"].mean())
    solve_s = time.perf_counter() - t_solve0

    mean_price = shift_means.mean(axis=0)
    se_price = shift_means.std(axis=0, ddof=1) / math.sqrt(n_shifts)
    t_crit = float(student_t.ppf(1.0 - alpha / 2.0, df=n_shifts - 1))
    ci_half = t_crit * se_price

    if not (np.all(np.isfinite(mean_price)) and np.all(np.isfinite(se_price))):
        raise ArithmeticError("Non-finite price or SE encountered.")

    iv_mid, iv_lo, iv_hi = [], [], []
    for k_idx, K_val in enumerate(strikes_arr):
        p_l = max(float(mean_price[k_idx] - ci_half[k_idx]), max(1.0 - float(K_val), 0.0))
        p_u = min(float(mean_price[k_idx] + ci_half[k_idx]), 1.0)
        il, ih = bs_iv_interval_from_price_bounds(p_l, p_u, float(K_val), T, F=1.0)
        iv_lo.append(il)
        iv_hi.append(ih)
        iv_mid.append(bs_iv_point_arb(float(mean_price[k_idx]), float(K_val), T, F=1.0))

    return {
        "status": f"simulated-RQMC({1.0 - alpha:.2f})",
        "sampling": "scrambled-Sobol RQMC + pilot-fitted 6-control bank",
        "method": method,
        "n": n,
        "n_shifts": n_shifts,
        "paths_per_shift": M,
        "total_paths": n_shifts * M,
        "base_seed": base_seed,
        "cold_setup_s": cold_setup_s,
        "solve_s": solve_s,
        "strikes": strikes_arr.tolist(),
        "price_mean": mean_price.tolist(),
        "price_se": se_price.tolist(),
        "price_ci_half": ci_half.tolist(),
        "iv_mid": iv_mid,
        "iv_ci_lo": iv_lo,
        "iv_ci_hi": iv_hi,
        "martingale_check_mean_F": float(F_means.mean()),
        "martingale_check_se_F": float(F_means.std(ddof=1) / math.sqrt(n_shifts)),
    }


def price_spx_bridge_certified(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_fn: Callable[[np.ndarray], np.ndarray],
    strikes: Sequence[float],
    levels: Sequence[int] = (64, 128, 256, 512),
    method: str = "ce_i_only",
    n_shifts: int = 16,
    n_half: int = 1024,
    base_seed: int = 202601,
    alpha: float = 0.05,
    safety_factor: float = 1.5,
    n_pilot_shifts: int = 2,
) -> Dict[str, object]:
    """RoughCert-SPX-Bridge: Exact Nested-Path Coupled Multilevel Envelope Pricer.

    Defaults to method='ce_i_only' (Channel-I CE projection + exact-variance F_{t_i}-adapted J_T),
    which preserves the full FSW (2025) weak rate min(3H+1/2, 1) with zero correlation attenuation.
    Uses independent pilot shifts for CV fitting so the R shift means are strictly i.i.d.
    """
    levels_list = sorted(int(x) for x in levels)
    if len(levels_list) < 3:
        raise ValueError("RoughCert-SPX-Bridge requires at least 3 nested dyadic levels.")
    for a, b in zip(levels_list[:-1], levels_list[1:]):
        if b != 2 * a:
            raise ValueError(f"Levels must be dyadic doubling: got {levels_list}")

    strikes_arr = np.asarray(strikes, dtype=np.float64)
    K_len = len(strikes_arr)
    n_max = levels_list[-1]
    L_count = len(levels_list)

    t_setup0 = time.perf_counter()
    caches = {
        n_lvl: build_volterra_grid_cache(H, T, n_lvl, need_ce=(method in ("ce_adapted", "ce_i_only")))
        for n_lvl in levels_list
    }
    cold_setup_s = time.perf_counter() - t_setup0

    t_solve0 = time.perf_counter()

    def _eval_lvl(idx_l: int, n_lvl: int, dW_l: np.ndarray, Y_l: np.ndarray) -> Dict[str, np.ndarray]:
        if method == "ce_i_only":
            return evaluate_ce_i_only_paths(caches[n_lvl], dW_l, Y_l, eta, rho, xi0_fn, strikes_arr, build_cv_bank=True)
        if method == "ce_adapted":
            return evaluate_ce_adapted_paths(caches[n_lvl], dW_l, Y_l, eta, rho, xi0_fn, strikes_arr, build_cv_bank=True)
        if method == "leftpoint":
            return evaluate_leftpoint_paths(dW_l, Y_l, H, T, eta, rho, xi0_fn, strikes_arr, build_cv_bank=True)
        raise ValueError(f"Unknown method: {method}")

    # 1. Fit per-level AND increment-optimal 6-control weights strictly on independent pilot shifts.
    #    Every control in X_cv (and its coupled difference dX_cv) has 100% exact zero expectation,
    #    so R main shift means and coupled increments are STRICTLY i.i.d. and 100% unbiased.
    p_C = [[] for _ in range(L_count)]
    p_X = [[] for _ in range(L_count)]
    pilot_half = max(512, n_half // 2)
    for p_idx in range(n_pilot_shifts):
        Zp = sample_rqmc_normals(dim=2 * n_max, n_half=pilot_half, seed=base_seed + 900000 + p_idx, antithetic=True)
        dW_f, Y_f = generate_exact_paths(caches[n_max], Zp)
        for idx_l, n_lvl in enumerate(levels_list):
            stride = n_max // n_lvl
            dW_l, Y_l = (dW_f, Y_f) if stride == 1 else coarsen_paths(dW_f, Y_f, stride=stride)
            res = _eval_lvl(idx_l, n_lvl, dW_l, Y_l)
            p_C[idx_l].append(res["C_paths"])
            p_X[idx_l].append(res["X_cv"])

    p_C_st = [np.vstack(p_C[l]) for l in range(L_count)]
    p_X_st = [np.vstack(p_X[l]) for l in range(L_count)]
    W_lvl = [_fit_multicv_pilot(p_C_st[l], p_X_st[l]) for l in range(L_count)]
    W_inc = [
        _fit_multicv_pilot(p_C_st[l + 1] - p_C_st[l], p_X_st[l + 1] - p_X_st[l])
        for l in range(L_count - 1)
    ]

    # 2. Evaluate R strictly independent RQMC shifts across all nested levels (low memory!)
    M = 2 * n_half
    shift_means_lvl = np.empty((L_count, n_shifts, K_len), dtype=np.float64)
    incr_shifts = np.empty((L_count - 1, n_shifts, K_len), dtype=np.float64)
    F_finest_means = np.empty(n_shifts, dtype=np.float64)

    for r in range(n_shifts):
        Z = sample_rqmc_normals(dim=2 * n_max, n_half=n_half, seed=base_seed + r, antithetic=True)
        dW_fine, Y_fine = generate_exact_paths(caches[n_max], Z)
        c_bars = []
        x_bars = []
        for idx_l, n_lvl in enumerate(levels_list):
            stride = n_max // n_lvl
            dW_l, Y_l = (dW_fine, Y_fine) if stride == 1 else coarsen_paths(dW_fine, Y_fine, stride=stride)
            res = _eval_lvl(idx_l, n_lvl, dW_l, Y_l)
            cb = res["C_paths"].mean(axis=0)
            xb = res["X_cv"].mean(axis=0)
            c_bars.append(cb)
            x_bars.append(xb)
            shift_means_lvl[idx_l, r] = cb - np.sum(xb * W_lvl[idx_l], axis=1)
            if idx_l == L_count - 1:
                F_finest_means[r] = float(res["F_T"].mean())
        for idx_l in range(L_count - 1):
            dcb = c_bars[idx_l + 1] - c_bars[idx_l]
            dxb = x_bars[idx_l + 1] - x_bars[idx_l]
            incr_shifts[idx_l, r] = dcb - np.sum(dxb * W_inc[idx_l], axis=1)
    solve_s = time.perf_counter() - t_solve0

    env = compute_bridge_envelope(
        shift_means_lvl=shift_means_lvl,
        levels_list=levels_list,
        H=H,
        T=T,
        strikes_arr=strikes_arr,
        alpha=alpha,
        safety_factor=safety_factor,
        incr_shifts_override=incr_shifts,
    )
    return {
        "method": f"RoughCert-SPX-Bridge ({method})",
        "H": H,
        "T": T,
        "eta": eta,
        "rho": rho,
        "n_shifts": n_shifts,
        "paths_per_shift": M,
        "total_paths": n_shifts * M,
        "base_seed": base_seed,
        "cold_setup_s": cold_setup_s,
        "solve_s": solve_s,
        "strikes": strikes_arr.tolist(),
        "martingale_check_mean_F": float(F_finest_means.mean()),
        **env,
    }
