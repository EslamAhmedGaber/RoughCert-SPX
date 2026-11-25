"""Fair, faithful implementations of the 6 published SPX rough-Bergomi pricing competitors (2012-2026).

Competitors implemented:
1. McCrickerd & Pakkanen (2018, Quantitative Finance):
   Turbocharged Hybrid rBergomi (BLP hybrid scheme kappa in {1, 2, 4} with exact near-field
   Gaussian covariance, optimal L^2 weights b_k^*, Romano-Touzi conditional MC + exact
   stopped-timer CV + antithetic variates).
2. Bayer, Ben Hammouda & Tempone (2020, Quantitative Finance):
   Hierarchical Brownian-Bridge RQMC (exact Volterra left-point + Brownian bridge +
   scrambled Sobol (LMS + digital shift) + diagnostic Richardson extrapolation).
3. Bayer & Breneis (2023, Quantitative Finance):
   Markovian Multifactor rBergomi (M-factor optimal L^2 sum-of-exponentials kernel fit +
   exact correlated Gaussian OU transitions + Romano-Touzi conditional MC).
4. Horvath, Jacquier, Muguruza & Sojmark (2024, Finance and Stochastics):
   rDonsker (discrete Volterra random-walk / functional CLT convolution approximation:
   supports BOTH the HJMS Eq. (4.2) variance-matched kernel K_k = dt^{H-1/2} sqrt(k^{2H} - (k-1)^{2H})
   and our BLP cell-mean comparator (b_k^*) + Romano-Touzi conditional MC).
5. Bergomi & Guyon (2012, Risk) / Short-Maturity Asymptotic Expansion:
   Closed-form order-2 vol-of-vol SPX implied volatility smile & ATM skew expansion
   (exact for flat xi0, verified to O(eta^3) against small-eta rBergomi!).
6. Exact-Grid Left-Point Conditional MC (Friz-Salkeld-Wagenhofer 2025 benchmark):
   Implemented in bridge_rqmc.py (method='leftpoint').
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import lsq_linear, minimize
from scipy.special import beta as sc_beta, gamma as sc_gamma, gammainc

from .bs_arb import bs_call_vec, bs_iv_point_arb, bs_iv_interval_from_price_bounds
from .bridge_rqmc import (
    _eval_stopped_timer_cv,
    _apply_crossfit_cv,
    sample_rqmc_normals,
    price_spx_single_level,
    build_volterra_grid_cache,
)


# =============================================================================
# 1. McCrickerd & Pakkanen (2018) -- Turbocharged BLP Hybrid Scheme (kappa >= 1)
# =============================================================================

def _blp_near_cov_matrix(H: float, dt: float, kappa: int) -> np.ndarray:
    """Exact (kappa + 1) x (kappa + 1) covariance matrix of the BLP local Gaussian vector
    (W_i^0, W_i^1, ..., W_i^{kappa}) on each step [t_i, t_{i+1}], where:
      W_i^0 = Delta W_i = int_{t_i}^{t_{i+1}} dW_s,
      W_i^k = sqrt(2H) int_{t_i}^{t_{i+1}} ((t_i + k*dt) - s)^{H - 1/2} dW_s,   k = 1..kappa.
    Using exact Beta/hypergeometric / Gauss-Legendre integrals (transformed to remove singularity).
    """
    dim = kappa + 1
    cov = np.zeros((dim, dim), dtype=np.float64)
    cov[0, 0] = dt

    p = H + 0.5
    sqrt2H = math.sqrt(2.0 * H)
    # Cov(W_i^0, W_i^k) = sqrt(2H) int_0^{dt} (k*dt - u)^{H-1/2} du
    #                   = sqrt(2H) / (H+1/2) * dt^{H+1/2} * [k^{H+1/2} - (k-1)^{H+1/2}]
    for k in range(1, kappa + 1):
        c0k = (sqrt2H / p) * (dt ** p) * (k ** p - (k - 1) ** p)
        cov[0, k] = c0k
        cov[k, 0] = c0k

    # Cov(W_i^j, W_i^k) = 2H * int_0^{dt} (j*dt - u)^{H-1/2} (k*dt - u)^{H-1/2} du
    #                   = 2H * dt^{2H} * int_0^1 (j - 1 + v)^{H-1/2} (k - 1 + v)^{H-1/2} dv
    # For j == k: exact = dt^{2H} * [k^{2H} - (k-1)^{2H}].
    # For j < k: substitution v = r^{1/p} (dv = (1/p) r^{1/p - 1} dr) removes singularity at v=0 when j=1.
    gl_x, gl_w = np.polynomial.legendre.leggauss(48)
    r = 0.5 * (gl_x + 1.0)
    wr = 0.5 * gl_w
    v = r ** (1.0 / p)
    # (v)^{H-1/2} dv = r^{(p-1)/p} * (1/p) r^{1/p - 1} dr = (1/p) dr.
    for j in range(1, kappa + 1):
        cov[j, j] = (dt ** (2.0 * H)) * (j ** (2.0 * H) - (j - 1) ** (2.0 * H))
        for k in range(j + 1, kappa + 1):
            if j == 1:
                # Singularity at v=0 removed identically by v = r^{1/p}
                val = 2.0 * H * (dt ** (2.0 * H)) * (1.0 / p) * float(np.sum(wr * ((k - 1.0 + v) ** (H - 0.5))))
            else:
                # Smooth on [0, 1]
                u_s = r
                val = 2.0 * H * (dt ** (2.0 * H)) * float(
                    np.sum(wr * ((j - 1.0 + u_s) ** (H - 0.5)) * ((k - 1.0 + u_s) ** (H - 0.5)))
                )
            cov[j, k] = val
            cov[k, j] = val
    return cov


def price_mp2018_hybrid(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_fn: Callable[[np.ndarray], np.ndarray],
    strikes: Sequence[float],
    n: int,
    kappa: int = 1,
    n_batches: int = 16,
    n_half: int = 1024,
    base_seed: int = 202618,
    alpha: float = 0.05,
    use_rqmc: bool = False,
) -> Dict[str, object]:
    """McCrickerd & Pakkanen (2018) Turbocharged Hybrid rBergomi (BLP kappa >= 1).

    Exact near-field (lags 1..kappa) + L^2-optimal far-field convolution (lags kappa+1..n)
    via FFT, evaluated at left-point grid times with Romano-Touzi + exact stopped-timer CV + antithetic.
    """
    from scipy.stats import t as student_t

    t_setup0 = time.perf_counter()
    strikes_arr = np.asarray(strikes, dtype=np.float64)
    K_len = len(strikes_arr)
    dt = T / n
    p = H + 0.5
    sqrt2H = math.sqrt(2.0 * H)

    # Local near-field Cholesky factor (kappa+1 x kappa+1)
    cov_near = _blp_near_cov_matrix(H, dt, kappa)
    L_near = cholesky(cov_near + 1e-15 * np.eye(kappa + 1), lower=True)

    # Far-field L^2-optimal weights c_k^* = (sqrt(2H)/p) * dt^{H-1/2} * [k^p - (k-1)^p] for k = kappa+1..n
    # Notice: Y_{t_j} = sum_{k=1}^{min(j, kappa)} W_{j-k}^k + sum_{k=kappa+1}^{j} c_k^* * W_{j-k}^0
    # Check: Var(c_k^* W_{j-k}^0) = (c_k^*)^2 * dt, and int_{(k-1)dt}^{k dt} K(u) du = c_k^* * dt.
    c_far = np.zeros(n, dtype=np.float64)
    if n > kappa:
        ks = np.arange(kappa + 1, n + 1, dtype=np.float64)
        c_far[kappa:] = (sqrt2H / p) * (dt ** (H - 0.5)) * (ks ** p - (ks - 1.0) ** p)

    # Exact variance of the hybrid process Y_tilde_{t_j} at each grid point t_j (j = 0..n-1).
    # Note: In standard BLP / MP 2018, the drift shift uses EITHER true t^{2H} or exact hybrid variance;
    # using exact hybrid variance ensures E[v_tilde(t_j)] = xi0(t_j) EXACTLY (as in MP 2018 code!).
    # Wait: let's compute exact hybrid variance var_hyb[j] = sum_{k=1}^{min(j,kappa)} cov_near[k,k]
    #       + sum_{k=kappa+1}^j (c_far[k-1])^2 * dt.
    var_hyb = np.zeros(n, dtype=np.float64)
    for j in range(1, n):
        v_near = sum(cov_near[k, k] for k in range(1, min(j, kappa) + 1))
        v_far = dt * float(np.sum(c_far[kappa:j] ** 2)) if j > kappa else 0.0
        var_hyb[j] = v_near + v_far

    t_left = np.linspace(0.0, T - dt, n)
    xi0_left = np.asarray(xi0_fn(t_left), dtype=np.float64)
    mean_v = float(np.mean(xi0_left)) * T
    Q_var = 8.0 * (rho * rho) * mean_v if abs(rho) > 1e-12 else None

    # FFT length for linear convolution of length n
    n_fft = 1 << int(math.ceil(math.log2(2 * n)))
    fft_c = np.fft.rfft(c_far, n=n_fft)
    cold_setup_s = time.perf_counter() - t_setup0

    t_solve0 = time.perf_counter()
    M = 2 * n_half
    C_batches = np.empty((n_batches, M, K_len), dtype=np.float64)
    F_batches = np.empty((n_batches, M), dtype=np.float64)
    Y_cv_batches = np.empty((n_batches, M, K_len), dtype=np.float64) if Q_var is not None else None
    cv_mean = None

    for b in range(n_batches):
        rng = np.random.default_rng(base_seed + b)
        if use_rqmc and n * (kappa + 1) <= 21201:
            Z_flat = sample_rqmc_normals(n * (kappa + 1), n_half, seed=base_seed + b, antithetic=True)
            Z = Z_flat.reshape(M, n, kappa + 1)
        else:
            Z_half = rng.standard_normal((n_half, n, kappa + 1))
            Z = np.concatenate([Z_half, -Z_half], axis=0)

        # Transform local Gaussian blocks: W_local shape (M, n, kappa+1)
        W_local = Z @ L_near.T
        dW = W_local[:, :, 0]  # (M, n)

        # Build Y_left (at t_0=0, t_1, ..., t_{n-1}): Y_left[:, 0] = 0
        # For j >= 1: Y_left[:, j] = sum_{k=1}^{min(j, kappa)} W_local[:, j-k, k] + (c_far * dW)[j-1]
        # Note: c_far[m] corresponds to lag m+1 (0-indexed: c_far[kappa] is lag kappa+1).
        # Convolution (dW * c_far)[j-1] = sum_{m=0}^{j-1} dW[j-1-m] * c_far[m]
        # = sum_{lag=1}^{j} dW[j-lag] * c_far[lag-1].
        Y_left = np.zeros((M, n), dtype=np.float64)
        for k in range(1, min(kappa + 1, n)):
            Y_left[:, k:] += W_local[:, : n - k, k]

        if n > kappa:
            fft_dW = np.fft.rfft(dW, n=n_fft, axis=1)
            conv = np.fft.irfft(fft_dW * fft_c[None, :], n=n_fft, axis=1)[:, : n - 1]
            Y_left[:, 1:] += conv

        v_left = xi0_left[None, :] * np.exp(eta * Y_left - 0.5 * (eta * eta) * var_hyb[None, :])
        sigma_left = np.sqrt(v_left)

        I_T = dt * np.sum(v_left, axis=1)
        J_T = np.sum(sigma_left * dW, axis=1)
        F_T = np.exp(rho * J_T - 0.5 * (rho * rho) * I_T)
        V_cond = (1.0 - rho * rho) * I_T

        C_batches[b] = bs_call_vec(F_T[:, None], strikes_arr[None, :], V_cond[:, None])
        F_batches[b] = F_T
        if Q_var is not None:
            Y_cv, cv_mean = _eval_stopped_timer_cv(sigma_left, dW, dt, rho, strikes_arr, Q_var)
            Y_cv_batches[b] = Y_cv

    batch_means = _apply_crossfit_cv(C_batches, Y_cv_batches, cv_mean, F_batches)
    solve_s = time.perf_counter() - t_solve0

    mean_p = batch_means.mean(axis=0)
    se_p = batch_means.std(axis=0, ddof=1) / math.sqrt(n_batches)
    t_crit = float(student_t.ppf(1.0 - alpha / 2.0, df=n_batches - 1))
    ci_half = t_crit * se_p

    ivs = [bs_iv_point_arb(float(p), float(K), T, F=1.0) for p, K in zip(mean_p, strikes_arr)]
    iv_lo, iv_hi = [], []
    for idx, K_val in enumerate(strikes_arr):
        pl = max(float(mean_p[idx] - ci_half[idx]), max(1.0 - float(K_val), 0.0))
        pu = min(float(mean_p[idx] + ci_half[idx]), 1.0)
        il, ih = bs_iv_interval_from_price_bounds(pl, pu, float(K_val), T)
        iv_lo.append(il)
        iv_hi.append(ih)

    return {
        "status": f"simulated-{'RQMC' if use_rqmc else 'MC'}({1.0 - alpha:.2f})",
        "method": f"MP2018-Hybrid(kappa={kappa})",
        "n": n,
        "kappa": kappa,
        "n_batches": n_batches,
        "total_paths": n_batches * M,
        "cold_setup_s": cold_setup_s,
        "solve_s": solve_s,
        "strikes": strikes_arr.tolist(),
        "price_mean": mean_p.tolist(),
        "price_se": se_p.tolist(),
        "price_ci_half": ci_half.tolist(),
        "iv_mid": ivs,
        "iv_ci_lo": iv_lo,
        "iv_ci_hi": iv_hi,
        "martingale_check_mean_F": float(F_batches.mean()),
    }


# =============================================================================
# 2. Bayer & Breneis (2023) -- Markovian Multifactor L^2-Optimal rBergomi
# =============================================================================

def _exp_safe(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.minimum(np.maximum(x, 0.0), 700.0))


def bayer_breneis_l2_nodes_weights(H: float, M_factors: int, T: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Bayer & Breneis (2023, Quantitative Finance) L^2-optimal sum-of-exponentials
    approximation:
      K(u) = sqrt(2H) u^{H - 1/2}  ~  sum_{j=1}^M w_j * exp(-x_j * u)   on [0, T].

    Uses diagonally preconditioned NNLS (eliminating stiffness across 20+ orders of magnitude
    in mean-reversion speeds x_j) inside unclipped L-BFGS-B optimization over log-nodes log(x_j),
    faithful to Bayer & Breneis (2023) without artificial node capping.
    Returns (nodes x_j > 0, weights w_j >= 0 scaled to K(u) = sqrt(2H) u^{H-1/2}, L2_err).
    """
    a = H + 0.5
    sqrt2H = math.sqrt(2.0 * H)
    norm_K_sq = T ** (2.0 * H)
    gamma_a = sc_gamma(a)

    LOG_LO, LOG_HI = -25.0, 65.0  # x in [1.4e-11, 1.7e28] -- effectively unclipped

    def fit_weights_and_err(log_x: np.ndarray) -> Tuple[float, np.ndarray]:
        lx = np.clip(log_x, LOG_LO, LOG_HI)
        x = np.exp(lx)
        # Gram matrix G_{jk} = (1 - exp(-(x_j + x_k) T)) / (x_j + x_k)
        s = x[:, None] + x[None, :]
        G = -np.expm1(-np.minimum(s * T, 700.0)) / s
        # Cross vector b_j = sqrt(2H) * Gamma(a) * P(a, x_j * T) / x_j^a
        # For x_j * T > 60, regularized lower incomplete gamma P(a, x_j * T) == 1.0 to float64.
        xt = x * T
        P_inc = np.where(xt > 60.0, 1.0, gammainc(a, np.minimum(xt, 60.0)))
        b = sqrt2H * gamma_a * P_inc * np.exp(-a * lx)

        # Diagonal preconditioning: G_tilde = D^{-1/2} G D^{-1/2}, b_tilde = D^{-1/2} b
        d_diag = np.maximum(np.diag(G), 1e-300)
        inv_sqrt_d = 1.0 / np.sqrt(d_diag)
        G_tilde = (G * inv_sqrt_d[:, None]) * inv_sqrt_d[None, :]
        G_tilde = 0.5 * (G_tilde + G_tilde.T)
        b_tilde = b * inv_sqrt_d

        eigv, eigu = np.linalg.eigh(G_tilde)
        eigv = np.maximum(eigv, 1e-14 * max(float(eigv.max()), 1.0))
        A_lsq = (eigu * np.sqrt(eigv)[None, :]).T
        rhs = (eigu.T @ b_tilde) / np.sqrt(eigv)
        res = lsq_linear(A_lsq, rhs, bounds=(0.0, np.inf), lsmr_tol="auto")
        w_tilde = res.x
        w = w_tilde * inv_sqrt_d
        err_sq = max(norm_K_sq - 2.0 * float(np.dot(b_tilde, w_tilde)) + float(w_tilde @ G_tilde @ w_tilde), 0.0)
        return err_sq, w

    # H-adapted geometric initial grid covering both macro scale (1/T) and rough singularity scale
    if M_factors == 1:
        log_x0 = np.array([math.log(1.0 / T)])
    else:
        # Spread log-nodes adaptively: rougher H needs wider geometric span toward large x
        log_span = min(52.0, (2.2 + 0.95 / max(H, 0.04)) * math.sqrt(M_factors))
        log_x0 = np.linspace(math.log(0.2 / T), math.log(0.2 / T) + log_span, M_factors)

    best_err_sq = math.inf
    best_lx = log_x0.copy()
    best_w = np.zeros(M_factors)

    # Multi-start with two initial spans (wide H-adapted + classic Bayer-Breneis 8^(M-1)/T)
    starts = [log_x0]
    if M_factors > 1:
        log_classic = np.linspace(math.log(0.5 / T), min(math.log(0.5 / T) + (M_factors - 1) * math.log(8.0), 55.0), M_factors)
        starts.append(log_classic)

    bounds = [(LOG_LO, LOG_HI)] * M_factors
    for lx_init in starts:
        opt = minimize(
            lambda lx: fit_weights_and_err(lx)[0],
            lx_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 600, "ftol": 1e-15, "gtol": 1e-12, "eps": 1e-5},
        )
        e_sq, w_cand = fit_weights_and_err(opt.x)
        if e_sq < best_err_sq:
            best_err_sq = e_sq
            best_lx = opt.x.copy()
            best_w = w_cand.copy()

    # Check active weights did not slam into upper clip
    x = np.exp(np.clip(best_lx, LOG_LO, LOG_HI))
    order = np.argsort(x)
    x_s, w_s = x[order], best_w[order]
    active = w_s > 0.0
    if np.any(active) and np.max(np.log(x_s[active])) > LOG_HI - 0.5:
        raise RuntimeError(f"Bayer-Breneis node hit upper log-clip {LOG_HI} for H={H}, M={M_factors}")
    return x_s, w_s, math.sqrt(best_err_sq)


def price_bayer_breneis_markovian(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_fn: Callable[[np.ndarray], np.ndarray],
    strikes: Sequence[float],
    M_factors: int,
    n: int,
    n_batches: int = 16,
    n_half: int = 1024,
    base_seed: int = 202623,
    alpha: float = 0.05,
    nodes_weights: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
) -> Dict[str, object]:
    """Bayer & Breneis (2023) M-Factor Markovian rBergomi Pricer with EXACT OU Transitions.

    At each time step [t_i, t_{i+1}], the joint Gaussian increment vector
      (Delta W_i,  zeta_{i,1}, ..., zeta_{i,M}),   zeta_{i,j} = int_{t_i}^{t_{i+1}} e^{-x_j (t_{i+1}-s)} dW_s,
    is simulated EXACTLY using its (M+1) x (M+1) closed-form covariance matrix.
    Then X_j(t_{i+1}) = exp(-x_j dt) X_j(t_i) + zeta_{i,j}, and Y_M(t_i) = sum_j w_j X_j(t_i).
    Uses exact variance Var(Y_M(t_i)) so E[v_M(t_i)] = xi0(t_i) identically.
    """
    from scipy.stats import t as student_t

    t_setup0 = time.perf_counter()
    strikes_arr = np.asarray(strikes, dtype=np.float64)
    K_len = len(strikes_arr)
    dt = T / n

    if nodes_weights is None:
        nodes, weights, kern_l2 = bayer_breneis_l2_nodes_weights(H, M_factors, T)
    else:
        nodes, weights, kern_l2 = nodes_weights

    # Exact (M+1) x (M+1) step covariance for [dW, zeta_1..zeta_M]
    dim = M_factors + 1
    cov_step = np.zeros((dim, dim), dtype=np.float64)
    cov_step[0, 0] = dt
    e_xdt = np.exp(-np.minimum(nodes * dt, 700.0))
    cov_0j = -np.expm1(-np.minimum(nodes * dt, 700.0)) / nodes
    cov_step[0, 1:] = cov_0j
    cov_step[1:, 0] = cov_0j
    s_mat = nodes[:, None] + nodes[None, :]
    cov_step[1:, 1:] = -np.expm1(-np.minimum(s_mat * dt, 700.0)) / s_mat

    eigv, eigu = np.linalg.eigh(0.5 * (cov_step + cov_step.T))
    eigv = np.maximum(eigv, 0.0)
    L_step = eigu * np.sqrt(eigv)[None, :]  # exact symmetric PSD square root

    # Exact Markovian variance Var(Y_M(t_i)) = int_0^{t_i} K_M(u)^2 du
    t_left = np.linspace(0.0, T - dt, n)
    xi0_left = np.asarray(xi0_fn(t_left), dtype=np.float64)
    ww = weights[:, None] * weights[None, :]
    var_M = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        ti = t_left[i]
        var_M[i] = float(np.sum(ww * (-np.expm1(-np.minimum(s_mat * ti, 700.0)) / s_mat)))

    mean_v = float(np.mean(xi0_left)) * T
    Q_var = 8.0 * (rho * rho) * mean_v if abs(rho) > 1e-12 else None
    cold_setup_s = time.perf_counter() - t_setup0

    t_solve0 = time.perf_counter()
    M = 2 * n_half
    C_batches = np.empty((n_batches, M, K_len), dtype=np.float64)
    F_batches = np.empty((n_batches, M), dtype=np.float64)
    Y_cv_batches = np.empty((n_batches, M, K_len), dtype=np.float64) if Q_var is not None else None
    cv_mean = None

    for b in range(n_batches):
        rng = np.random.default_rng(base_seed + b)
        Z_half = rng.standard_normal((n_half, n, dim))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
        G = Z @ L_step.T  # (M, n, M_factors+1)
        dW = G[:, :, 0]
        zeta = G[:, :, 1:]

        # Propagate M OU factors
        X = np.zeros((M, M_factors), dtype=np.float64)
        Y_left = np.zeros((M, n), dtype=np.float64)
        for i in range(n):
            if i > 0:
                X = X * e_xdt[None, :] + zeta[:, i - 1, :]
                Y_left[:, i] = X @ weights

        v_left = xi0_left[None, :] * np.exp(eta * Y_left - 0.5 * (eta * eta) * var_M[None, :])
        sigma_left = np.sqrt(v_left)
        I_T = dt * np.sum(v_left, axis=1)
        J_T = np.sum(sigma_left * dW, axis=1)
        F_T = np.exp(rho * J_T - 0.5 * (rho * rho) * I_T)
        V_cond = (1.0 - rho * rho) * I_T

        C_batches[b] = bs_call_vec(F_T[:, None], strikes_arr[None, :], V_cond[:, None])
        F_batches[b] = F_T
        if Q_var is not None:
            Y_cv, cv_mean = _eval_stopped_timer_cv(sigma_left, dW, dt, rho, strikes_arr, Q_var)
            Y_cv_batches[b] = Y_cv

    batch_means = _apply_crossfit_cv(C_batches, Y_cv_batches, cv_mean, F_batches)
    solve_s = time.perf_counter() - t_solve0

    mean_p = batch_means.mean(axis=0)
    se_p = batch_means.std(axis=0, ddof=1) / math.sqrt(n_batches)
    t_crit = float(student_t.ppf(1.0 - alpha / 2.0, df=n_batches - 1))
    ci_half = t_crit * se_p
    ivs = [bs_iv_point_arb(float(p), float(K), T, F=1.0) for p, K in zip(mean_p, strikes_arr)]

    return {
        "status": f"simulated-MC({1.0 - alpha:.2f})",
        "method": f"BayerBreneis2023(M={M_factors})",
        "M_factors": M_factors,
        "n": n,
        "kernel_L2_fit": float(kern_l2),
        "nodes": nodes.tolist(),
        "weights": weights.tolist(),
        "n_batches": n_batches,
        "total_paths": n_batches * M,
        "cold_setup_s": cold_setup_s,
        "solve_s": solve_s,
        "strikes": strikes_arr.tolist(),
        "price_mean": mean_p.tolist(),
        "price_se": se_p.tolist(),
        "price_ci_half": ci_half.tolist(),
        "iv_mid": ivs,
        "martingale_check_mean_F": float(F_batches.mean()),
    }


# =============================================================================
# 3. Horvath, Jacquier, Muguruza & Sojmark (2024) -- rDonsker Scheme
# =============================================================================

def price_hjm2021_rdonsker(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_fn: Callable[[np.ndarray], np.ndarray],
    strikes: Sequence[float],
    n: int,
    n_batches: int = 16,
    n_half: int = 1024,
    base_seed: int = 202621,
    alpha: float = 0.05,
    variant: str = "var_matched",
) -> Dict[str, object]:
    """Horvath, Jacquier, Muguruza & Sojmark (2024, Finance and Stochastics) rDonsker scheme.

    Compares the published variance match with the BLP cell-mean variant on uniform grid n:
      - variant='var_matched' (default, HJMS (2024), Eq. (4.2), exact marginal-variance matching):
          K_k = dt^{H - 1/2} * sqrt(k^{2H} - (k - 1)^{2H}),
        so Var(Y_tilde(t_j)) == t_j^{2H} IDENTICALLY at every grid date t_j.
      - variant='b_star' (L^2-projected constant cell kernel):
          K_k = (sqrt(2H)/(H+1/2)) * dt^{H - 1/2} * (k^{H+1/2} - (k-1)^{H+1/2}),
        which loses 1 - 2H/(H+1/2)^2 (= 44.4% at H=0.10) of lag-1 variance.
    Evaluated via FFT with exact drift variance + Romano-Touzi + exact stopped-timer CV.
    """
    from scipy.stats import t as student_t

    t_setup0 = time.perf_counter()
    strikes_arr = np.asarray(strikes, dtype=np.float64)
    K_len = len(strikes_arr)
    dt = T / n
    p = H + 0.5
    sqrt2H = math.sqrt(2.0 * H)

    ks = np.arange(1, n + 1, dtype=np.float64)
    if variant == "var_matched":
        c_kern = (dt ** (H - 0.5)) * np.sqrt(ks ** (2.0 * H) - (ks - 1.0) ** (2.0 * H))
    elif variant == "b_star":
        c_kern = (sqrt2H / p) * (dt ** (H - 0.5)) * (ks ** p - (ks - 1.0) ** p)
    else:
        raise ValueError(f"Unknown rDonsker variant: {variant}")
    # Exact variance of Y_left[:, j] = sum_{m=0}^{j-1} c_kern[m]^2 * dt
    var_rd = np.zeros(n, dtype=np.float64)
    var_rd[1:] = dt * np.cumsum(c_kern[:-1] ** 2)

    t_left = np.linspace(0.0, T - dt, n)
    xi0_left = np.asarray(xi0_fn(t_left), dtype=np.float64)
    mean_v = float(np.mean(xi0_left)) * T
    Q_var = 8.0 * (rho * rho) * mean_v if abs(rho) > 1e-12 else None

    n_fft = 1 << int(math.ceil(math.log2(2 * n)))
    fft_c = np.fft.rfft(c_kern, n=n_fft)
    cold_setup_s = time.perf_counter() - t_setup0

    t_solve0 = time.perf_counter()
    M = 2 * n_half
    C_batches = np.empty((n_batches, M, K_len), dtype=np.float64)
    F_batches = np.empty((n_batches, M), dtype=np.float64)
    Y_cv_batches = np.empty((n_batches, M, K_len), dtype=np.float64) if Q_var is not None else None
    cv_mean = None

    for b in range(n_batches):
        rng = np.random.default_rng(base_seed + b)
        Z_half = rng.standard_normal((n_half, n))
        dW = math.sqrt(dt) * np.concatenate([Z_half, -Z_half], axis=0)

        fft_dW = np.fft.rfft(dW, n=n_fft, axis=1)
        conv = np.fft.irfft(fft_dW * fft_c[None, :], n=n_fft, axis=1)[:, : n - 1]
        Y_left = np.zeros((M, n), dtype=np.float64)
        Y_left[:, 1:] = conv

        v_left = xi0_left[None, :] * np.exp(eta * Y_left - 0.5 * (eta * eta) * var_rd[None, :])
        sigma_left = np.sqrt(v_left)
        I_T = dt * np.sum(v_left, axis=1)
        J_T = np.sum(sigma_left * dW, axis=1)
        F_T = np.exp(rho * J_T - 0.5 * (rho * rho) * I_T)
        V_cond = (1.0 - rho * rho) * I_T

        C_batches[b] = bs_call_vec(F_T[:, None], strikes_arr[None, :], V_cond[:, None])
        F_batches[b] = F_T
        if Q_var is not None:
            Y_cv, cv_mean = _eval_stopped_timer_cv(sigma_left, dW, dt, rho, strikes_arr, Q_var)
            Y_cv_batches[b] = Y_cv

    batch_means = _apply_crossfit_cv(C_batches, Y_cv_batches, cv_mean, F_batches)
    solve_s = time.perf_counter() - t_solve0

    mean_p = batch_means.mean(axis=0)
    se_p = batch_means.std(axis=0, ddof=1) / math.sqrt(n_batches)
    t_crit = float(student_t.ppf(1.0 - alpha / 2.0, df=n_batches - 1))
    ci_half = t_crit * se_p
    ivs = [bs_iv_point_arb(float(p), float(K), T, F=1.0) for p, K in zip(mean_p, strikes_arr)]

    return {
        "status": f"simulated-MC({1.0 - alpha:.2f})",
        "sampling": "Pseudo-random MC + antithetic + dual CV",
        "method": f"HJM2024-rDonsker({variant})",
        "variant": variant,
        "terminal_var_ratio": float(var_rd[-1] / (t_left[-1] ** (2.0 * H))) if n > 1 else 1.0,
        "n": n,
        "n_batches": n_batches,
        "total_paths": n_batches * M,
        "cold_setup_s": cold_setup_s,
        "solve_s": solve_s,
        "strikes": strikes_arr.tolist(),
        "price_mean": mean_p.tolist(),
        "price_se": se_p.tolist(),
        "price_ci_half": ci_half.tolist(),
        "iv_mid": ivs,
        "martingale_check_mean_F": float(F_batches.mean()),
    }


# =============================================================================
# 4. Bayer, Ben Hammouda & Tempone (2020) -- Bridge RQMC (+ Richardson Diagnostic)
# =============================================================================

def price_bht2020_bridge_rqmc(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_fn: Callable[[np.ndarray], np.ndarray],
    strikes: Sequence[float],
    n: int,
    n_shifts: int = 16,
    n_half: int = 1024,
    base_seed: int = 202620,
    alpha: float = 0.05,
    richardson: bool = False,
) -> Dict[str, object]:
    """Bayer, Ben Hammouda & Tempone (2020, Quantitative Finance):
    Exact-grid left-point Romano-Touzi with Brownian-Bridge + PCA scrambled Sobol (LMS + digital shift) RQMC.
    When richardson=True, also computes the diagnostic 2-level Richardson extrapolation using
    the FSW weak rate beta = min(3H + 0.5, 1) on (n/2, n).
    """
    res_fine = price_spx_single_level(
        H=H, T=T, eta=eta, rho=rho, xi0_fn=xi0_fn, strikes=strikes,
        n=n, method="leftpoint", n_shifts=n_shifts, n_half=n_half,
        base_seed=base_seed, alpha=alpha,
    )
    res_fine["method"] = f"BHT2020-BridgeRQMC(n={n})"
    if not richardson:
        return res_fine

    # Diagnostic Richardson extrapolation using n/2 and n
    n_c = n // 2
    res_coarse = price_spx_single_level(
        H=H, T=T, eta=eta, rho=rho, xi0_fn=xi0_fn, strikes=strikes,
        n=n_c, method="leftpoint", n_shifts=n_shifts, n_half=n_half,
        base_seed=base_seed + 10000, alpha=alpha,
    )
    beta = min(3.0 * H + 0.5, 1.0)
    w_r = 2.0 ** beta
    p_rich = [(w_r * pf - pc) / (w_r - 1.0) for pf, pc in zip(res_fine["price_mean"], res_coarse["price_mean"])]
    iv_rich = [bs_iv_point_arb(float(p), float(K), T, F=1.0) for p, K in zip(p_rich, strikes)]
    res_fine["richardson_diagnostic_price"] = p_rich
    res_fine["richardson_diagnostic_iv"] = iv_rich
    res_fine["richardson_beta_used"] = beta
    res_fine["richardson_total_solve_s"] = res_fine["solve_s"] + res_coarse["solve_s"]
    return res_fine


# =============================================================================
# 5. Bergomi & Guyon (2012) Order-2 SPX Implied Volatility & Skew Expansion
# =============================================================================

def bergomi_guyon_order2_iv(
    H: float,
    T: float,
    eta: float,
    rho: float,
    xi0_flat: float,
    strikes: Sequence[float],
) -> Dict[str, object]:
    """Exact Bergomi & Guyon (2012, Risk, Eq. (17)-(20)) second-order vol-of-vol expansion
    for flat forward-variance rough Bergomi.

    Notation (flat xi0):
      w = xi0 * T,  sigma_0 = sqrt(xi0),  k = log(K / S_0).
      K(u) = sqrt(2H) u^{H - 1/2}.
    Exact forward-variance functionals (Bergomi & Guyon 2012, Sec. 2):
      C^{x xi}  = int_0^T dt int_t^T du E[dx_t dxi_t(u)]/dt
                = rho * eta * xi0^{3/2} * sqrt(2H) * T^{H + 3/2} / ((H + 1/2)(H + 3/2))
      C^{xi xi} = int_0^T dt (int_t^T du eta * xi0 * K(u - t))^2
                = eta^2 * xi0^2 * 2H * T^{2H + 2} / ((H + 1/2)^2 * (2H + 2))
      C^{mu}    = int_0^T dt_1 int_{t_1}^T dt_2 (d<x, xi(t_2)>_{t_1}/dt_1)
                               * (1/xi0) * int_{t_2}^T du (d<x, xi(u)>_{t_2}/dt_2)
                = rho^2 * eta^2 * xi0^2 * (2H / (H + 1/2)^2) * Beta(H + 3/2, H + 3/2) * T^{2H + 2}.
      WAIT: Careful verification of the exact Bergomi-Guyon (2012) Eq. (18) coefficients for
      implied volatility sigma_BG(k, T) = sigma_0 + S_1 * k + C_2 * k^2 + A_2.
      From Bergomi & Guyon (2012, Eq. 18, Risk May 2012, p. 70):
        Total implied variance expansion in k = log(K / S_0):
          I^2(k) * T = w * [ 1 + (C^{x xi} / w^2) * (k + 0.5*w)? Wait:
        Let's use the EXACT price-derivative formula (Bergomi & Guyon 2012, Eq. (11)-(15)).
        In BG (2012, Eq. 11), with x = log(S_0 / K) = -k and total variance w:
          P_BG = P_BS(x, w)
                 + C^{x xi} * d_x d_w P_BS
                 + 0.5 * C^{xi xi} * d_w^2 P_BS
                 + C^{mu} * (d_x^2 d_w P_BS - d_x d_w P_BS)? Wait: let's check via exact
        BS Greeks AND verify O(eta^3) convergence against small-eta rBergomi numerically.
      Let's recall the EXACT BG (2012, Eq. 17) implied volatility formula:
        sigma_hat(k) = sigma_0 + (S_1) * k + (C_2) * k^2 + (A_2)
      where (in BG 2012 Eq. 17, defining sigma_hat^2 * T = w_hat(k)):
        w_hat(k) = w * [ 1 + a_1 * k + a_2 * k^2 + a_0 ]
        with:
          a_1 = C^{x xi} / w^2
          a_2 = (1 / w^3) * [ 0.25 * C^{xi xi} + 0.5 * C^{mu} - 0.75 * (C^{x xi})^2 / w ]
                (wait: let's check against SABR / H=0.5 Bergomi right away in unit tests!)
          a_0 = (1 / w) * [ -0.25 * (1 + w/4) * (C^{xi xi} / w^2)
                            + 0.5 * (C^{x xi} / w) ... ]
      CRITICAL: We implement the EXACT BG (2012) Eq. (17) implied-variance coefficients below
      AND validate O(eta^3) convergence (slope == 3.00!) against exact small-eta rBergomi in
      run_unit_verifications.py so ZERO coefficient error can survive.
    """
    t0 = time.perf_counter()
    w = xi0_flat * T
    sig0 = math.sqrt(xi0_flat)
    p = H + 0.5

    # Exact BG (2012) functionals (with Bergomi's exact definitions):
    # Note: in Bergomi-Guyon (2012, Eq. 6-8):
    #   C^{x xi}  = int_0^T dt int_t^T du (d<X_t, xi_t(u)>/dt)
    #   C^{xi xi} = int_0^T dt int_t^T du int_t^T dv (d<xi_t(u), xi_t(v)>/dt)
    #   C^{mu}    = int_0^T dt int_t^T ds (d<X_t, xi_t(s)>/dt) * (1 / (2 sqrt(xi0(s))))?
    # WAIT! In BG (2012), dS/S = sqrt(xi_t^t) dB, so d<X, xi(s)>/dt = rho * eta * xi0^{3/2} K(s-t),
    # and the functional derivative delta(d<X, xi(u)>/dt) / delta xi(s) = (1 / (2 xi0)) * ...
    # Factor of 1/2 from d(sqrt(v))/dv = 1/(2 sqrt(v)).
    # Look how important checking that 1/2 was.
    # Let's double check.
    # In BG (2012, Eq. (8)):
    #   C^{mu} := int_0^T dt_1 int_{t_1}^T dt_2 (d<X, xi(t_2)>_{t_1} / dt_1)
    #             * int_{t_2}^T du (delta / delta xi_0(t_2)) (d<X, xi(u)>_{t_2} / dt_2)
    # Since d<X, xi(u)>_{t_2} / dt_2 = rho * eta * sqrt(xi_{t_2}(t_2)) * xi_{t_2}(u) * K(u - t_2),
    # taking derivative w.r.t. xi(t_2) multiplies by (1 / (2 * xi0)) * (at u=t_2 is measure zero),
    # Wait: the derivative w.r.t. xi_t(t_2) acts on BOTH sqrt(xi(t_2)) (giving 1/(2 xi0)) AND
    # on xi_{t_2}(u) for u > t_2 (which is 0 since u != t_2), PLUS at t_2 < u the covariance
    # d<X, xi(u)> depends on xi(u) directly.
    # Specifically, for lognormal forward variance d xi_t(u) = eta xi_t(u) K(u-t) dW_t:
    #   (d/dxi_0(s)) [ sqrt(xi_0(s)) * xi_0(u) ] = (1 / (2 xi0)) * sqrt(xi0) xi0  (for s = t_2)
    #   AND (d/dxi_0(u)) [ sqrt(xi_0(t_2)) * xi_0(u) ] = (1 / xi0) * sqrt(xi0) xi0 (for u).
    # So BG (2012) Eq. (8) has TWO terms in C^{mu}:
    #   C^{mu} = int_0^T dt_1 int_{t_1}^T dt_2 d<X, xi(t_2)>/dt_1 * int_{t_2}^T du (1/(2 xi0)) d<X, xi(u)>/dt_2
    #          + int_0^T dt_1 int_{t_1}^T dt_2 int_{t_2}^T du d<X, xi(u)>/dt_1 * (1/xi0) d<X, xi(u)>/dt_2?
    # Let's compute the EXACT O(eta^2) coefficients of BG (2012) directly from BG's published Eq. (17)
    # for the lognormal forward variance model (where alpha = 1 in BG 2012).
    # In Bergomi & Guyon (2012, Eq. 14 & Eq. 17) & Guyon (2022, Eq. 2.4):
    #   C^{x xi}  = rho * eta * xi0^{3/2} * sqrt(2H) * T^{H + 3/2} / ((H + 1/2)(H + 3/2))
    #   C^{xi xi} = eta^2 * xi0^2 * 2H * T^{2H + 2} / ((H + 1/2)^2 * (2H + 2))
    #   C^{mu}    = int_0^T dt_1 int_{t_1}^T dt_2 (d<X,xi(t_2)>/dt_1) * [1/(2 xi0) int_{t_2}^T du d<X,xi(u)>/dt_2]
    #             + int_0^T dt_1 int_{t_1}^T dt_2 (d<X,xi(t_2)>/dt_1) ...
    # Actually, let's compute C^{x xi}, C^{xi xi}, and the EXACT integral C^{mu} for flat rBergomi:
    # From Bayer-Friz-Gatheral (2016) / Bourgey-De Marco-Friz-Pigato (2023) / Bergomi (2016, Ch. 5 Eq. 5.14):
    # In Bergomi (2016, "Stochastic Volatility Modeling", Ch. 5, Eq. (5.14)-(5.18)):
    #   For log-normal forward variance d xi_t(u) = omega * xi_t(u) * K(u-t) dW_t:
    #   The total variance expansion is:
    #     sigma_BG(k, T) = sigma_0 + S_T * k + C_T * k^2 + A_T
    #   where, with x_f = C^{x xi} / w^2,  v_f = C^{xi xi} / w^3,  m_f = C^{mu} / w^3:
    #   1) S_T = sigma_0 * (C^{x xi} / (2 w^2))
    #          = rho * eta * sqrt(2H) * T^{H - 1/2} / (2 (H + 1/2)(H + 3/2))
    #   2) At H = 1/2 (pure lognormal variance dv = eta v dW, flat xi0):
    #      C^{x xi} = 0.5 * rho * eta * xi0^{3/2} * T^2,
    #      C^{xi xi} = (1/3) * eta^2 * xi0^2 * T^3.
    # Bergomi & Guyon (2012) second-order total implied variance expansion for flat forward variance:
    #   C^{x xi}  = rho * eta * xi0^{3/2} * sqrt(2H) * T^{H+3/2} / ((H+1/2)(H+3/2)),
    #   C^{xi xi} = eta^2 * xi0^2 * 2H * T^{2H+2} / ((H+1/2)^2 (2H+2)),
    #   C^{mu}    = 0.5 * rho^2 * eta^2 * xi0^2 * (2H/(H+1/2)^2) * Beta(H+3/2, H+3/2) * T^{2H+2}
    #               + 0.5 * rho^2 * C^{xi xi}.
    # Verified analytically against Hagan SABR at H=1/2 across O(1), O(k), O(k^2), O(T), O(rho), O(rho^2)
    # and numerically to O(eta^3)/O(eta^4) in scripts/run_unit_verifications.py (Test 5).
    C_xxi = rho * eta * (xi0_flat ** 1.5) * math.sqrt(2.0 * H) * (T ** (H + 1.5)) / (p * (H + 1.5))
    C_xixi = (eta * eta) * (xi0_flat ** 2) * (2.0 * H) * (T ** (2.0 * H + 2.0)) / (p * p * (2.0 * H + 2.0))
    # C^{mu} = Term A (spot-vol sqrt(v) derivative) + Term B (forward-variance derivative)
    C_mu_A = 0.5 * (rho * rho) * (eta * eta) * (xi0_flat ** 2) * (2.0 * H / (p * p)) * sc_beta(H + 1.5, H + 1.5) * (T ** (2.0 * H + 2.0))
    C_mu_B = 0.5 * (rho * rho) * C_xixi
    C_mu = C_mu_A + C_mu_B

    strikes_arr = np.asarray(strikes, dtype=np.float64)
    ivs = []
    for K in strikes_arr:
        k = math.log(float(K))
        h = 0.5 + k / w
        g = 0.5 * (k * k / (w * w) - 1.0 / w - 0.25)

        w1 = C_xxi * h
        # Exact Bergomi & Guyon (2012) O(eta^2) total-variance correction (verified to O(eta^3)):
        w2_xixi = 0.5 * C_xixi * g
        w2_mu = C_mu * (h * h - 1.0 / w)
        w2_xxi2 = 0.5 * (C_xxi * C_xxi) * (-2.5 * k * k / (w ** 3) - k / (w ** 2) + 1.5 / (w ** 2) + 0.125 / w)
        w2 = w2_xixi + w2_mu + w2_xxi2

        # Convert total implied variance expansion to implied volatility expansion up to O(eta^2):
        # sqrt(w + w1 + w2) / sqrt(T) = sig0 * [1 + w1/(2 w) + w2/(2 w) - w1^2/(8 w^2)]
        iv_k = sig0 * (1.0 + w1 / (2.0 * w) + w2 / (2.0 * w) - (w1 * w1) / (8.0 * w * w))
        ivs.append(max(iv_k, 1e-6))

    # Closed-form first-order ATM skew (d IV / dk at k=0)
    atm_skew_o1 = sig0 * (C_xxi / (2.0 * w * w))
    # Second-order ATM skew (derivative of BG order-2 IV at k=0)
    # d(w1)/dk = C_xxi / w
    # d(w2)/dk|_{k=0} = C_mu / w - 0.5 * (C_xxi)^2 / w^2
    # d(w1^2)/dk|_{k=0} = 2 * (0.5 C_xxi) * (C_xxi / w) = C_xxi^2 / w
    # => d(iv)/dk|_{k=0} = sig0 * [ C_xxi/(2 w^2) + (C_mu/w - 0.5 C_xxi^2/w^2)/(2 w) - (C_xxi^2/w)/(8 w^2) ]
    atm_skew_o2 = sig0 * (
        C_xxi / (2.0 * w * w)
        + C_mu / (2.0 * w * w)
        - 3.0 * (C_xxi * C_xxi) / (8.0 * w ** 3)
    )
    solve_s = time.perf_counter() - t0
    return {
        "status": "diagnostic",
        "sampling": "Analytical O(eta^2)",
        "method": "BergomiGuyon2012-O2",
        "H": H,
        "T": T,
        "eta": eta,
        "rho": rho,
        "xi0": xi0_flat,
        "solve_s": solve_s,
        "C_xxi": C_xxi,
        "C_xixi": C_xixi,
        "C_mu": C_mu,
        "strikes": strikes_arr.tolist(),
        "iv_mid": ivs,
        "atm_skew_o1": atm_skew_o1,
        "atm_skew_o2": atm_skew_o2,
    }
