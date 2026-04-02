"""Heston stochastic volatility model.

Provides calibration from historical returns, Monte-Carlo path simulation
via Euler-Maruyama discretization, and synthetic training-data generation
for the neural-network ensemble.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.models.black_scholes import price as bs_price
from src.utils.config import HESTON_BOUNDS, HESTON_DEFAULTS
from src.utils.math_utils import safe_exp, safe_sqrt


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_heston(returns: np.ndarray) -> dict:
    """Estimate Heston parameters from historical log returns.

    Uses a simplified moment-matching approach:

    * **v0** -- sample variance of returns (annualised).
    * **theta** -- long-run average of rolling variance.
    * **kappa** -- estimated from the first-order autocorrelation of the
      rolling variance series.
    * **xi** -- standard deviation of variance changes.
    * **rho** -- correlation between returns and variance changes.

    The parameters are then refined via ``scipy.optimize.minimize``
    (L-BFGS-B) on a negative log-likelihood objective constructed from
    the empirical moments.

    If calibration fails for any reason (not enough data, optimizer does
    not converge, etc.) the function falls back to
    :data:`~src.utils.config.HESTON_DEFAULTS`.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of historical **log** returns (daily).

    Returns
    -------
    dict
        Keys ``"v0"``, ``"kappa"``, ``"theta"``, ``"xi"``, ``"rho"``.
    """
    try:
        returns = np.asarray(returns, dtype=np.float64).ravel()

        # Need a reasonable amount of data
        if len(returns) < 60:
            return dict(HESTON_DEFAULTS)

        # --- Moment-matching initial estimates --------------------------------
        ann_factor = 252  # trading days per year

        # v0: annualised sample variance
        v0_init = float(np.var(returns) * ann_factor)

        # Rolling variance (20-day window, annualised)
        window = 20
        n = len(returns)
        rolling_var = np.array([
            np.var(returns[i : i + window]) * ann_factor
            for i in range(n - window + 1)
        ])

        # theta: long-run average of rolling variance
        theta_init = float(np.mean(rolling_var))

        # kappa: estimated from lag-1 autocorrelation of rolling variance
        if len(rolling_var) > 1:
            rv_mean = np.mean(rolling_var)
            rv_centred = rolling_var - rv_mean
            denom = np.sum(rv_centred ** 2)
            if denom > 1e-12:
                acf1 = np.sum(rv_centred[:-1] * rv_centred[1:]) / denom
                acf1 = np.clip(acf1, 0.01, 0.999)
                # Mean-reversion speed: kappa ~= -log(acf1) * ann_factor / window
                kappa_init = float(-np.log(acf1) * ann_factor / window)
            else:
                kappa_init = HESTON_DEFAULTS["kappa"]
        else:
            kappa_init = HESTON_DEFAULTS["kappa"]

        # xi: standard deviation of variance changes (annualised)
        if len(rolling_var) > 1:
            var_changes = np.diff(rolling_var)
            xi_init = float(np.std(var_changes) * np.sqrt(ann_factor / window))
        else:
            xi_init = HESTON_DEFAULTS["xi"]

        # rho: correlation between returns and variance changes
        if len(rolling_var) > 1:
            aligned_returns = returns[window - 1 : window - 1 + len(rolling_var) - 1]
            if len(aligned_returns) == len(rolling_var) - 1:
                corr_matrix = np.corrcoef(aligned_returns, np.diff(rolling_var))
                rho_init = float(corr_matrix[0, 1])
                if np.isnan(rho_init):
                    rho_init = HESTON_DEFAULTS["rho"]
            else:
                rho_init = HESTON_DEFAULTS["rho"]
        else:
            rho_init = HESTON_DEFAULTS["rho"]

        # Clamp initial estimates to bounds
        def _clamp(val: float, key: str) -> float:
            lo, hi = HESTON_BOUNDS[key]
            return float(np.clip(val, lo, hi))

        v0_init = _clamp(v0_init, "v0")
        kappa_init = _clamp(kappa_init, "kappa")
        theta_init = _clamp(theta_init, "theta")
        xi_init = _clamp(xi_init, "xi")
        rho_init = _clamp(rho_init, "rho")

        # --- L-BFGS-B optimisation on negative log-likelihood -----------------
        # Simplified Gaussian log-likelihood based on the first two moments of
        # the Heston model variance dynamics.

        def _neg_log_likelihood(x: np.ndarray) -> float:
            v0, kappa, theta, xi, rho = x
            # Predicted conditional variance at each step
            dt = 1.0 / ann_factor
            predicted_vars = np.empty(len(rolling_var))
            predicted_vars[0] = v0
            for i in range(1, len(rolling_var)):
                predicted_vars[i] = (
                    predicted_vars[i - 1]
                    + kappa * (theta - predicted_vars[i - 1]) * (window * dt)
                )
                predicted_vars[i] = max(predicted_vars[i], 1e-8)

            # Gaussian NLL
            residuals = rolling_var - predicted_vars
            variance_of_residuals = max(np.var(residuals), 1e-8)
            nll = 0.5 * len(residuals) * np.log(variance_of_residuals)
            nll += 0.5 * np.sum(residuals ** 2) / variance_of_residuals

            # Penalty for rho mismatch
            if len(rolling_var) > 1:
                aligned_ret = returns[window - 1 : window - 1 + len(rolling_var) - 1]
                if len(aligned_ret) == len(rolling_var) - 1:
                    emp_corr = np.corrcoef(aligned_ret, np.diff(rolling_var))
                    if not np.isnan(emp_corr[0, 1]):
                        nll += 10.0 * (rho - emp_corr[0, 1]) ** 2

            return float(nll)

        x0 = np.array([v0_init, kappa_init, theta_init, xi_init, rho_init])
        bounds = [HESTON_BOUNDS[k] for k in ("v0", "kappa", "theta", "xi", "rho")]

        result = minimize(
            _neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-8},
        )

        if not result.success:
            return dict(HESTON_DEFAULTS)

        v0, kappa, theta, xi, rho = result.x

        # Clamp to bounds (belt-and-suspenders)
        v0 = _clamp(v0, "v0")
        kappa = _clamp(kappa, "kappa")
        theta = _clamp(theta, "theta")
        xi = _clamp(xi, "xi")
        rho = _clamp(rho, "rho")

        # --- Feller condition: 2*kappa*theta > xi^2 --------------------------
        if 2 * kappa * theta <= xi ** 2:
            xi = float(safe_sqrt(2 * kappa * theta * 0.99))
            xi = _clamp(xi, "xi")

        return {"v0": v0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho}

    except Exception:
        return dict(HESTON_DEFAULTS)


# ---------------------------------------------------------------------------
# Monte-Carlo simulation
# ---------------------------------------------------------------------------

def simulate_paths(
    S0: float,
    r: float,
    q: float,
    params: dict,
    T: float,
    n_steps: int = 100,
    n_paths: int = 1000,
) -> np.ndarray:
    """Simulate price paths under the Heston stochastic volatility model.

    Uses the Euler-Maruyama discretisation with full truncation (variance
    is floored at zero at each time step).

    The two Brownian motions are correlated via Cholesky decomposition:

    .. math::

        dW_2 = \\rho \\, dW_1 + \\sqrt{1 - \\rho^2} \\, dZ

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    params : dict
        Heston parameters with keys ``"v0"``, ``"kappa"``, ``"theta"``,
        ``"xi"``, ``"rho"``.
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps (default 100).
    n_paths : int
        Number of Monte-Carlo paths (default 1000).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_steps + 1)`` containing simulated
        stock price paths.
    """
    v0 = params["v0"]
    kappa = params["kappa"]
    theta = params["theta"]
    xi = params["xi"]
    rho = params["rho"]

    dt = T / n_steps

    # Pre-allocate arrays
    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))

    S[:, 0] = S0
    v[:, 0] = v0

    # Random draws (independent standard normals)
    Z1 = np.random.standard_normal((n_paths, n_steps))
    Z2 = np.random.standard_normal((n_paths, n_steps))

    # Cholesky: correlated Brownian increments
    # dW1 = Z1 * sqrt(dt)
    # dW2 = (rho * Z1 + sqrt(1 - rho^2) * Z2) * sqrt(dt)
    sqrt_dt = np.sqrt(dt)
    rho_comp = np.sqrt(max(1.0 - rho ** 2, 0.0))

    for t in range(n_steps):
        v_pos = np.maximum(v[:, t], 0.0)  # full truncation
        sqrt_v = np.sqrt(v_pos)

        dW1 = Z1[:, t] * sqrt_dt
        dW2 = (rho * Z1[:, t] + rho_comp * Z2[:, t]) * sqrt_dt

        # Price SDE: dS = (r - q) * S * dt + sqrt(v) * S * dW1
        S[:, t + 1] = S[:, t] * np.exp(
            (r - q - 0.5 * v_pos) * dt + sqrt_v * dW1
        )

        # Variance SDE: dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2
        v[:, t + 1] = v_pos + kappa * (theta - v_pos) * dt + xi * sqrt_v * dW2

        # Full truncation on variance
        v[:, t + 1] = np.maximum(v[:, t + 1], 0.0)

    return S


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_training_data(
    S0: float,
    K_range: tuple[float, float],
    T_range: tuple[float, float],
    r: float,
    q: float,
    params: dict,
    n_samples: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (features, labels) pairs for neural-network training.

    For each sample a random strike *K* and maturity *T* are drawn.  A short
    Heston path is simulated to obtain a realistic spot price, and the
    Black-Scholes analytical price is computed as the supervised target.

    Feature vector (7 elements):

    0. ``S / K`` -- moneyness
    1. ``T`` -- time to expiration
    2. ``sqrt(v0)`` -- historical volatility proxy
    3. ``r`` -- risk-free rate
    4. ``q`` -- dividend yield
    5. ``RSI_14`` -- random realistic RSI in [20, 80]
    6. ``vol_ratio`` -- random realistic vol ratio in [0.5, 2.0]

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K_range : tuple[float, float]
        (min_strike, max_strike) for uniform sampling.
    T_range : tuple[float, float]
        (min_maturity, max_maturity) in years for uniform sampling.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    params : dict
        Heston parameters dict.
    n_samples : int
        Number of training samples to generate (default 50 000).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, y)`` where ``X`` has shape ``(n_samples, 1, 7)`` (single-step
        LSTM input) and ``y`` has shape ``(n_samples, 1)``.
    """
    sigma_hist = safe_sqrt(params["v0"])

    # Random strikes and maturities
    K_vals = np.random.uniform(K_range[0], K_range[1], size=n_samples)
    T_vals = np.random.uniform(T_range[0], T_range[1], size=n_samples)

    # Random realistic RSI and vol_ratio values
    rsi_vals = np.random.uniform(20.0, 80.0, size=n_samples)
    vol_ratio_vals = np.random.uniform(0.5, 2.0, size=n_samples)

    # Simulate a short batch of Heston paths to get realistic spot prices
    # Use a small number of steps for speed
    paths = simulate_paths(S0, r, q, params, T=0.1, n_steps=5, n_paths=n_samples)
    S_sim = paths[:, -1]  # terminal prices from short simulation

    # Build features and labels
    X = np.empty((n_samples, 1, 7), dtype=np.float64)
    y = np.empty((n_samples, 1), dtype=np.float64)

    for i in range(n_samples):
        S_i = float(S_sim[i])
        K_i = float(K_vals[i])
        T_i = float(T_vals[i])

        moneyness = S_i / K_i if K_i > 0 else 1.0

        X[i, 0, :] = [moneyness, T_i, sigma_hist, r, q, rsi_vals[i], vol_ratio_vals[i]]

        # Label: Black-Scholes price as supervised target
        bs_result = bs_price(
            S=S_i, K=K_i, T=T_i, r=r, sigma=sigma_hist, q=q, option_type="call",
        )
        y[i, 0] = bs_result.price

    return X, y
