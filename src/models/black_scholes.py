"""Black-Scholes analytical pricing model for European options.

Implements pricing, Greeks, and implied-volatility inversion.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.models import GreeksResult, PricingResult
from src.utils.config import BS_IV_MAX_ITER, BS_IV_SIGMA_BOUNDS, BS_IV_TOL
from src.utils.math_utils import clamp, safe_exp, safe_log, safe_sqrt


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
) -> PricingResult:
    """Return the Black-Scholes price for a European option.

    Parameters
    ----------
    S : float
        Current underlying spot price.
    K : float
        Strike price.
    T : float
        Time to expiration **in years**.
    r : float
        Continuously-compounded risk-free interest rate.
    sigma : float
        Annualised volatility of the underlying.
    q : float
        Continuous dividend yield.
    option_type : str
        ``"call"`` or ``"put"`` (lowercase).

    Returns
    -------
    PricingResult
        Analytical BS price wrapped in a ``PricingResult`` dataclass.
    """
    option_type = option_type.lower()

    # --- Edge case: expiry already passed or at expiry -----------------------
    if T <= 0:
        if option_type == "call":
            intrinsic = max(S - K, 0.0)  # max(S - K, 0)
        else:
            intrinsic = max(K - S, 0.0)  # max(K - S, 0)
        return PricingResult(price=intrinsic, model_name="Black-Scholes")

    # --- Edge case: zero volatility (deterministic) --------------------------
    if sigma <= 0:
        # Forward value of spot discounted by dividend yield
        forward = S * safe_exp(-q * T)  # F = S * exp(-qT)
        df = safe_exp(-r * T)           # discount factor = exp(-rT)
        if option_type == "call":
            p = max(forward - K, 0.0) * df  # max(F - K, 0) * exp(-rT)
        else:
            p = max(K - forward, 0.0) * df  # max(K - F, 0) * exp(-rT)
        return PricingResult(price=p, model_name="Black-Scholes")

    # --- Standard Black-Scholes formula --------------------------------------
    sqrt_T = safe_sqrt(T)                                    # sqrt(T)

    # d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d1 = (safe_log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)

    # d2 = d1 - sigma * sqrt(T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        # C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
        p = (
            S * safe_exp(-q * T) * norm.cdf(d1)
            - K * safe_exp(-r * T) * norm.cdf(d2)
        )
    else:
        # P = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)
        p = (
            K * safe_exp(-r * T) * norm.cdf(-d2)
            - S * safe_exp(-q * T) * norm.cdf(-d1)
        )

    return PricingResult(price=float(p), model_name="Black-Scholes")


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
) -> GreeksResult:
    """Return analytical Black-Scholes Greeks.

    Notes
    -----
    * **Theta** is returned as the *per-day* rate (annual theta / 365).
    * **Vega** is returned as the price change for a *1 percentage-point*
      increase in volatility (annual vega * 0.01).

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type
        Same as :func:`price`.

    Returns
    -------
    GreeksResult
    """
    option_type = option_type.lower()

    # --- Edge case: at / past expiry -----------------------------------------
    if T <= 0:
        if option_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return GreeksResult(
            delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            model_name="Black-Scholes",
        )

    # --- Edge case: zero volatility ------------------------------------------
    if sigma <= 0:
        forward = S * safe_exp(-q * T)  # F = S * exp(-qT)
        if option_type == "call":
            delta = safe_exp(-q * T) if forward > K else 0.0
        else:
            delta = -safe_exp(-q * T) if forward < K else 0.0
        return GreeksResult(
            delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            model_name="Black-Scholes",
        )

    # --- Intermediate quantities ---------------------------------------------
    sqrt_T = safe_sqrt(T)  # sqrt(T)

    # d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d1 = (safe_log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)

    # d2 = d1 - sigma * sqrt(T)
    d2 = d1 - sigma * sqrt_T

    # Common terms
    exp_neg_qT = safe_exp(-q * T)   # exp(-qT)
    exp_neg_rT = safe_exp(-r * T)   # exp(-rT)
    n_d1 = norm.pdf(d1)             # n(d1) — standard normal PDF

    # --- Delta ---------------------------------------------------------------
    if option_type == "call":
        # Delta_call = exp(-qT) * N(d1)
        delta = exp_neg_qT * norm.cdf(d1)
    else:
        # Delta_put = -exp(-qT) * N(-d1)
        delta = -exp_neg_qT * norm.cdf(-d1)

    # --- Gamma (same for call and put) ---------------------------------------
    # Gamma = exp(-qT) * n(d1) / (S * sigma * sqrt(T))
    gamma = exp_neg_qT * n_d1 / (S * sigma * sqrt_T)

    # --- Theta ---------------------------------------------------------------
    # Common term: -(S * sigma * exp(-qT) * n(d1)) / (2 * sqrt(T))
    theta_common = -(S * sigma * exp_neg_qT * n_d1) / (2.0 * sqrt_T)

    if option_type == "call":
        # Theta_call = common + q*S*exp(-qT)*N(d1) - r*K*exp(-rT)*N(d2)
        theta_annual = (
            theta_common
            + q * S * exp_neg_qT * norm.cdf(d1)
            - r * K * exp_neg_rT * norm.cdf(d2)
        )
    else:
        # Theta_put = common - q*S*exp(-qT)*N(-d1) + r*K*exp(-rT)*N(-d2)
        theta_annual = (
            theta_common
            - q * S * exp_neg_qT * norm.cdf(-d1)
            + r * K * exp_neg_rT * norm.cdf(-d2)
        )

    theta = theta_annual / 365.0  # per-day theta

    # --- Vega (same for call and put) ----------------------------------------
    # Vega = S * exp(-qT) * n(d1) * sqrt(T)
    vega_annual = S * exp_neg_qT * n_d1 * sqrt_T
    vega = vega_annual * 0.01  # price change per 1% vol move

    # --- Rho -----------------------------------------------------------------
    if option_type == "call":
        # Rho_call = K * T * exp(-rT) * N(d2)
        rho = K * T * exp_neg_rT * norm.cdf(d2)
    else:
        # Rho_put = -K * T * exp(-rT) * N(-d2)
        rho = -K * T * exp_neg_rT * norm.cdf(-d2)

    return GreeksResult(
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
        rho=float(rho),
        model_name="Black-Scholes",
    )


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    tol: float = BS_IV_TOL,
    max_iter: int = BS_IV_MAX_ITER,
) -> float:
    """Recover the implied volatility from an observed market price.

    Strategy
    --------
    1. Newton-Raphson iterations using vega as the derivative.
    2. If Newton fails to converge within *max_iter* steps, fall back to a
       bisection search over ``BS_IV_SIGMA_BOUNDS``.

    Parameters
    ----------
    market_price : float
        Observed option premium.
    S, K, T, r, q, option_type
        Same as :func:`price`.
    tol : float
        Convergence tolerance (default from config).
    max_iter : int
        Maximum iterations (default from config).

    Returns
    -------
    float
        Implied volatility, clamped to ``BS_IV_SIGMA_BOUNDS``.
    """
    sigma_lo, sigma_hi = BS_IV_SIGMA_BOUNDS

    # --- Newton-Raphson ------------------------------------------------------
    sigma = 0.3  # reasonable starting guess
    for _ in range(max_iter):
        bs_price = price(S, K, T, r, sigma, q, option_type).price
        diff = bs_price - market_price

        if abs(diff) < tol:
            return clamp(sigma, sigma_lo, sigma_hi)

        # Vega (annual, un-scaled) for Newton step
        sqrt_T = safe_sqrt(T)
        d1 = (safe_log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        vega_val = S * safe_exp(-q * T) * norm.pdf(d1) * sqrt_T  # annual vega

        if abs(vega_val) < 1e-12:
            break  # vega too small — Newton will diverge

        sigma -= diff / vega_val  # sigma_{n+1} = sigma_n - f / f'

        # Keep sigma within bounds during iteration
        sigma = clamp(sigma, sigma_lo, sigma_hi)

    # --- Bisection fallback --------------------------------------------------
    lo, hi = sigma_lo, sigma_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        bs_price = price(S, K, T, r, mid, q, option_type).price
        diff = bs_price - market_price

        if abs(diff) < tol or (hi - lo) < tol:
            return clamp(mid, sigma_lo, sigma_hi)

        if diff > 0:
            hi = mid
        else:
            lo = mid

    return clamp(0.5 * (lo + hi), sigma_lo, sigma_hi)
