"""Bayesian Binomial Tree pricing model for European options.

Uses the Cox-Ross-Rubinstein (CRR) binomial tree with a Bayesian prior
on volatility.  A LogNormal prior is placed on sigma, samples are drawn,
and each sample drives one CRR tree evaluation.  The posterior distribution
of option prices yields a mean estimate and credible interval.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import lognorm

from src.models import GreeksResult, PricingResult
from src.utils.config import BAYESIAN_DEFAULT_SAMPLES, BAYESIAN_DEFAULT_STEPS
from src.utils.math_utils import safe_exp


# ---------------------------------------------------------------------------
# Internal: single CRR tree evaluation
# ---------------------------------------------------------------------------

def _crr_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
    steps: int,
) -> float:
    """Evaluate a single CRR binomial tree for a European option.

    Parameters
    ----------
    S : float
        Current underlying spot price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate (continuously compounded).
    sigma : float
        Annualised volatility.
    q : float
        Continuous dividend yield.
    option_type : str
        ``"call"`` or ``"put"``.
    steps : int
        Number of time steps in the tree.

    Returns
    -------
    float
        Option price from the CRR tree.
    """
    option_type = option_type.lower()

    # --- Edge case: at or past expiry ----------------------------------------
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    # --- Edge case: zero or negative volatility (deterministic) ---------------
    if sigma <= 0:
        forward = S * safe_exp(-q * T)
        df = safe_exp(-r * T)
        if option_type == "call":
            return max(forward - K, 0.0) * df
        else:
            return max(K - forward, 0.0) * df

    # --- CRR parameters ------------------------------------------------------
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # --- Terminal payoffs (vectorised) ----------------------------------------
    # At step N, the spot at node j (j = 0 .. steps) is S * u^j * d^(steps-j)
    j = np.arange(steps + 1)
    spot_terminal = S * (u ** j) * (d ** (steps - j))

    if option_type == "call":
        payoffs = np.maximum(spot_terminal - K, 0.0)
    else:
        payoffs = np.maximum(K - spot_terminal, 0.0)

    # --- Backward induction (vectorised, no per-node loop) --------------------
    disc = np.exp(-r * dt)
    for _ in range(steps):
        payoffs = disc * (p * payoffs[1:] + (1.0 - p) * payoffs[:-1])

    return float(payoffs[0])


# ---------------------------------------------------------------------------
# Public: Bayesian pricing
# ---------------------------------------------------------------------------

def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
    steps: int = BAYESIAN_DEFAULT_STEPS,
    n_samples: int = BAYESIAN_DEFAULT_SAMPLES,
    vol_history: list[float] | None = None,
) -> PricingResult:
    """Price a European option using a Bayesian Binomial Tree.

    A LogNormal prior is placed on volatility.  ``n_samples`` draws are
    taken and each drives one CRR tree evaluation.  The result is the
    posterior mean price with a 95% credible interval.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type
        Standard option parameters (see :func:`_crr_tree`).
    steps : int
        Number of CRR tree steps per evaluation.
    n_samples : int
        Number of volatility samples to draw.
    vol_history : list[float] | None
        If provided, a list of historical volatility estimates (e.g. at
        different look-back windows).  A LogNormal distribution is fit to
        these values and used as the prior.  If *None*, a default LogNormal
        prior centred on *sigma* with shape 0.2 is used.

    Returns
    -------
    PricingResult
    """
    # --- Build the prior on sigma --------------------------------------------
    if vol_history is not None and len(vol_history) >= 2:
        vol_arr = np.array(vol_history, dtype=float)
        # scipy lognorm.fit returns (shape, loc, scale); we fix loc=0
        shape, loc, scale = lognorm.fit(vol_arr, floc=0)
        sigma_samples = lognorm.rvs(shape, loc=loc, scale=scale, size=n_samples)
    else:
        # Default prior centred on sigma with shape=0.2
        shape = 0.2
        sigma_samples = lognorm.rvs(shape, loc=0, scale=sigma, size=n_samples)

    # Clamp sigma samples to a sensible range to avoid degenerate trees
    sigma_samples = np.clip(sigma_samples, 1e-6, 5.0)

    # --- Run CRR tree for each sampled sigma ---------------------------------
    prices = np.array(
        [_crr_tree(S, K, T, r, s, q, option_type, steps) for s in sigma_samples]
    )

    # --- Aggregate results ----------------------------------------------------
    mean_price = float(np.mean(prices))
    ci_lo = float(np.percentile(prices, 2.5))
    ci_hi = float(np.percentile(prices, 97.5))

    return PricingResult(
        price=mean_price,
        model_name="Bayesian Tree",
        confidence_interval=(ci_lo, ci_hi),
        posterior_samples=prices,
        metadata={"steps": steps, "n_samples": n_samples},
    )


# ---------------------------------------------------------------------------
# Public: Greeks via finite differences
# ---------------------------------------------------------------------------

def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
    steps: int = BAYESIAN_DEFAULT_STEPS,
) -> GreeksResult:
    """Compute finite-difference Greeks from the CRR tree.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type
        Standard option parameters.
    steps : int
        Number of CRR tree steps.

    Returns
    -------
    GreeksResult
    """
    p0 = _crr_tree(S, K, T, r, sigma, q, option_type, steps)

    # --- Delta ---------------------------------------------------------------
    h_s = 0.01 * S
    p_up = _crr_tree(S + h_s, K, T, r, sigma, q, option_type, steps)
    p_dn = _crr_tree(S - h_s, K, T, r, sigma, q, option_type, steps)
    delta = (p_up - p_dn) / (2.0 * h_s)

    # --- Gamma ---------------------------------------------------------------
    gamma = (p_up - 2.0 * p0 + p_dn) / (h_s ** 2)

    # --- Theta ---------------------------------------------------------------
    h_t = 1.0 / 365.0
    T_shifted = max(T - h_t, 0.0)
    p_t = _crr_tree(S, K, T_shifted, r, sigma, q, option_type, steps)
    theta = (p_t - p0) / h_t

    # --- Vega ----------------------------------------------------------------
    h_v = 0.01
    p_v_up = _crr_tree(S, K, T, r, sigma + h_v, q, option_type, steps)
    p_v_dn = _crr_tree(S, K, T, r, max(sigma - h_v, 1e-8), q, option_type, steps)
    # Effective h may differ if sigma - h_v was clamped
    h_v_dn = sigma - max(sigma - h_v, 1e-8)
    vega = (p_v_up - p_v_dn) / (h_v + h_v_dn) * 0.01  # per 1% vol move

    # --- Rho -----------------------------------------------------------------
    h_r = 0.01
    p_r_up = _crr_tree(S, K, T, r + h_r, sigma, q, option_type, steps)
    p_r_dn = _crr_tree(S, K, T, r - h_r, sigma, q, option_type, steps)
    rho = (p_r_up - p_r_dn) / (2.0 * h_r)

    return GreeksResult(
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
        rho=float(rho),
        model_name="Bayesian Tree",
    )
