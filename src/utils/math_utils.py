"""Safe math helpers used across pricing models."""

from __future__ import annotations

import numpy as np


def safe_log(x: float) -> float:
    """Compute ln(x) safely, clamping x to a tiny positive floor.

    Avoids ``log(0)`` or ``log(negative)`` domain errors.
    """
    return float(np.log(max(x, 1e-10)))


def safe_sqrt(x: float) -> float:
    """Compute sqrt(x) safely, clamping x to zero.

    Returns 0.0 for any negative input instead of raising.
    """
    return float(np.sqrt(max(x, 0.0)))


def safe_exp(x: float) -> float:
    """Compute exp(x) safely, clipping the exponent to [-500, 500].

    Prevents overflow (exp(710) = inf) and underflow (exp(-746) = 0).
    """
    return float(np.exp(np.clip(x, -500, 500)))


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return float(max(lo, min(value, hi)))


def compute_rsi(prices: "pd.Series", period: int = 14) -> float:
    """Standard RSI calculation: average gains vs average losses over *period*.

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g. closing prices).
    period : int
        Look-back window (default 14).

    Returns
    -------
    float
        RSI value in [0, 100].  Returns 50.0 when there is not enough data.
    """
    import pandas as pd  # local import to keep module lightweight

    if not isinstance(prices, pd.Series) or len(prices) < period + 1:
        return 50.0

    deltas = prices.diff().dropna()
    if len(deltas) < period:
        return 50.0

    gains = deltas.clip(lower=0.0)
    losses = (-deltas).clip(lower=0.0)

    avg_gain = gains.iloc[:period].mean()
    avg_loss = losses.iloc[:period].mean()

    # Wilder smoothing for the remaining values
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period

    if avg_loss < 1e-12:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return float(np.clip(rsi, 0.0, 100.0))


def compute_vol_ratio(
    prices: "pd.Series", short: int = 5, long: int = 20
) -> float:
    """Ratio of short-term to long-term annualised volatility.

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g. closing prices).
    short : int
        Short-term look-back window (default 5).
    long : int
        Long-term look-back window (default 20).

    Returns
    -------
    float
        vol_short / vol_long.  Returns 1.0 when there is not enough data.
    """
    import pandas as pd  # local import

    if not isinstance(prices, pd.Series) or len(prices) < long + 1:
        return 1.0

    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < long:
        return 1.0

    vol_short = log_returns.iloc[-short:].std() * np.sqrt(252)
    vol_long = log_returns.iloc[-long:].std() * np.sqrt(252)

    if vol_long < 1e-12:
        return 1.0

    return float(vol_short / vol_long)
