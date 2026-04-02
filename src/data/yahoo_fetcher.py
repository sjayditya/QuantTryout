"""Single gateway to Yahoo Finance via yfinance.

ALL yfinance interactions go through this module.  NSE tickers get the
``.NS`` suffix appended here -- callers should pass the bare symbol
(e.g. ``"RELIANCE"``, not ``"RELIANCE.NS"``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from src.utils.config import CACHE_TTL_MARKET_DATA, DEFAULT_VOLATILITY_LOOKBACK

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _nse_ticker(symbol: str) -> str:
    """Return the Yahoo Finance ticker with ``.NS`` suffix."""
    return f"{symbol.strip()}.NS"


# ---------------------------------------------------------------------------
# Market snapshot
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_MARKET_DATA)
def fetch_stock_data(symbol: str) -> dict | None:
    """Fetch a real-time market snapshot for an NSE-listed stock.

    Args:
        symbol: Bare NSE symbol (e.g. ``"RELIANCE"``).

    Returns:
        A dict with keys: ``current_price``, ``prev_close``,
        ``day_change_pct``, ``high_52w``, ``low_52w``, ``market_cap``,
        ``dividend_yield``, ``currency`` -- or ``None`` on any failure.
    """
    try:
        ticker = yf.Ticker(_nse_ticker(symbol))
        info = ticker.info

        current_price: float = info.get("currentPrice") or info.get("regularMarketPrice", 0.0)
        prev_close: float = info.get("previousClose") or info.get("regularMarketPreviousClose", 0.0)

        if prev_close and prev_close != 0:
            day_change_pct = ((current_price - prev_close) / prev_close) * 100
        else:
            day_change_pct = 0.0

        return {
            "current_price": float(current_price),
            "prev_close": float(prev_close),
            "day_change_pct": round(float(day_change_pct), 2),
            "high_52w": float(info.get("fiftyTwoWeekHigh", 0.0)),
            "low_52w": float(info.get("fiftyTwoWeekLow", 0.0)),
            "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
            "currency": info.get("currency", "INR"),
        }
    except Exception:
        logger.exception("Failed to fetch stock data for %s", symbol)
        return None


# ---------------------------------------------------------------------------
# Historical OHLCV
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_MARKET_DATA)
def fetch_historical(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    """Fetch historical OHLCV data for an NSE stock.

    Args:
        symbol: Bare NSE symbol.
        period: yfinance period string (e.g. ``"1y"``, ``"6mo"``).

    Returns:
        A pandas DataFrame with columns ``Open``, ``High``, ``Low``,
        ``Close``, ``Volume`` -- or ``None`` on failure.
    """
    try:
        ticker = yf.Ticker(_nse_ticker(symbol))
        hist = ticker.history(period=period)

        if hist is None or hist.empty:
            return None

        return hist
    except Exception:
        logger.exception("Failed to fetch historical data for %s", symbol)
        return None


# ---------------------------------------------------------------------------
# Option chain
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_MARKET_DATA)
def fetch_option_chain(symbol: str) -> dict | None:
    """Fetch the option chain for the nearest expiry of an NSE stock.

    Args:
        symbol: Bare NSE symbol.

    Returns:
        A dict ``{"calls": DataFrame, "puts": DataFrame,
        "expirations": list[str]}`` -- or ``None`` on failure.
    """
    try:
        ticker = yf.Ticker(_nse_ticker(symbol))
        expirations = ticker.options  # tuple of date strings

        if not expirations:
            return None

        nearest_expiry = expirations[0]
        chain = ticker.option_chain(nearest_expiry)

        return {
            "calls": chain.calls,
            "puts": chain.puts,
            "expirations": list(expirations),
        }
    except Exception:
        logger.exception("Failed to fetch option chain for %s", symbol)
        return None


# ---------------------------------------------------------------------------
# Historical volatility
# ---------------------------------------------------------------------------

def compute_historical_volatility(
    prices: pd.Series,
    window: int = DEFAULT_VOLATILITY_LOOKBACK,
) -> float:
    """Compute annualised historical volatility from a price series.

    Uses the standard deviation of log-returns scaled to an annual figure
    (assuming 252 trading days per year).

    Args:
        prices: A ``pd.Series`` of closing prices (oldest first).
        window: Number of trailing trading days to use.

    Returns:
        Annualised volatility as a float.  Falls back to ``0.3`` (30 %)
        when there is insufficient data.
    """
    default_vol: float = 0.3

    if prices is None or len(prices) < 2:
        return default_vol

    # Use the most recent *window* prices (plus one for the first return)
    tail = prices.tail(window + 1)

    if len(tail) < 2:
        return default_vol

    log_returns = np.log(tail / tail.shift(1)).dropna()

    if log_returns.empty:
        return default_vol

    annualised_vol = float(log_returns.std() * np.sqrt(252))

    # Sanity: if result is non-finite or essentially zero, fall back
    if not np.isfinite(annualised_vol) or annualised_vol < 1e-8:
        return default_vol

    return annualised_vol
