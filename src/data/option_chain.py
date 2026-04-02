"""Option chain formatting and ATM strike utilities.

Transforms raw yfinance option-chain DataFrames into clean, display-ready
tables and provides helper functions for strike selection.
"""

from __future__ import annotations

import pandas as pd


# Columns we want to keep for the UI display
_DISPLAY_COLUMNS: list[str] = [
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
]

# Friendlier column labels for display
_COLUMN_LABELS: dict[str, str] = {
    "strike": "Strike",
    "lastPrice": "Last Price",
    "bid": "Bid",
    "ask": "Ask",
    "volume": "Volume",
    "openInterest": "Open Interest",
    "impliedVolatility": "Implied Vol",
}


def format_option_chain(chain_data: dict, option_type: str) -> pd.DataFrame:
    """Format a raw option chain for display.

    Selects the relevant columns, renames them to human-friendly labels,
    and fills missing values.

    Args:
        chain_data: Dict returned by
            :func:`src.data.yahoo_fetcher.fetch_option_chain` with keys
            ``"calls"`` and ``"puts"``.
        option_type: Either ``"calls"`` or ``"puts"``.

    Returns:
        A cleaned DataFrame ready for ``st.dataframe()``.  Returns an
        empty DataFrame if the input is invalid.
    """
    if not chain_data or option_type not in chain_data:
        return pd.DataFrame()

    raw: pd.DataFrame = chain_data[option_type]

    if raw is None or raw.empty:
        return pd.DataFrame()

    # Keep only the columns that exist in the raw data
    available = [c for c in _DISPLAY_COLUMNS if c in raw.columns]

    if not available:
        return pd.DataFrame()

    formatted = raw[available].copy()

    # Fill missing numeric values with 0 / "--"
    for col in formatted.columns:
        if formatted[col].dtype in ("float64", "int64"):
            formatted[col] = formatted[col].fillna(0)
        else:
            formatted[col] = formatted[col].fillna("--")

    # Format implied volatility as percentage
    if "impliedVolatility" in formatted.columns:
        formatted["impliedVolatility"] = (
            formatted["impliedVolatility"].apply(
                lambda v: round(v * 100, 2) if isinstance(v, (int, float)) else v
            )
        )

    formatted = formatted.rename(columns=_COLUMN_LABELS)
    return formatted.reset_index(drop=True)


def get_atm_strike(current_price: float, strikes: list[float]) -> float:
    """Find the at-the-money strike closest to the current spot price.

    Args:
        current_price: The current market price of the underlying.
        strikes: List of available strike prices.

    Returns:
        The strike price with the smallest absolute distance to
        *current_price*.  Returns ``current_price`` itself if *strikes*
        is empty.
    """
    if not strikes:
        return current_price

    return min(strikes, key=lambda s: abs(s - current_price))
