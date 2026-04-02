"""Nifty 500 stock universe loader and search utilities.

Loads the Nifty 500 CSV and provides fuzzy-ish search with ranked results
for the stock selector in the sidebar.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path to the CSV relative to the project root
# ---------------------------------------------------------------------------
_CSV_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "nifty500.csv"


@st.cache_resource
def load_nifty500() -> pd.DataFrame:
    """Read the Nifty 500 CSV into a DataFrame (cached for session lifetime).

    Returns:
        DataFrame with columns: symbol, company_name, sector, industry.

    Raises:
        FileNotFoundError: If the CSV file is missing from the expected path.
    """
    df = pd.read_csv(_CSV_PATH)
    # Normalise column names just in case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def search_stocks(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Case-insensitive search across symbol and company_name.

    Ranking order:
        1. Exact symbol match (highest priority)
        2. Symbol starts with *query*
        3. company_name contains *query* as substring

    Args:
        query: The user-typed search string.
        df: The Nifty 500 DataFrame (from ``load_nifty500``).

    Returns:
        Up to 10 matching rows sorted by the ranking described above.
    """
    if not query or not query.strip():
        return df.head(10)

    q = query.strip().upper()
    q_lower = query.strip().lower()

    # Build boolean masks
    symbol_upper = df["symbol"].str.upper()
    company_lower = df["company_name"].str.lower()

    exact_symbol = symbol_upper == q
    prefix_symbol = symbol_upper.str.startswith(q) & ~exact_symbol
    substring_company = company_lower.str.contains(q_lower, na=False) & ~exact_symbol & ~prefix_symbol

    # Assign a rank for sorting (lower is better)
    rank = pd.Series(float("inf"), index=df.index)
    rank[exact_symbol] = 0
    rank[prefix_symbol] = 1
    rank[substring_company] = 2

    matched = df[rank < float("inf")].copy()
    matched["_rank"] = rank[rank < float("inf")]
    matched = matched.sort_values("_rank").drop(columns="_rank").head(10)

    return matched.reset_index(drop=True)


def get_stock_info(symbol: str, df: pd.DataFrame) -> dict | None:
    """Look up a single stock by exact symbol.

    Args:
        symbol: NSE symbol (e.g. ``"RELIANCE"``).
        df: The Nifty 500 DataFrame.

    Returns:
        A dict with keys ``symbol``, ``company_name``, ``sector``,
        ``industry`` -- or ``None`` if the symbol is not found.
    """
    if not symbol:
        return None

    mask = df["symbol"].str.upper() == symbol.strip().upper()
    matches = df[mask]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()
