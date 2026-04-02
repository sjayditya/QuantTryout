"""Sidebar orchestration for OptiPrice India.

Phase 1 handles stock search, info-card display, and a price sparkline.
Phase 2 adds option-configuration controls (type, strike, expiry, vol, etc.).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.data.nifty500 import get_stock_info, load_nifty500
from src.data.yahoo_fetcher import (
    compute_historical_volatility,
    fetch_historical,
    fetch_stock_data,
)
from src.ui.components import render_sparkline, render_stock_info_card
from src.ui.search import render_search_bar
from src.utils.config import DEFAULT_RISK_FREE_RATE, VOLATILITY_LOOKBACK_OPTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_to_nearest(value: float, base: float) -> float:
    """Round *value* to the nearest multiple of *base*."""
    return round(value / base) * base


def _default_strike(price: float) -> float:
    """Return a sensible default strike price rounded to the nearest 50 or 10."""
    if price > 500:
        return _round_to_nearest(price, 50.0)
    return _round_to_nearest(price, 10.0)


# ---------------------------------------------------------------------------
# Main sidebar renderer
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render the sidebar and return the current selection + options config.

    Returns:
        A dictionary with the following keys:

        - ``"symbol"`` -- the selected NSE symbol (``str``) or ``None``.
        - ``"stock_data"`` -- dict of current market data or ``None``.
        - ``"historical"`` -- ``pd.DataFrame`` of historical prices or ``None``.
        - ``"option_type"`` -- ``"call"`` or ``"put"`` (``None`` until submitted).
        - ``"K"`` -- strike price as ``float`` (``None`` until submitted).
        - ``"T"`` -- time to expiry in years (``None`` until submitted).
        - ``"r"`` -- risk-free rate as decimal (``None`` until submitted).
        - ``"sigma"`` -- volatility as decimal (``None`` until submitted).
        - ``"q"`` -- dividend yield as decimal (``None`` until submitted).
        - ``"submitted"`` -- ``True`` when the user clicks *Calculate*.
    """
    result: dict = {
        "symbol": None,
        "stock_data": None,
        "historical": None,
        "option_type": None,
        "K": None,
        "T": None,
        "r": None,
        "sigma": None,
        "q": None,
        "submitted": None,
    }

    with st.sidebar:
        st.markdown("### Stock Selection")

        symbol: str | None = render_search_bar()

        if symbol is None:
            st.caption("Search and select a stock to get started.")
            return result

        # Fetch live data and historical prices
        stock_data: dict | None = fetch_stock_data(symbol)
        historical: pd.DataFrame | None = fetch_historical(symbol)

        if stock_data is None:
            st.error(f"Could not fetch data for {symbol}.")
            result["symbol"] = symbol
            return result

        result["symbol"] = symbol
        result["stock_data"] = stock_data
        result["historical"] = historical

        # Get company info from Nifty CSV (not from yfinance)
        nifty_df = load_nifty500()
        csv_info = get_stock_info(symbol, nifty_df)
        nifty_info: dict = csv_info if csv_info else {
            "symbol": symbol,
            "company_name": symbol,
            "sector": "N/A",
        }

        render_stock_info_card(stock_data, nifty_info)

        # Show a sparkline if historical data is available
        if historical is not None and not historical.empty:
            if "Close" in historical.columns:
                render_sparkline(historical["Close"])

        # ---------------------------------------------------------------
        # Options configuration form
        # ---------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Options Configuration")

        current_price: float = stock_data.get("current_price", 0.0)

        # Pre-compute historical volatility for display / default
        hist_vol: float = 0.30
        close_series: pd.Series | None = None
        if historical is not None and not historical.empty and "Close" in historical.columns:
            close_series = historical["Close"]

        # Session-state defaults for expiry linkage
        if "oc_expiry_date" not in st.session_state:
            st.session_state["oc_expiry_date"] = date.today() + timedelta(days=30)
        if "oc_days_to_expiry" not in st.session_state:
            st.session_state["oc_days_to_expiry"] = 30

        with st.form("options_config"):
            # 1. Option type
            option_type_label: str = st.radio(
                "Option Type", ["Call", "Put"], horizontal=True
            )

            # 2. Strike price
            default_strike: float = _default_strike(current_price) if current_price > 0 else 100.0
            strike: float = st.number_input(
                "Strike Price (\u20b9)",
                min_value=0.01,
                value=default_strike,
                step=10.0,
                format="%.2f",
            )

            # 3. Expiry date & days to expiry
            expiry_date: date = st.date_input(
                "Expiry Date",
                value=st.session_state["oc_expiry_date"],
                min_value=date.today() + timedelta(days=1),
            )
            days_to_expiry_from_date: int = max((expiry_date - date.today()).days, 1)

            days_to_expiry: int = st.number_input(
                "Days to Expiry",
                min_value=1,
                max_value=3650,
                value=days_to_expiry_from_date,
                step=1,
            )

            T: float = days_to_expiry / 365.0

            # 4. Risk-free rate
            r_pct: float = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=15.0,
                value=DEFAULT_RISK_FREE_RATE * 100,
                step=0.25,
                help="Annualized yield on Indian 10-year Government Securities",
            )

            # 5. Volatility
            vol_lookback: int = st.selectbox(
                "Volatility Lookback (days)",
                options=VOLATILITY_LOOKBACK_OPTIONS,
                index=VOLATILITY_LOOKBACK_OPTIONS.index(252),
            )

            if close_series is not None:
                hist_vol = compute_historical_volatility(close_series, window=vol_lookback)

            st.metric("Historical Volatility", f"{hist_vol * 100:.2f}%")

            override_vol: bool = st.checkbox("Override Volatility")
            if override_vol:
                sigma_pct: float = st.slider(
                    "Volatility (%)",
                    min_value=1.0,
                    max_value=200.0,
                    value=round(hist_vol * 100, 2),
                    step=0.5,
                )
            else:
                sigma_pct = hist_vol * 100

            # 6. Dividend yield
            raw_div_yield = stock_data.get("dividend_yield")
            div_yield_display: float = (raw_div_yield * 100) if raw_div_yield else 0.0

            q_pct: float = st.number_input(
                "Dividend Yield (%)",
                min_value=0.0,
                max_value=50.0,
                value=round(div_yield_display, 4),
                step=0.1,
                format="%.4f",
            )

            # 7. Submit
            submitted: bool = st.form_submit_button("Calculate")

        # Sync session-state for expiry linkage on next rerun
        st.session_state["oc_expiry_date"] = expiry_date
        st.session_state["oc_days_to_expiry"] = days_to_expiry

        if submitted:
            result["option_type"] = option_type_label.lower()
            result["K"] = float(strike)
            result["T"] = T
            result["r"] = r_pct / 100.0
            result["sigma"] = sigma_pct / 100.0
            result["q"] = q_pct / 100.0
            result["submitted"] = True

    return result
