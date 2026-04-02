"""Typeahead search component for stock selection in the sidebar."""

from __future__ import annotations

import streamlit as st

from src.data.nifty500 import load_nifty500, search_stocks


def render_search_bar() -> str | None:
    """Render a typeahead search bar for Nifty 50 stock selection.

    Displays a text input for searching stocks by symbol or company name,
    followed by a selectbox showing filtered results. The selected symbol
    is persisted in ``st.session_state`` so it survives Streamlit reruns.

    Returns:
        The bare NSE symbol string (e.g. ``"RELIANCE"``) if a stock is
        selected, or ``None`` if nothing has been chosen yet.
    """
    # Ensure session-state key exists
    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol: str | None = None

    query: str = st.text_input(
        "Search stocks",
        key="stock_search",
        placeholder="Search stocks (e.g. RELIANCE, TCS...)",
        label_visibility="collapsed",
    )

    # Load full Nifty 50 list and filter by query
    nifty_df = load_nifty500()
    matches_df = search_stocks(query, nifty_df) if query else nifty_df.head(10)

    if not matches_df.empty:
        # Build display labels: "SYMBOL — Company Name"
        options: list[str] = [
            f"{row['symbol']} \u2014 {row['company_name']}"
            for _, row in matches_df.iterrows()
        ]

        selected_label: str = st.selectbox(
            "Select a stock",
            options=options,
            key="stock_select_box",
            label_visibility="collapsed",
        )

        if selected_label:
            # Extract bare symbol (everything before the em-dash)
            symbol: str = selected_label.split(" \u2014 ")[0].strip()
            st.session_state.selected_symbol = symbol
    elif query:
        st.caption("No matching stocks found.")

    return st.session_state.selected_symbol
