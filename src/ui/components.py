"""Reusable UI components for OptiPrice India."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models import GreeksResult, PricingResult
from src.utils.config import (
    ACCENT_GREEN,
    ACCENT_RED,
    BG_SECONDARY,
    BORDER_COLOR,
    CHART_CONFIG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_market_cap(value: float) -> str:
    """Format a market-cap value into a human-readable Indian convention string.

    Args:
        value: Market capitalisation in absolute terms (e.g. in rupees).

    Returns:
        A formatted string such as ``"2.45 L Cr"``, ``"8,320 Cr"``, or the
        raw number with commas when it is below 1 Cr.
    """
    if value >= 1e12:
        return f"{value / 1e12:.2f} L Cr"
    if value >= 1e7:
        return f"{value / 1e7:,.0f} Cr"
    return f"{value:,.0f}"


# ---------------------------------------------------------------------------
# Stock info card
# ---------------------------------------------------------------------------

def render_stock_info_card(stock_data: dict, nifty_info: dict) -> None:
    """Render a styled stock-information card using the ``stock-card`` CSS class.

    The card shows the company name, symbol, current price, daily change
    percentage, 52-week range, market capitalisation, and sector.

    Args:
        stock_data: Dictionary returned by ``fetch_stock_data()`` with keys
            such as ``current_price``, ``day_change_pct``, ``week52_low``,
            ``week52_high``, and ``market_cap``.
        nifty_info: Dictionary from the Nifty 500 list containing
            ``company_name``, ``symbol``, and ``sector``.
    """
    symbol: str = nifty_info.get("symbol", "")
    company_name: str = nifty_info.get("company_name", symbol)
    sector: str = nifty_info.get("sector", "N/A")

    current_price: float = stock_data.get("current_price", 0.0)
    day_change_pct: float = stock_data.get("day_change_pct", 0.0)
    week52_low: float = stock_data.get("low_52w", 0.0)
    week52_high: float = stock_data.get("high_52w", 0.0)
    market_cap: float = stock_data.get("market_cap", 0.0)

    change_class: str = "positive" if day_change_pct >= 0 else "negative"
    change_sign: str = "+" if day_change_pct >= 0 else ""

    card_html: str = f"""
    <div class="stock-card">
        <div class="stock-name">{company_name}</div>
        <div class="stock-meta" style="margin-bottom:0.6rem;">{symbol}</div>
        <div class="stock-price" style="color:{TEXT_PRIMARY};">
            &#8377;{current_price:,.2f}
        </div>
        <div class="stock-change {change_class}" style="margin-bottom:0.8rem;">
            {change_sign}{day_change_pct:.2f}%
        </div>
        <div class="stock-meta">
            <b>52W Range:</b> &#8377;{week52_low:,.2f} &mdash; &#8377;{week52_high:,.2f}
        </div>
        <div class="stock-meta">
            <b>Market Cap:</b> &#8377;{format_market_cap(market_cap)}
        </div>
        <div class="stock-meta">
            <b>Sector:</b> {sector}
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

def render_sparkline(prices: pd.Series) -> None:
    """Render a minimal sparkline chart of recent closing prices.

    Displays the last 30 days of price data as a tiny Plotly line chart
    with no axis labels, minimal margins, and a transparent background.
    The line is green when the latest price exceeds the earliest, red
    otherwise.

    Args:
        prices: A ``pandas.Series`` of closing prices indexed by date.
            Only the last 30 values are used.
    """
    recent: pd.Series = prices.tail(30)

    if recent.empty:
        return

    line_color: str = (
        ACCENT_GREEN if recent.iloc[-1] >= recent.iloc[0] else ACCENT_RED
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.values,
            mode="lines",
            line=dict(color=line_color, width=2),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template=CHART_CONFIG["template"],
        paper_bgcolor=CHART_CONFIG["paper_bgcolor"],
        plot_bgcolor=CHART_CONFIG["plot_bgcolor"],
        font=CHART_CONFIG["font"],
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=80,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Price card
# ---------------------------------------------------------------------------

def render_price_card(
    result: PricingResult,
    greeks: GreeksResult | None,
    color: str,
    card_class: str,
) -> None:
    """Render a styled price card for a single pricing model.

    Args:
        result: The pricing result to display.
        greeks: Optional Greeks result; if provided, delta is shown.
        color: Hex colour used for the price text.
        card_class: CSS class for the card (e.g. ``"card-bs"``,
            ``"card-bayesian"``, ``"card-nn"``).
    """
    delta_html: str = ""
    if greeks is not None:
        delta_html = (
            f'<div style="font-size:0.85rem; color:{TEXT_SECONDARY};">'
            f"\u0394 = {greeks.delta:+.4f}</div>"
        )

    ci_html: str = ""
    if result.confidence_interval is not None:
        lower, upper = result.confidence_interval
        ci_html = (
            f'<div style="font-size:0.80rem; color:{TEXT_SECONDARY};">'
            f"CI: \u20b9{lower:,.2f} \u2013 \u20b9{upper:,.2f}</div>"
        )

    score_html: str = ""
    if result.confidence_score is not None:
        score_html = (
            f'<div style="font-size:0.80rem; color:{TEXT_SECONDARY};">'
            f"Confidence: {result.confidence_score * 100:.1f}%</div>"
        )

    card_html: str = f"""
    <div class="model-card {card_class}">
        <div style="font-size:0.70rem; text-transform:uppercase; letter-spacing:0.08em;
                    color:{TEXT_SECONDARY}; margin-bottom:0.3rem;">
            {result.model_name}
        </div>
        <div style="font-size:1.6rem; font-weight:700; color:{color};">
            \u20b9{result.price:,.2f}
        </div>
        {delta_html}
        {ci_html}
        {score_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Greeks comparison table
# ---------------------------------------------------------------------------

def render_greeks_table(greeks_results: list[GreeksResult]) -> None:
    """Render a comparison table of Greeks across pricing models.

    Rows correspond to individual Greeks (Delta, Gamma, Theta, Vega, Rho)
    and columns correspond to each model.

    Args:
        greeks_results: A list of :class:`GreeksResult` objects (one per model).
    """
    if not greeks_results:
        st.info("No Greeks data available.")
        return

    greek_names: list[str] = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
    data: dict[str, list[str]] = {"Greek": greek_names}

    for gr in greeks_results:
        data[gr.model_name] = [
            f"{gr.delta:+.6f}",
            f"{gr.gamma:.6f}",
            f"{gr.theta:+.6f}",
            f"{gr.vega:.6f}",
            f"{gr.rho:+.6f}",
        ]

    df = pd.DataFrame(data).set_index("Greek")

    st.dataframe(
        df.style.format(precision=6),
        use_container_width=True,
    )
