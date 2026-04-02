"""OptiPrice India — Neural-Augmented Options Pricing Engine for Indian Equities."""

import streamlit as st
import numpy as np

from src.data.yahoo_fetcher import compute_historical_volatility, fetch_stock_data
from src.models import GreeksResult, PricingResult
from src.models.bayesian_tree import greeks as bayes_greeks
from src.models.bayesian_tree import price as bayes_price
from src.models.black_scholes import greeks as bs_greeks
from src.models.black_scholes import price as bs_price
from src.models.neural_ensemble import price as nn_price
from src.ui.charts import (
    create_convergence_plot,
    create_greeks_radar,
    create_posterior_histogram,
    create_price_comparison_bar,
)
from src.ui.charts_extended import (
    create_ensemble_disagreement,
    create_price_vs_strike,
    create_sensitivity_tornado,
)
from src.ui.components import render_greeks_table, render_price_card
from src.ui.sidebar import render_sidebar
from src.ui.styles import inject_custom_css
from src.utils.config import BAYESIAN_COLOR, BS_COLOR, NN_COLOR, VOLATILITY_LOOKBACK_OPTIONS

st.set_page_config(
    page_title="OptiPrice India",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# Header with market context
col_title, col_market = st.columns([3, 1])
with col_title:
    st.title("OptiPrice India")
    st.caption("Neural-Augmented Options Pricing Engine for Indian Equities")

with col_market:
    # Fetch Nifty 50 index level
    try:
        nifty_data = fetch_stock_data("NIFTY")
        if nifty_data:
            nifty_price = nifty_data.get("current_price", 0)
            nifty_change = nifty_data.get("day_change_pct", 0)
            change_color = "#00e676" if nifty_change >= 0 else "#ff1744"
            st.markdown(
                f'<div style="text-align:right; font-size:0.75rem; color:#8888a0;">NIFTY 50</div>'
                f'<div style="text-align:right; font-size:1.1rem; font-weight:600;">{nifty_price:,.1f}</div>'
                f'<div style="text-align:right; font-size:0.85rem; color:{change_color};">'
                f'{nifty_change:+.2f}%</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

st.markdown("---")

# ── Sidebar: stock selection + configuration ──
ctx = render_sidebar()

# ── Main content area ──
if not ctx["symbol"] or not ctx["stock_data"]:
    st.markdown(
        "👈 **Search and select a stock** from the Nifty 500 universe to begin pricing options."
    )
    st.stop()

if not ctx["submitted"]:
    st.info(
        f"Selected: **{ctx['symbol']}** — "
        f"₹{ctx['stock_data']['current_price']:,.2f}. "
        "Configure options in the sidebar and click **Calculate**."
    )
    st.stop()

# ── Run pricing models ──
S = ctx["stock_data"]["current_price"]
K = ctx["K"]
T = ctx["T"]
r = ctx["r"]
sigma = ctx["sigma"]
q = ctx["q"]
option_type = ctx["option_type"]

pricing_results: list[PricingResult] = []
greeks_results: list[GreeksResult] = []

# Black-Scholes (always runs)
with st.spinner("Computing Black-Scholes..."):
    bs_result = bs_price(S, K, T, r, sigma, q, option_type)
    bs_greeks_result = bs_greeks(S, K, T, r, sigma, q, option_type)
    pricing_results.append(bs_result)
    greeks_results.append(bs_greeks_result)

# Bayesian Binomial Tree
with st.spinner("Computing Bayesian Binomial Tree..."):
    vol_history: list[float] | None = None
    hist = ctx.get("historical")
    if hist is not None and not hist.empty and "Close" in hist.columns:
        vol_history = [
            compute_historical_volatility(hist["Close"], window=w)
            for w in VOLATILITY_LOOKBACK_OPTIONS
        ]

    bayes_result = bayes_price(
        S, K, T, r, sigma, q, option_type, vol_history=vol_history,
    )
    bayes_greeks_result = bayes_greeks(S, K, T, r, sigma, q, option_type)
    pricing_results.append(bayes_result)
    greeks_results.append(bayes_greeks_result)

# Neural Network Ensemble
with st.spinner("Computing Neural Network Ensemble..."):
    try:
        nn_result = nn_price(
            S, K, T, r, sigma, q, option_type,
            historical_data=ctx.get("historical"),
            symbol=ctx["symbol"],
        )
        pricing_results.append(nn_result)
        
        # For now, use BS Greeks for NN (can be enhanced later)
        nn_greeks_result = GreeksResult(
            model_name="Neural Ensemble",
            delta=bs_greeks_result.delta,
            gamma=bs_greeks_result.gamma,
            theta=bs_greeks_result.theta,
            vega=bs_greeks_result.vega,
            rho=bs_greeks_result.rho,
        )
        greeks_results.append(nn_greeks_result)
    except Exception as e:
        st.warning(f"Neural Ensemble computation failed: {e}. Showing BS and Bayesian only.")

# ── Display results ──
st.markdown("---")

# Price cards row
st.subheader("Option Prices")
cols = st.columns(3)

with cols[0]:
    render_price_card(bs_result, bs_greeks_result, BS_COLOR, "card-bs")

with cols[1]:
    render_price_card(bayes_result, bayes_greeks_result, BAYESIAN_COLOR, "card-bayesian")

with cols[2]:
    if len(pricing_results) >= 3:
        render_price_card(pricing_results[2], greeks_results[2], NN_COLOR, "card-nn")
    else:
        st.markdown(
            '<div class="model-card card-nn" style="opacity:0.3;">'
            '<div style="font-size:0.70rem; text-transform:uppercase; letter-spacing:0.08em; '
            'color:#8888a0;">Neural Ensemble</div>'
            '<div style="font-size:1.2rem; color:#8888a0;">Training...</div>'
            '</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# Tabs for detailed views
tab_prices, tab_greeks, tab_bayesian, tab_neural, tab_sensitivity = st.tabs(
    ["Prices", "Greeks", "Bayesian", "Neural", "Sensitivity"]
)

with tab_prices:
    st.plotly_chart(
        create_price_comparison_bar(pricing_results),
        use_container_width=True,
    )
    
    # Price vs Strike curve
    st.markdown("### Price vs Strike")
    st.caption("Compare how option prices vary across different strike prices")
    
    # Generate prices for a range of strikes
    strike_range = np.linspace(S * 0.7, S * 1.3, 15)
    results_by_strike = {}
    
    for strike in strike_range:
        strike_results = []
        strike_results.append(bs_price(S, strike, T, r, sigma, q, option_type))
        strike_results.append(bayes_price(S, strike, T, r, sigma, q, option_type, vol_history=vol_history))
        if len(pricing_results) >= 3:
            try:
                strike_results.append(nn_price(S, strike, T, r, sigma, q, option_type, 
                                               historical_data=ctx.get("historical"), 
                                               symbol=ctx["symbol"]))
            except Exception:
                pass
        results_by_strike[strike] = strike_results
    
    st.plotly_chart(
        create_price_vs_strike(results_by_strike, S),
        use_container_width=True,
    )

with tab_greeks:
    render_greeks_table(greeks_results)
    st.plotly_chart(
        create_greeks_radar(greeks_results),
        use_container_width=True,
    )

with tab_bayesian:
    if bayes_result.posterior_samples is not None and bayes_result.confidence_interval is not None:
        st.plotly_chart(
            create_posterior_histogram(
                bayes_result.posterior_samples,
                bayes_result.price,
                bayes_result.confidence_interval,
            ),
            use_container_width=True,
        )

    # Convergence plot: run tree at various step counts
    from src.models.bayesian_tree import _crr_tree

    convergence_steps = [10, 20, 50, 100, 200, 500]
    convergence_data = [
        (steps, _crr_tree(S, K, T, r, sigma, q, option_type, steps))
        for steps in convergence_steps
    ]
    st.plotly_chart(
        create_convergence_plot(convergence_data, bs_result.price),
        use_container_width=True,
    )

with tab_neural:
    if len(pricing_results) >= 3 and pricing_results[2].ensemble_prices:
        st.markdown("### Ensemble Disagreement")
        st.caption("Individual network predictions and confidence-weighted blending")
        
        nn_result = pricing_results[2]
        ensemble_prices = nn_result.ensemble_prices
        mean_price = float(np.mean(ensemble_prices))
        
        st.plotly_chart(
            create_ensemble_disagreement(
                ensemble_prices,
                mean_price,
                nn_result.price,
            ),
            use_container_width=True,
        )
        
        # Show confidence metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ensemble Mean", f"₹{mean_price:.2f}")
        with col2:
            ensemble_std = float(np.std(ensemble_prices))
            st.metric("Ensemble Std Dev", f"₹{ensemble_std:.2f}")
        with col3:
            if hasattr(nn_result, 'confidence_score') and nn_result.confidence_score:
                st.metric("Confidence Score", f"{nn_result.confidence_score:.1%}")
    else:
        st.info("Neural ensemble data not available. The model may still be training.")

with tab_sensitivity:
    st.markdown("### Parameter Sensitivity Analysis")
    st.caption("Impact of ±10% parameter changes on option price")
    
    # Compute sensitivities for Black-Scholes
    sensitivities = {}
    
    # S (spot price)
    sensitivities["Spot Price"] = (
        bs_price(S * 0.9, K, T, r, sigma, q, option_type).price,
        bs_price(S * 1.1, K, T, r, sigma, q, option_type).price,
    )
    
    # Volatility
    sensitivities["Volatility"] = (
        bs_price(S, K, T, r, sigma * 0.9, q, option_type).price,
        bs_price(S, K, T, r, sigma * 1.1, q, option_type).price,
    )
    
    # Time to expiry
    sensitivities["Time to Expiry"] = (
        bs_price(S, K, T * 0.9, r, sigma, q, option_type).price,
        bs_price(S, K, T * 1.1, r, sigma, q, option_type).price,
    )
    
    # Risk-free rate
    sensitivities["Risk-Free Rate"] = (
        bs_price(S, K, T, r * 0.9, sigma, q, option_type).price,
        bs_price(S, K, T, r * 1.1, sigma, q, option_type).price,
    )
    
    st.plotly_chart(
        create_sensitivity_tornado(bs_result, sensitivities),
        use_container_width=True,
    )

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#8888a0; font-size:0.75rem;">'
    'OptiPrice India • Built with Streamlit • Powered by Yahoo Finance • '
    'Models: Black-Scholes, Bayesian Trees, Neural Ensemble (LSTM)'
    '</div>',
    unsafe_allow_html=True,
)
