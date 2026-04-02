"""Extended chart components for OptiPrice India."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.models import PricingResult
from src.utils.config import (
    BAYESIAN_COLOR,
    BS_COLOR,
    CHART_CONFIG,
    NN_COLOR,
)

MODEL_COLORS: dict[str, str] = {
    "Black-Scholes": BS_COLOR,
    "Bayesian Tree": BAYESIAN_COLOR,
    "Neural Ensemble": NN_COLOR,
}


# ---------------------------------------------------------------------------
# Price vs Strike curve
# ---------------------------------------------------------------------------

def create_price_vs_strike(
    results_by_strike: dict[float, list[PricingResult]],
    current_price: float,
) -> go.Figure:
    """Create a line chart showing option price as a function of strike.

    Args:
        results_by_strike: Dict mapping strike prices to lists of PricingResults
            from different models.
        current_price: Current stock price (shown as vertical reference line).

    Returns:
        A Plotly go.Figure ready for rendering.
    """
    fig = go.Figure()

    strikes = sorted(results_by_strike.keys())
    model_names = ["Black-Scholes", "Bayesian Tree", "Neural Ensemble"]

    for model_name in model_names:
        prices = []
        for strike in strikes:
            results = results_by_strike[strike]
            matching = [r for r in results if r.model_name == model_name]
            if matching:
                prices.append(matching[0].price)
            else:
                prices.append(None)

        color = MODEL_COLORS.get(model_name, BS_COLOR)
        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=prices,
                mode="lines+markers",
                name=model_name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )

    # Current stock price reference line
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="#8888a0",
        annotation_text=f"Spot ₹{current_price:,.0f}",
        annotation_position="top",
    )

    fig.update_layout(
        template=CHART_CONFIG["template"],
        paper_bgcolor=CHART_CONFIG["paper_bgcolor"],
        plot_bgcolor=CHART_CONFIG["plot_bgcolor"],
        font=CHART_CONFIG["font"],
        margin=CHART_CONFIG["margin"],
        title="Option Price vs Strike Price",
        xaxis_title="Strike Price (₹)",
        yaxis_title="Option Price (₹)",
        showlegend=True,
        height=400,
    )

    return fig


# ---------------------------------------------------------------------------
# Sensitivity tornado chart
# ---------------------------------------------------------------------------

def create_sensitivity_tornado(
    base_result: PricingResult,
    sensitivities: dict[str, tuple[float, float]],
) -> go.Figure:
    """Create a tornado chart showing parameter sensitivity.

    Args:
        base_result: The baseline pricing result.
        sensitivities: Dict mapping parameter names to (price_down, price_up)
            tuples representing ±10% perturbations.

    Returns:
        A Plotly go.Figure ready for rendering.
    """
    fig = go.Figure()

    params = list(sensitivities.keys())
    base_price = base_result.price

    # Calculate deviations from base
    deviations_down = [base_price - sensitivities[p][0] for p in params]
    deviations_up = [sensitivities[p][1] - base_price for p in params]

    # Sort by total impact (sum of absolute deviations)
    total_impact = [abs(d) + abs(u) for d, u in zip(deviations_down, deviations_up)]
    sorted_indices = sorted(range(len(params)), key=lambda i: total_impact[i], reverse=True)
    params = [params[i] for i in sorted_indices]
    deviations_down = [deviations_down[i] for i in sorted_indices]
    deviations_up = [deviations_up[i] for i in sorted_indices]

    # Downside bars (negative impact)
    fig.add_trace(
        go.Bar(
            y=params,
            x=[-d for d in deviations_down],
            orientation="h",
            name="-10%",
            marker_color="#ff1744",
            text=[f"-₹{d:.2f}" for d in deviations_down],
            textposition="inside",
        )
    )

    # Upside bars (positive impact)
    fig.add_trace(
        go.Bar(
            y=params,
            x=deviations_up,
            orientation="h",
            name="+10%",
            marker_color="#00e676",
            text=[f"+₹{u:.2f}" for u in deviations_up],
            textposition="inside",
        )
    )

    fig.update_layout(
        template=CHART_CONFIG["template"],
        paper_bgcolor=CHART_CONFIG["paper_bgcolor"],
        plot_bgcolor=CHART_CONFIG["plot_bgcolor"],
        font=CHART_CONFIG["font"],
        margin=CHART_CONFIG["margin"],
        title=f"Sensitivity Analysis — {base_result.model_name}",
        xaxis_title="Price Impact (₹)",
        barmode="relative",
        showlegend=True,
        height=350,
    )

    return fig


# ---------------------------------------------------------------------------
# Ensemble disagreement plot
# ---------------------------------------------------------------------------

def create_ensemble_disagreement(
    ensemble_prices: list[float],
    mean_price: float,
    blended_price: float,
) -> go.Figure:
    """Create a strip plot showing individual ensemble predictions.

    Args:
        ensemble_prices: List of predictions from individual networks.
        mean_price: Ensemble mean price.
        blended_price: Final blended price (NN + BS weighted).

    Returns:
        A Plotly go.Figure ready for rendering.
    """
    fig = go.Figure()

    # Individual network predictions as scatter points
    fig.add_trace(
        go.Scatter(
            x=ensemble_prices,
            y=[0] * len(ensemble_prices),
            mode="markers",
            marker=dict(
                size=12,
                color=NN_COLOR,
                opacity=0.6,
                line=dict(width=1, color="#ffffff"),
            ),
            name="Individual Networks",
            showlegend=True,
        )
    )

    # Ensemble mean
    fig.add_vline(
        x=mean_price,
        line_dash="dash",
        line_color=NN_COLOR,
        line_width=2,
        annotation_text=f"Ensemble Mean ₹{mean_price:.2f}",
        annotation_position="top",
    )

    # Blended price (if different from mean)
    if abs(blended_price - mean_price) > 0.01:
        fig.add_vline(
            x=blended_price,
            line_dash="dot",
            line_color=BS_COLOR,
            line_width=2,
            annotation_text=f"Blended ₹{blended_price:.2f}",
            annotation_position="bottom",
        )

    fig.update_layout(
        template=CHART_CONFIG["template"],
        paper_bgcolor=CHART_CONFIG["paper_bgcolor"],
        plot_bgcolor=CHART_CONFIG["plot_bgcolor"],
        font=CHART_CONFIG["font"],
        margin=CHART_CONFIG["margin"],
        title="Neural Ensemble Disagreement",
        xaxis_title="Predicted Price (₹)",
        yaxis=dict(visible=False),
        showlegend=True,
        height=250,
    )

    return fig


# ---------------------------------------------------------------------------
# Volatility surface heatmap
# ---------------------------------------------------------------------------

def create_volatility_surface(
    strike_range: list[float],
    expiry_range: list[float],
    prices_grid: np.ndarray,
    model_name: str,
) -> go.Figure:
    """Create a heatmap showing option prices across strikes and expiries.

    Args:
        strike_range: List of strike prices (x-axis).
        expiry_range: List of time to expiry values in years (y-axis).
        prices_grid: 2D array of option prices, shape (len(expiry_range), len(strike_range)).
        model_name: Name of the model used for pricing.

    Returns:
        A Plotly go.Figure ready for rendering.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=prices_grid,
            x=strike_range,
            y=expiry_range,
            colorscale="Viridis",
            colorbar=dict(title="Price (₹)"),
        )
    )

    fig.update_layout(
        template=CHART_CONFIG["template"],
        paper_bgcolor=CHART_CONFIG["paper_bgcolor"],
        plot_bgcolor=CHART_CONFIG["plot_bgcolor"],
        font=CHART_CONFIG["font"],
        margin=CHART_CONFIG["margin"],
        title=f"Volatility Surface — {model_name}",
        xaxis_title="Strike Price (₹)",
        yaxis_title="Time to Expiry (Years)",
        height=450,
    )

    return fig
