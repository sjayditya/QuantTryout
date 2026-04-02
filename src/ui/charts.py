"""Chart components for OptiPrice India."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.models import GreeksResult, PricingResult
from src.utils.config import (
    BAYESIAN_COLOR,
    BS_COLOR,
    CHART_CONFIG,
    NN_COLOR,
)

# Canonical colour mapping --------------------------------------------------
MODEL_COLORS: dict[str, str] = {
    "Black-Scholes": BS_COLOR,
    "Bayesian Tree": BAYESIAN_COLOR,
    "Neural Ensemble": NN_COLOR,
}


# ---------------------------------------------------------------------------
# Price comparison bar chart
# ---------------------------------------------------------------------------

def create_price_comparison_bar(results: list[PricingResult]) -> go.Figure:
    """Create a grouped bar chart comparing price estimates across models.

    Each bar represents one model's estimated option price.  If the model
    provides a ``confidence_interval`` (Bayesian) or ``ensemble_prices``
    (Neural Ensemble), error bars are shown.

    Args:
        results: One to three :class:`PricingResult` objects from different
            pricing models.

    Returns:
        A Plotly :class:`go.Figure` ready for rendering.
    """
    fig = go.Figure()

    for res in results:
        color = MODEL_COLORS.get(res.model_name, BS_COLOR)

        # Determine error bar values if available
        error_y: dict | None = None

        if res.confidence_interval is not None:
            lower, upper = res.confidence_interval
            error_y = dict(
                type="data",
                symmetric=False,
                array=[upper - res.price],
                arrayminus=[res.price - lower],
                color=color,
                thickness=1.5,
                width=6,
            )
        elif res.ensemble_prices is not None and len(res.ensemble_prices) > 1:
            import numpy as np

            std = float(np.std(res.ensemble_prices))
            error_y = dict(
                type="data",
                symmetric=True,
                array=[std],
                color=color,
                thickness=1.5,
                width=6,
            )

        fig.add_trace(
            go.Bar(
                x=[res.model_name],
                y=[res.price],
                name=res.model_name,
                marker_color=color,
                error_y=error_y,
                text=[f"\u20b9{res.price:,.2f}"],
                textposition="outside",
                width=0.35,
            )
        )

    fig.update_layout(
        template=CHART_CONFIG["template"],
        paper_bgcolor=CHART_CONFIG["paper_bgcolor"],
        plot_bgcolor=CHART_CONFIG["plot_bgcolor"],
        font=CHART_CONFIG["font"],
        margin=CHART_CONFIG["margin"],
        title="Model Price Comparison",
        yaxis_title="Option Price (\u20b9)",
        showlegend=False,
        bargap=0.3,
        height=380,
    )

    return fig


# ---------------------------------------------------------------------------
# Bayesian posterior histogram
# ---------------------------------------------------------------------------

def create_posterior_histogram(
    samples: np.ndarray,
    point_estimate: float,
    ci: tuple[float, float],
) -> go.Figure:
    """Create a histogram of Bayesian posterior price samples.

    Displays vertical dashed lines for the point estimate (mean) and the
    95 % credible interval bounds.

    Args:
        samples: 1-D array of posterior price samples.
        point_estimate: The mean (point) estimate shown as an amber line.
        ci: ``(lower, upper)`` bounds of the 95 % credible interval.

    Returns:
        A Plotly :class:`go.Figure` ready for rendering.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=samples,
            marker_color=BAYESIAN_COLOR,
            opacity=0.75,
            name="Posterior samples",
        )
    )

    # Point estimate (mean) — amber dashed line
    fig.add_vline(
        x=point_estimate,
        line_dash="dash",
        line_color=BAYESIAN_COLOR,
        annotation_text=f"Mean ₹{point_estimate:,.2f}",
        annotation_position="top right",
    )

    # 95 % CI bounds — lighter colour
    ci_color = "#ffe082"  # lighter amber
    fig.add_vline(
        x=ci[0],
        line_dash="dash",
        line_color=ci_color,
        annotation_text=f"Lower ₹{ci[0]:,.2f}",
        annotation_position="bottom left",
    )
    fig.add_vline(
        x=ci[1],
        line_dash="dash",
        line_color=ci_color,
        annotation_text=f"Upper ₹{ci[1]:,.2f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        template=CHART_CONFIG["template"],
        paper_bgcolor=CHART_CONFIG["paper_bgcolor"],
        plot_bgcolor=CHART_CONFIG["plot_bgcolor"],
        font=CHART_CONFIG["font"],
        margin=CHART_CONFIG["margin"],
        title="Bayesian Posterior Distribution",
        xaxis_title="Option Price (₹)",
        yaxis_title="Frequency",
        showlegend=False,
        height=380,
    )

    return fig


# ---------------------------------------------------------------------------
# Binomial tree convergence plot
# ---------------------------------------------------------------------------

def create_convergence_plot(
    convergence_data: list[tuple[int, float]],
    bs_price: float,
) -> go.Figure:
    """Create a line plot showing binomial tree convergence.

    Shows how the tree-based price converges towards the Black-Scholes
    analytical price as the number of steps increases.

    Args:
        convergence_data: Sequence of ``(steps, price)`` tuples.
        bs_price: The Black-Scholes analytical reference price.

    Returns:
        A Plotly :class:`go.Figure` ready for rendering.
    """
    steps = [d[0] for d in convergence_data]
    prices = [d[1] for d in convergence_data]

    fig = go.Figure()

    # Tree convergence line
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=prices,
            mode="lines+markers",
            line=dict(color=BAYESIAN_COLOR, width=2),
            marker=dict(size=5),
            name="Tree Price",
        )
    )

    # BS analytical reference line
    fig.add_hline(
        y=bs_price,
        line_dash="dash",
        line_color=BS_COLOR,
        annotation_text="BS Analytical",
        annotation_position="top left",
    )

    fig.update_layout(
        template=CHART_CONFIG["template"],
        paper_bgcolor=CHART_CONFIG["paper_bgcolor"],
        plot_bgcolor=CHART_CONFIG["plot_bgcolor"],
        font=CHART_CONFIG["font"],
        margin=CHART_CONFIG["margin"],
        title="Binomial Tree Convergence",
        xaxis_title="Tree Steps",
        yaxis_title="Option Price (₹)",
        showlegend=True,
        height=380,
    )

    return fig


# ---------------------------------------------------------------------------
# Greeks radar chart
# ---------------------------------------------------------------------------

def create_greeks_radar(greeks_list: list[GreeksResult]) -> go.Figure:
    """Create a radar (polar) chart comparing Greeks across models.

    Each Greek is normalised to ``[0, 1]`` using the absolute maximum
    across all models so the shapes are directly comparable.

    Args:
        greeks_list: One :class:`GreeksResult` per model.

    Returns:
        A Plotly :class:`go.Figure` ready for rendering.
    """
    categories = ["Delta", "Gamma", "Theta", "Vega", "Rho"]

    # Collect raw absolute values per Greek across all models
    raw: dict[str, list[float]] = {cat: [] for cat in categories}
    for g in greeks_list:
        raw["Delta"].append(abs(g.delta))
        raw["Gamma"].append(abs(g.gamma))
        raw["Theta"].append(abs(g.theta))
        raw["Vega"].append(abs(g.vega))
        raw["Rho"].append(abs(g.rho))

    # Compute max per Greek for normalisation (guard against zero)
    max_vals: dict[str, float] = {
        cat: max(vals) if max(vals) != 0 else 1.0 for cat, vals in raw.items()
    }

    fig = go.Figure()

    for idx, g in enumerate(greeks_list):
        values = [
            abs(g.delta) / max_vals["Delta"],
            abs(g.gamma) / max_vals["Gamma"],
            abs(g.theta) / max_vals["Theta"],
            abs(g.vega) / max_vals["Vega"],
            abs(g.rho) / max_vals["Rho"],
        ]
        # Close the polygon by repeating the first value
        values.append(values[0])

        color = MODEL_COLORS.get(g.model_name, BS_COLOR)

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=color,
                opacity=0.2,
                line=dict(color=color, width=2),
                name=g.model_name,
            )
        )

    fig.update_layout(
        template=CHART_CONFIG["template"],
        paper_bgcolor=CHART_CONFIG["paper_bgcolor"],
        plot_bgcolor=CHART_CONFIG["plot_bgcolor"],
        font=CHART_CONFIG["font"],
        margin=CHART_CONFIG["margin"],
        title="Greeks Comparison",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        showlegend=True,
        height=420,
    )

    return fig
