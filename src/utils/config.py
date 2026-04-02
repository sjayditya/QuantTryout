"""Centralized configuration constants for OptiPrice India."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Market defaults
# ---------------------------------------------------------------------------
DEFAULT_RISK_FREE_RATE: float = 0.07  # Indian 10Y G-Sec yield (~7%)
DEFAULT_VOLATILITY_LOOKBACK: int = 252  # Trading days (1 year)
VOLATILITY_LOOKBACK_OPTIONS: list[int] = [30, 60, 90, 252]

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
CACHE_TTL_MARKET_DATA: int = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Model colors (consistent everywhere)
# ---------------------------------------------------------------------------
BS_COLOR: str = "#448aff"       # Blue — Black-Scholes
BAYESIAN_COLOR: str = "#ffab00"  # Amber — Bayesian Binomial Tree
NN_COLOR: str = "#b388ff"       # Purple — Neural Ensemble

# ---------------------------------------------------------------------------
# UI colors (from PRD §5.2)
# ---------------------------------------------------------------------------
BG_PRIMARY: str = "#0a0a0f"
BG_SECONDARY: str = "#12121a"
BG_ELEVATED: str = "#1a1a28"
BORDER_COLOR: str = "#2a2a3d"
TEXT_PRIMARY: str = "#e8e8f0"
TEXT_SECONDARY: str = "#8888a0"
ACCENT_GREEN: str = "#00e676"
ACCENT_RED: str = "#ff1744"
ACCENT_BLUE: str = "#448aff"
ACCENT_AMBER: str = "#ffab00"
ACCENT_PURPLE: str = "#b388ff"

# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
BS_IV_TOL: float = 1e-6
BS_IV_MAX_ITER: int = 100
BS_IV_SIGMA_BOUNDS: tuple[float, float] = (0.01, 5.0)

# ---------------------------------------------------------------------------
# Bayesian Binomial Tree
# ---------------------------------------------------------------------------
BAYESIAN_DEFAULT_STEPS: int = 100
BAYESIAN_MAX_STEPS: int = 500
BAYESIAN_DEFAULT_SAMPLES: int = 1000

# ---------------------------------------------------------------------------
# Neural Network Ensemble
# ---------------------------------------------------------------------------
NN_ENSEMBLE_SIZE: int = 5
NN_HIDDEN_DIM: int = 64
NN_DENSE_DIM: int = 32
NN_LSTM_LAYERS: int = 2
NN_INPUT_DIM: int = 7  # moneyness, T, sigma_hist, r, q, RSI_14, vol_ratio
NN_TRAINING_SAMPLES: int = 50_000
NN_EPOCHS: int = 50
NN_BATCH_SIZE: int = 256
NN_LEARNING_RATE: float = 1e-3
NN_MAX_TRAINING_TIME: int = 60  # seconds

# Blending: price_final = w * price_NN + (1-w) * price_BS
# w = sigmoid(-alpha * sigma_ensemble + beta)
NN_BLEND_ALPHA: float = 5.0
NN_BLEND_BETA: float = 2.0

# ---------------------------------------------------------------------------
# Heston stochastic volatility parameter bounds
# ---------------------------------------------------------------------------
HESTON_BOUNDS: dict[str, tuple[float, float]] = {
    "v0": (0.01, 1.0),      # Initial variance
    "kappa": (0.1, 10.0),   # Mean-reversion speed
    "theta": (0.01, 1.0),   # Long-run variance
    "xi": (0.1, 2.0),       # Vol of vol
    "rho": (-0.95, 0.0),    # Correlation
}

# Default Heston params (fallback when calibration fails)
HESTON_DEFAULTS: dict[str, float] = {
    "v0": 0.04,
    "kappa": 2.0,
    "theta": 0.04,
    "xi": 0.3,
    "rho": -0.7,
}

# ---------------------------------------------------------------------------
# Plotly chart configuration
# ---------------------------------------------------------------------------
CHART_CONFIG: dict = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "JetBrains Mono, monospace", "color": TEXT_PRIMARY},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
}


def get_device() -> str:
    """Return the best available torch device string.

    Returns 'cuda' if a CUDA GPU is available, otherwise 'cpu'.
    The actual device selection (user toggle) is handled in the UI layer
    via st.session_state; this function just detects capability.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
