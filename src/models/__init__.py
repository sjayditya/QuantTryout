from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PricingResult:
    """Result from any pricing model."""

    price: float
    model_name: str
    confidence_interval: tuple[float, float] | None = None
    posterior_samples: np.ndarray | None = None
    ensemble_prices: list[float] | None = None
    confidence_score: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GreeksResult:
    """Option Greeks from any pricing model."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    model_name: str
