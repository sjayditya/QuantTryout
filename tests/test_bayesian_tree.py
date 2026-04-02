"""Test suite for the Bayesian Binomial Tree pricing model."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.bayesian_tree import _crr_tree, greeks, price
from src.models.black_scholes import price as bs_price


# ---------------------------------------------------------------------------
# Common test parameters
# ---------------------------------------------------------------------------

PARAMS = dict(S=100.0, K=100.0, T=0.5, r=0.05, sigma=0.25, q=0.0)


# ---------------------------------------------------------------------------
# 1. CRR tree convergence to Black-Scholes
# ---------------------------------------------------------------------------

class TestCRRConvergence:
    """With many steps and fixed sigma the CRR tree should converge to BS."""

    def test_call_converges_to_bs(self):
        tree_px = _crr_tree(**PARAMS, option_type="call", steps=500)
        bs_px = bs_price(**PARAMS, option_type="call").price
        # Within 0.5% of BS
        assert abs(tree_px - bs_px) / bs_px < 0.005, (
            f"CRR call {tree_px:.4f} vs BS {bs_px:.4f}"
        )

    def test_put_converges_to_bs(self):
        tree_px = _crr_tree(**PARAMS, option_type="put", steps=500)
        bs_px = bs_price(**PARAMS, option_type="put").price
        assert abs(tree_px - bs_px) / bs_px < 0.005, (
            f"CRR put {tree_px:.4f} vs BS {bs_px:.4f}"
        )


# ---------------------------------------------------------------------------
# 2. Posterior mean near BS with tight prior
# ---------------------------------------------------------------------------

class TestPosteriorMean:
    """With a very tight LogNormal prior the posterior mean should be close to BS."""

    def test_tight_prior_call(self):
        # Supply a vol_history tightly clustered around sigma so the
        # fitted LogNormal prior is very narrow.
        np.random.seed(42)
        sigma = PARAMS["sigma"]
        tight_hist = [sigma * 0.99, sigma, sigma * 1.01, sigma * 0.995, sigma * 1.005]
        result = price(
            **PARAMS,
            option_type="call",
            steps=100,
            n_samples=2000,
            vol_history=tight_hist,
        )
        bs_px = bs_price(**PARAMS, option_type="call").price
        assert abs(result.price - bs_px) / bs_px < 0.03, (
            f"Posterior mean {result.price:.4f} vs BS {bs_px:.4f}"
        )

    def test_tight_prior_put(self):
        np.random.seed(42)
        sigma = PARAMS["sigma"]
        tight_hist = [sigma * 0.99, sigma, sigma * 1.01, sigma * 0.995, sigma * 1.005]
        result = price(
            **PARAMS,
            option_type="put",
            steps=100,
            n_samples=2000,
            vol_history=tight_hist,
        )
        bs_px = bs_price(**PARAMS, option_type="put").price
        assert abs(result.price - bs_px) / bs_px < 0.03, (
            f"Posterior mean {result.price:.4f} vs BS {bs_px:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. CI width: wider prior -> wider CI
# ---------------------------------------------------------------------------

class TestCIWidth:
    """A wider prior on sigma should produce a wider confidence interval."""

    def test_wider_prior_gives_wider_ci(self):
        np.random.seed(123)

        # Tight prior: shape=0.1 centred on sigma
        from scipy.stats import lognorm

        sigma = PARAMS["sigma"]
        n_samples = 1000
        steps = 100

        # Tight prior
        tight_samples = lognorm.rvs(0.1, loc=0, scale=sigma, size=n_samples)
        tight_samples = np.clip(tight_samples, 1e-6, 5.0)
        tight_prices = np.array([
            _crr_tree(PARAMS["S"], PARAMS["K"], PARAMS["T"], PARAMS["r"],
                      s, PARAMS["q"], "call", steps)
            for s in tight_samples
        ])
        tight_width = np.percentile(tight_prices, 97.5) - np.percentile(tight_prices, 2.5)

        # Wide prior
        wide_samples = lognorm.rvs(0.5, loc=0, scale=sigma, size=n_samples)
        wide_samples = np.clip(wide_samples, 1e-6, 5.0)
        wide_prices = np.array([
            _crr_tree(PARAMS["S"], PARAMS["K"], PARAMS["T"], PARAMS["r"],
                      s, PARAMS["q"], "call", steps)
            for s in wide_samples
        ])
        wide_width = np.percentile(wide_prices, 97.5) - np.percentile(wide_prices, 2.5)

        assert wide_width > tight_width, (
            f"Wide CI {wide_width:.4f} should exceed tight CI {tight_width:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """T=0 returns intrinsic, works for both call and put."""

    def test_call_at_expiry_itm(self):
        px = _crr_tree(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.25, q=0.0,
                       option_type="call", steps=100)
        assert px == pytest.approx(10.0)

    def test_call_at_expiry_otm(self):
        px = _crr_tree(S=90.0, K=100.0, T=0.0, r=0.05, sigma=0.25, q=0.0,
                       option_type="call", steps=100)
        assert px == pytest.approx(0.0)

    def test_put_at_expiry_itm(self):
        px = _crr_tree(S=90.0, K=100.0, T=0.0, r=0.05, sigma=0.25, q=0.0,
                       option_type="put", steps=100)
        assert px == pytest.approx(10.0)

    def test_put_at_expiry_otm(self):
        px = _crr_tree(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.25, q=0.0,
                       option_type="put", steps=100)
        assert px == pytest.approx(0.0)

    def test_price_function_at_expiry(self):
        """The public price() function should also handle T=0."""
        result = price(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.25, q=0.0,
                       option_type="call", steps=100, n_samples=50)
        assert result.price == pytest.approx(10.0)

    def test_zero_volatility(self):
        """sigma=0 gives deterministic discounted payoff."""
        import math
        px = _crr_tree(S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.0, q=0.0,
                       option_type="call", steps=100)
        expected = max(110.0 - 100.0, 0.0) * math.exp(-0.05)
        assert px == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Greeks finite differences
# ---------------------------------------------------------------------------

class TestGreeks:
    """Sanity checks on finite-difference Greeks."""

    def test_call_delta_in_range(self):
        g = greeks(**PARAMS, option_type="call", steps=100)
        assert 0.0 <= g.delta <= 1.0, f"Call delta {g.delta} out of [0,1]"

    def test_put_delta_in_range(self):
        g = greeks(**PARAMS, option_type="put", steps=100)
        assert -1.0 <= g.delta <= 0.0, f"Put delta {g.delta} out of [-1,0]"

    def test_gamma_non_negative(self):
        g = greeks(**PARAMS, option_type="call", steps=100)
        assert g.gamma >= -1e-8, f"Gamma {g.gamma} is negative"

    def test_vega_non_negative(self):
        g = greeks(**PARAMS, option_type="call", steps=100)
        assert g.vega >= -1e-8, f"Vega {g.vega} is negative"

    def test_model_name(self):
        g = greeks(**PARAMS, option_type="call", steps=100)
        assert g.model_name == "Bayesian Tree"


# ---------------------------------------------------------------------------
# 6. PricingResult metadata and fields
# ---------------------------------------------------------------------------

class TestPricingResultFields:
    """Ensure the returned PricingResult is well-formed."""

    def test_model_name(self):
        np.random.seed(0)
        result = price(**PARAMS, option_type="call", steps=50, n_samples=100)
        assert result.model_name == "Bayesian Tree"

    def test_confidence_interval_exists(self):
        np.random.seed(0)
        result = price(**PARAMS, option_type="call", steps=50, n_samples=100)
        assert result.confidence_interval is not None
        lo, hi = result.confidence_interval
        assert lo <= result.price <= hi

    def test_posterior_samples_shape(self):
        np.random.seed(0)
        n = 200
        result = price(**PARAMS, option_type="call", steps=50, n_samples=n)
        assert result.posterior_samples is not None
        assert len(result.posterior_samples) == n

    def test_metadata_keys(self):
        np.random.seed(0)
        result = price(**PARAMS, option_type="call", steps=50, n_samples=100)
        assert "steps" in result.metadata
        assert "n_samples" in result.metadata
        assert result.metadata["steps"] == 50
        assert result.metadata["n_samples"] == 100

    def test_vol_history_prior(self):
        """When vol_history is supplied, the prior should be fit to it."""
        np.random.seed(0)
        vol_hist = [0.20, 0.22, 0.25, 0.28, 0.30]
        result = price(**PARAMS, option_type="call", steps=50, n_samples=200,
                       vol_history=vol_hist)
        assert result.price > 0
        assert result.confidence_interval is not None
