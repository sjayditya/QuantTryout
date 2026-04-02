"""Comprehensive test suite for the Black-Scholes pricing model."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from src.models.black_scholes import greeks, implied_volatility, price


# ---------------------------------------------------------------------------
# 1. Hull textbook reference prices
# ---------------------------------------------------------------------------

class TestHullTextbook:
    """Hull (Options, Futures, …) canonical example: S=42, K=40, T=0.5, r=0.10, sigma=0.20, q=0."""

    PARAMS = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20, q=0.0)

    def test_call_price(self):
        result = price(**self.PARAMS, option_type="call")
        assert result.model_name == "Black-Scholes"
        assert abs(result.price - 4.76) < 0.02

    def test_put_price(self):
        result = price(**self.PARAMS, option_type="put")
        assert result.model_name == "Black-Scholes"
        assert abs(result.price - 0.81) < 0.02


# ---------------------------------------------------------------------------
# 2. Put-call parity
# ---------------------------------------------------------------------------

class TestPutCallParity:
    """C - P = S*exp(-qT) - K*exp(-rT)  for European options."""

    RANDOM_SETS = [
        dict(S=100, K=100, T=1.0, r=0.05, sigma=0.25, q=0.0),
        dict(S=50, K=55, T=0.25, r=0.08, sigma=0.30, q=0.02),
        dict(S=200, K=180, T=2.0, r=0.03, sigma=0.15, q=0.01),
        dict(S=80, K=90, T=0.5, r=0.10, sigma=0.40, q=0.05),
        dict(S=150, K=150, T=0.75, r=0.06, sigma=0.20, q=0.03),
        dict(S=120, K=110, T=1.5, r=0.04, sigma=0.35, q=0.0),
    ]

    @pytest.mark.parametrize("params", RANDOM_SETS)
    def test_parity(self, params):
        call = price(**params, option_type="call").price
        put = price(**params, option_type="put").price
        S, K, T, r, q = params["S"], params["K"], params["T"], params["r"], params["q"]

        # C - P should equal S*exp(-qT) - K*exp(-rT)
        lhs = call - put
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

        assert abs(lhs - rhs) < 1e-8, f"Put-call parity violated: {lhs} != {rhs}"


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """T=0, sigma=0, deep ITM, deep OTM."""

    def test_call_at_expiry_itm(self):
        result = price(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.20, q=0.0, option_type="call")
        assert result.price == pytest.approx(10.0)

    def test_put_at_expiry_itm(self):
        result = price(S=90.0, K=100.0, T=0.0, r=0.05, sigma=0.20, q=0.0, option_type="put")
        assert result.price == pytest.approx(10.0)

    def test_call_at_expiry_otm(self):
        result = price(S=90.0, K=100.0, T=0.0, r=0.05, sigma=0.20, q=0.0, option_type="call")
        assert result.price == pytest.approx(0.0)

    def test_put_at_expiry_otm(self):
        result = price(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.20, q=0.0, option_type="put")
        assert result.price == pytest.approx(0.0)

    def test_zero_vol_call_itm(self):
        # With sigma=0 the option has deterministic payoff
        result = price(S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.0, q=0.0, option_type="call")
        # Forward = 110, deterministic payoff = max(110 - 100, 0) * exp(-0.05)
        expected = max(110.0 - 100.0, 0.0) * math.exp(-0.05)
        assert result.price == pytest.approx(expected, abs=1e-6)

    def test_zero_vol_put_itm(self):
        result = price(S=90.0, K=100.0, T=1.0, r=0.05, sigma=0.0, q=0.0, option_type="put")
        expected = max(100.0 - 90.0, 0.0) * math.exp(-0.05)
        assert result.price == pytest.approx(expected, abs=1e-6)

    def test_deep_itm_call(self):
        # Deep ITM call: price should be close to S*exp(-qT) - K*exp(-rT)
        result = price(S=200.0, K=50.0, T=0.5, r=0.05, sigma=0.20, q=0.0, option_type="call")
        lower_bound = 200.0 - 50.0 * math.exp(-0.05 * 0.5)
        assert result.price >= lower_bound - 1e-6

    def test_deep_otm_call(self):
        # Deep OTM call: price should be very small
        result = price(S=50.0, K=200.0, T=0.5, r=0.05, sigma=0.20, q=0.0, option_type="call")
        assert result.price < 0.01

    def test_negative_time(self):
        # T < 0 treated same as T = 0
        result = price(S=110.0, K=100.0, T=-0.1, r=0.05, sigma=0.20, q=0.0, option_type="call")
        assert result.price == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 4. Greeks sanity checks
# ---------------------------------------------------------------------------

class TestGreeksSanity:
    """Basic constraints that must always hold."""

    PARAMS = dict(S=100.0, K=100.0, T=0.5, r=0.05, sigma=0.25, q=0.0)

    def test_call_delta_range(self):
        g = greeks(**self.PARAMS, option_type="call")
        assert 0.0 <= g.delta <= 1.0

    def test_put_delta_range(self):
        g = greeks(**self.PARAMS, option_type="put")
        assert -1.0 <= g.delta <= 0.0

    def test_gamma_non_negative(self):
        g = greeks(**self.PARAMS, option_type="call")
        assert g.gamma >= 0.0

    def test_vega_non_negative(self):
        g = greeks(**self.PARAMS, option_type="call")
        assert g.vega >= 0.0

    def test_call_rho_positive(self):
        g = greeks(**self.PARAMS, option_type="call")
        assert g.rho >= 0.0

    def test_put_rho_negative(self):
        g = greeks(**self.PARAMS, option_type="put")
        assert g.rho <= 0.0

    def test_greeks_at_expiry(self):
        g = greeks(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.25, q=0.0, option_type="call")
        assert g.delta == 1.0
        assert g.gamma == 0.0
        assert g.theta == 0.0
        assert g.vega == 0.0

    def test_greeks_model_name(self):
        g = greeks(**self.PARAMS, option_type="call")
        assert g.model_name == "Black-Scholes"


# ---------------------------------------------------------------------------
# 5. ATM delta approximation
# ---------------------------------------------------------------------------

class TestATMDelta:
    """ATM call delta is approximately 0.5 (exact when q=0 and r is small)."""

    def test_atm_call_delta_near_half(self):
        # Use short maturity and low r so drift term is small and delta ~ 0.5
        g = greeks(S=100.0, K=100.0, T=0.25, r=0.02, sigma=0.20, q=0.0, option_type="call")
        assert abs(g.delta - 0.5) < 0.1

    def test_atm_put_delta_near_neg_half(self):
        g = greeks(S=100.0, K=100.0, T=0.25, r=0.02, sigma=0.20, q=0.0, option_type="put")
        assert abs(g.delta + 0.5) < 0.1


# ---------------------------------------------------------------------------
# 6. Implied volatility round-trip
# ---------------------------------------------------------------------------

class TestImpliedVolatility:
    """Compute BS price with known sigma, recover sigma via IV."""

    CASES = [
        dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type="call"),
        dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type="put"),
        dict(S=50.0, K=55.0, T=0.25, r=0.08, q=0.02, sigma=0.35, option_type="call"),
        dict(S=200.0, K=180.0, T=2.0, r=0.03, q=0.01, sigma=0.15, option_type="put"),
        dict(S=100.0, K=120.0, T=0.5, r=0.05, q=0.0, sigma=0.50, option_type="call"),
    ]

    @pytest.mark.parametrize("case", CASES)
    def test_round_trip(self, case):
        # Make a copy so parametrize data is not mutated across runs
        case = dict(case)
        sigma_true = case.pop("sigma")
        opt_type = case.pop("option_type")

        # Price with known volatility
        market = price(**case, sigma=sigma_true, option_type=opt_type).price

        # Recover implied volatility
        iv = implied_volatility(
            market_price=market,
            **case,
            option_type=opt_type,
        )

        assert abs(iv - sigma_true) < 1e-3, (
            f"IV round-trip failed: expected {sigma_true}, got {iv}"
        )
