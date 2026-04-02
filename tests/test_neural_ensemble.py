"""Tests for the Neural Network Ensemble pricing model.

All tests run on CPU with small datasets for speed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.neural_ensemble import NeuralEnsemble, OptionPricingLSTM, price


# ---------------------------------------------------------------------------
# 1. OptionPricingLSTM forward pass
# ---------------------------------------------------------------------------


def test_lstm_forward_shape():
    """Input (batch=4, seq=1, features=7) should produce output (4, 1)."""
    model = OptionPricingLSTM()
    x = torch.randn(4, 1, 7)
    out = model(x)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Ensemble produces 5 distinct predictions
# ---------------------------------------------------------------------------


def test_ensemble_distinct_predictions():
    """After brief training with different seeds, the 5 networks should
    not all produce identical predictions."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 1, 7)).astype(np.float32)
    y = rng.standard_normal((1000, 1)).astype(np.float32)

    ensemble = NeuralEnsemble(n_models=5, device="cpu")
    ensemble.train_ensemble(X, y, epochs=5, lr=1e-3, batch_size=256)

    X_test = rng.standard_normal((10, 1, 7)).astype(np.float32)

    # Collect individual predictions
    X_t = torch.tensor(X_test, dtype=torch.float32)
    preds = []
    for model in ensemble.models:
        model.eval()
        with torch.no_grad():
            p = model(X_t).numpy()
        preds.append(p)

    stacked = np.stack(preds, axis=0)  # (5, 10, 1)
    # At least some variance across models for at least one sample
    std_across_models = np.std(stacked, axis=0)  # (10, 1)
    assert np.max(std_across_models) > 1e-6, (
        "All 5 models produced identical predictions — seeds may not be working"
    )


# ---------------------------------------------------------------------------
# 3. Ensemble mean reasonable on constant-label data
# ---------------------------------------------------------------------------


def test_ensemble_mean_reasonable():
    """When labels are all 10.0, ensemble mean should be within 50% of 10.0
    after brief training."""
    rng = np.random.default_rng(99)
    n = 1000
    X = rng.standard_normal((n, 1, 7)).astype(np.float32)
    y = np.full((n, 1), 10.0, dtype=np.float32)

    ensemble = NeuralEnsemble(n_models=5, device="cpu")
    ensemble.train_ensemble(X, y, epochs=5, lr=1e-3, batch_size=256)

    X_test = rng.standard_normal((20, 1, 7)).astype(np.float32)
    mean_preds, _std_preds = ensemble.predict(X_test)
    overall_mean = float(np.mean(mean_preds))

    assert 5.0 <= overall_mean <= 15.0, (
        f"Ensemble mean {overall_mean:.2f} is not within 50% of 10.0"
    )


# ---------------------------------------------------------------------------
# 4. Save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip():
    """Predictions after save+load should match the original ensemble."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((500, 1, 7)).astype(np.float32)
    y = rng.standard_normal((500, 1)).astype(np.float32)

    ensemble = NeuralEnsemble(n_models=5, device="cpu")
    ensemble.train_ensemble(X, y, epochs=3, lr=1e-3, batch_size=256)

    X_test = rng.standard_normal((5, 1, 7)).astype(np.float32)
    mean_before, std_before = ensemble.predict(X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_weights.pt")
        ensemble.save_weights(path)

        ensemble2 = NeuralEnsemble(n_models=5, device="cpu")
        loaded = ensemble2.load_weights(path)
        assert loaded is True, "load_weights should return True"

        mean_after, std_after = ensemble2.predict(X_test)
        np.testing.assert_allclose(mean_before, mean_after, atol=1e-5)
        np.testing.assert_allclose(std_before, std_after, atol=1e-5)

    # Loading from a non-existent path should return False
    ensemble3 = NeuralEnsemble(n_models=5, device="cpu")
    assert ensemble3.load_weights("/tmp/nonexistent_weights.pt") is False


# ---------------------------------------------------------------------------
# 5. Blending toward BS when ensemble disagreement is high
# ---------------------------------------------------------------------------


def test_blending_toward_bs_on_high_disagreement():
    """When ensemble members disagree strongly, the blended price should be
    close to the Black-Scholes price (i.e. blend weight w -> 0)."""
    from src.models import black_scholes

    S, K, T, r, sigma, q = 100.0, 100.0, 0.5, 0.07, 0.25, 0.01

    bs_price = black_scholes.price(S, K, T, r, sigma, q, "call").price

    # Build an ensemble with wildly different weights (untrained = random)
    ensemble = NeuralEnsemble(n_models=5, device="cpu")
    # Do NOT train — random weights will give high disagreement

    # Monkey-patch the ensemble into the price function by calling it directly
    # We test the blending formula: high sigma_ensemble -> w near 0 -> blended ~ bs
    from src.models.neural_ensemble import _build_features, NN_BLEND_ALPHA, NN_BLEND_BETA

    X_input = _build_features(S, K, T, sigma, r, q)
    mean_preds, std_preds = ensemble.predict(X_input)
    sigma_ensemble = float(std_preds[0, 0])

    w = 1.0 / (1.0 + np.exp(-(-NN_BLEND_ALPHA * sigma_ensemble + NN_BLEND_BETA)))
    nn_mean = float(mean_preds[0, 0])
    blended = w * nn_mean + (1.0 - w) * bs_price

    # When sigma_ensemble is large, w should be small and blended near bs_price
    # With random weights, sigma should be non-trivial.
    # The key assertion: the blended price should be closer to BS than to nn_mean
    dist_to_bs = abs(blended - bs_price)
    dist_to_nn = abs(blended - nn_mean)

    # If w < 0.5, blended is closer to BS
    assert w < 0.5 or dist_to_bs <= dist_to_nn + 1e-6, (
        f"Blended price should lean toward BS when disagreement is high. "
        f"w={w:.4f}, blended={blended:.4f}, bs={bs_price:.4f}, nn={nn_mean:.4f}"
    )
