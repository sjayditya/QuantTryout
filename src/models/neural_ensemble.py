"""Neural Network Ensemble pricing model for European options.

An ensemble of LSTM networks is trained on Heston-calibrated synthetic data.
Predictions are blended with Black-Scholes prices using a confidence-based
sigmoid weighting scheme.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import GreeksResult, PricingResult
from src.utils.config import (
    NN_BATCH_SIZE,
    NN_BLEND_ALPHA,
    NN_BLEND_BETA,
    NN_DENSE_DIM,
    NN_ENSEMBLE_SIZE,
    NN_EPOCHS,
    NN_HIDDEN_DIM,
    NN_INPUT_DIM,
    NN_LEARNING_RATE,
    NN_LSTM_LAYERS,
    NN_MAX_TRAINING_TIME,
)
from src.utils.cache import get_weights_path


# ---------------------------------------------------------------------------
# LSTM pricing network
# ---------------------------------------------------------------------------

class OptionPricingLSTM(nn.Module):
    """Single LSTM-based option pricing network.

    Architecture
    ------------
    LSTM(input_dim=7, hidden_size=64, num_layers=2, batch_first=True)
        -> Linear(64, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(
        self,
        input_dim: int = NN_INPUT_DIM,
        hidden_dim: int = NN_HIDDEN_DIM,
        dense_dim: int = NN_DENSE_DIM,
        num_layers: int = NN_LSTM_LAYERS,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, input_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, 1)``.
        """
        # lstm_out shape: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last time-step's hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

class NeuralEnsemble:
    """Ensemble of ``n_models`` LSTM pricing networks."""

    def __init__(
        self, n_models: int = NN_ENSEMBLE_SIZE, device: str = "cpu"
    ) -> None:
        self.n_models = n_models
        self.device = device
        self.models: list[OptionPricingLSTM] = [
            OptionPricingLSTM().to(device) for _ in range(n_models)
        ]

    # ---- training ----------------------------------------------------------

    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = NN_EPOCHS,
        lr: float = NN_LEARNING_RATE,
        batch_size: int = NN_BATCH_SIZE,
        progress_callback=None,
    ) -> None:
        """Train all ensemble members on the same data with different seeds.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, seq_len, 7)``.
        y : np.ndarray
            Shape ``(n_samples, 1)``.
        epochs : int
            Number of training epochs per model.
        lr : float
            Learning rate for Adam optimiser.
        batch_size : int
            Mini-batch size.
        progress_callback : callable | None
            If provided, called with ``(current_step, total_steps)`` after
            each epoch.
        """
        total_steps = self.n_models * epochs
        current_step = 0
        start_time = time.time()

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(X_t, y_t)

        for i, model in enumerate(self.models):
            torch.manual_seed(42 + i)
            model.train()
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for _epoch in range(epochs):
                for X_batch, y_batch in loader:
                    optimiser.zero_grad()
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    optimiser.step()

                current_step += 1
                if progress_callback is not None:
                    progress_callback(current_step, total_steps)

                # Enforce time limit
                if time.time() - start_time > NN_MAX_TRAINING_TIME:
                    return

    # ---- inference ---------------------------------------------------------

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ensemble mean and std of predictions.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, seq_len, 7)``.

        Returns
        -------
        mean_predictions : np.ndarray
            Shape ``(n_samples, 1)``.
        std_predictions : np.ndarray
            Shape ``(n_samples, 1)``.
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        all_preds: list[np.ndarray] = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_t).cpu().numpy()
            all_preds.append(pred)

        stacked = np.stack(all_preds, axis=0)  # (n_models, n_samples, 1)
        mean_preds = np.mean(stacked, axis=0)
        std_preds = np.std(stacked, axis=0)
        return mean_preds, std_preds

    # ---- persistence -------------------------------------------------------

    def save_weights(self, path: str) -> None:
        """Save all model state dicts as a single dict to *path*."""
        state = {
            f"model_{i}": model.state_dict()
            for i, model in enumerate(self.models)
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load_weights(self, path: str) -> bool:
        """Load weights from *path*.  Returns *True* on success, *False* if
        the file does not exist."""
        p = Path(path)
        if not p.is_file():
            return False
        state = torch.load(p, map_location=self.device, weights_only=True)
        for i, model in enumerate(self.models):
            key = f"model_{i}"
            if key in state:
                model.load_state_dict(state[key])
        return True


# ---------------------------------------------------------------------------
# Helper: build feature vector
# ---------------------------------------------------------------------------

def _build_features(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float,
    historical_data=None,
) -> np.ndarray:
    """Return a (1, 1, 7) feature array for a single prediction."""
    rsi = 50.0
    vol_ratio = 1.0

    if historical_data is not None:
        try:
            import pandas as pd
            from src.utils.math_utils import compute_rsi, compute_vol_ratio

            if isinstance(historical_data, pd.DataFrame):
                close = historical_data["Close"]
            elif isinstance(historical_data, pd.Series):
                close = historical_data
            else:
                close = None

            if close is not None and len(close) > 0:
                rsi = compute_rsi(close)
                vol_ratio = compute_vol_ratio(close)
        except Exception:
            pass

    features = np.array(
        [S / K, T, sigma, r, q, rsi, vol_ratio], dtype=np.float32
    )
    return features.reshape(1, 1, 7)


# ---------------------------------------------------------------------------
# Public: price
# ---------------------------------------------------------------------------

def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
    historical_data=None,
    symbol: str = "UNKNOWN",
    device: str = "cpu",
) -> PricingResult:
    """Price a European option using the Neural Ensemble model.

    Loads pre-trained weights if available; otherwise trains on synthetic data
    generated via Heston calibration.  The final price is a confidence-
    weighted blend of the ensemble mean and the Black-Scholes analytical
    price.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type
        Standard option parameters.
    historical_data : pd.DataFrame | pd.Series | None
        Historical price data used for RSI / vol-ratio features and Heston
        calibration.
    symbol : str
        Ticker symbol, used as a cache key for weights.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    PricingResult
    """
    ensemble = NeuralEnsemble(n_models=NN_ENSEMBLE_SIZE, device=device)

    weights_path = str(get_weights_path(symbol))
    loaded = ensemble.load_weights(weights_path)

    if not loaded:
        # Train from scratch using Heston-calibrated synthetic data
        try:
            from src.models.heston import calibrate_heston, generate_training_data

            heston_params = calibrate_heston(historical_data)
            X_train, y_train = generate_training_data(
                S=S, K=K, T=T, r=r, q=q,
                heston_params=heston_params,
                option_type=option_type,
            )
        except Exception:
            # Fallback: simple synthetic data around the current params
            rng = np.random.default_rng(42)
            n = 5000
            moneyness = rng.uniform(0.8, 1.2, n)
            t_arr = rng.uniform(0.01, 2.0, n)
            sig_arr = rng.uniform(0.1, 0.6, n)
            r_arr = np.full(n, r)
            q_arr = np.full(n, q)
            rsi_arr = rng.uniform(30, 70, n)
            vr_arr = rng.uniform(0.5, 1.5, n)
            X_train = np.stack(
                [moneyness, t_arr, sig_arr, r_arr, q_arr, rsi_arr, vr_arr],
                axis=-1,
            ).reshape(n, 1, 7).astype(np.float32)

            # Use BS as labels
            from src.models import black_scholes
            y_train = np.array(
                [
                    black_scholes.price(
                        S * moneyness[j],
                        S,
                        t_arr[j],
                        r,
                        sig_arr[j],
                        q,
                        option_type,
                    ).price
                    for j in range(n)
                ],
                dtype=np.float32,
            ).reshape(n, 1)

        # Train with optional streamlit progress bar
        progress_callback = None
        try:
            import streamlit as st

            bar = st.progress(0, text="Training neural ensemble...")

            def _progress(cur: int, total: int) -> None:
                bar.progress(cur / total, text=f"Training neural ensemble... {cur}/{total}")

            progress_callback = _progress
        except Exception:
            pass

        ensemble.train_ensemble(
            X_train,
            y_train,
            epochs=NN_EPOCHS,
            lr=NN_LEARNING_RATE,
            batch_size=NN_BATCH_SIZE,
            progress_callback=progress_callback,
        )
        ensemble.save_weights(weights_path)

    # Build feature vector and predict
    X_input = _build_features(S, K, T, sigma, r, q, historical_data)
    mean_preds, std_preds = ensemble.predict(X_input)
    nn_mean = float(mean_preds[0, 0])
    sigma_ensemble = float(std_preds[0, 0])

    # Individual predictions for metadata
    X_t = torch.tensor(X_input, dtype=torch.float32, device=device)
    individual: list[float] = []
    for model in ensemble.models:
        model.eval()
        with torch.no_grad():
            p = model(X_t).cpu().item()
        individual.append(p)

    # Blending with Black-Scholes
    from src.models import black_scholes

    bs_result = black_scholes.price(S, K, T, r, sigma, q, option_type)
    bs_price = bs_result.price

    # w = sigmoid(-alpha * sigma_ensemble + beta)
    w = 1.0 / (1.0 + np.exp(-(-NN_BLEND_ALPHA * sigma_ensemble + NN_BLEND_BETA)))
    blended_price = float(w * nn_mean + (1.0 - w) * bs_price)

    # Confidence score: high agreement -> high confidence
    confidence_score = 1.0 - min(sigma_ensemble / (abs(nn_mean) + 1e-8), 1.0)

    return PricingResult(
        price=blended_price,
        model_name="Neural Ensemble",
        ensemble_prices=individual,
        confidence_score=float(confidence_score),
        metadata={
            "nn_mean": nn_mean,
            "bs_price": bs_price,
            "blend_weight": float(w),
            "device": device,
        },
    )


# ---------------------------------------------------------------------------
# Public: greeks via finite differences
# ---------------------------------------------------------------------------

def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
    historical_data=None,
    symbol: str = "UNKNOWN",
    device: str = "cpu",
) -> GreeksResult:
    """Compute finite-difference Greeks from the Neural Ensemble model.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type
        Standard option parameters.
    historical_data, symbol, device
        Passed through to :func:`price`.

    Returns
    -------
    GreeksResult
    """
    _kw = dict(
        historical_data=historical_data, symbol=symbol, device=device,
    )

    p0 = price(S, K, T, r, sigma, q, option_type, **_kw).price

    # --- Delta ---------------------------------------------------------------
    h_s = 0.01 * S
    p_up = price(S + h_s, K, T, r, sigma, q, option_type, **_kw).price
    p_dn = price(S - h_s, K, T, r, sigma, q, option_type, **_kw).price
    delta = (p_up - p_dn) / (2.0 * h_s)

    # --- Gamma ---------------------------------------------------------------
    gamma = (p_up - 2.0 * p0 + p_dn) / (h_s ** 2)

    # --- Theta ---------------------------------------------------------------
    h_t = 1.0 / 365.0
    T_shifted = max(T - h_t, 0.0)
    p_t = price(S, K, T_shifted, r, sigma, q, option_type, **_kw).price
    theta = (p_t - p0) / h_t

    # --- Vega ----------------------------------------------------------------
    h_v = 0.01
    p_v_up = price(S, K, T, r, sigma + h_v, q, option_type, **_kw).price
    p_v_dn = price(S, K, T, r, max(sigma - h_v, 1e-8), q, option_type, **_kw).price
    h_v_dn = sigma - max(sigma - h_v, 1e-8)
    vega = (p_v_up - p_v_dn) / (h_v + h_v_dn) * 0.01  # per 1% vol move

    # --- Rho -----------------------------------------------------------------
    h_r = 0.01
    p_r_up = price(S, K, T, r + h_r, sigma, q, option_type, **_kw).price
    p_r_dn = price(S, K, T, r - h_r, sigma, q, option_type, **_kw).price
    rho = (p_r_up - p_r_dn) / (2.0 * h_r)

    return GreeksResult(
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
        rho=float(rho),
        model_name="Neural Ensemble",
    )
