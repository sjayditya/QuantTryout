"""Utility helpers for managing cached model weights."""

from __future__ import annotations

from pathlib import Path


_MODELS_DIR = Path("data/models")


def get_weights_path(symbol: str) -> Path:
    """Return the path where ensemble weights for *symbol* are stored.

    Returns ``data/models/{symbol}_ensemble.pt``.
    """
    return _MODELS_DIR / f"{symbol}_ensemble.pt"


def weights_exist(symbol: str) -> bool:
    """Return *True* if a pre-trained weights file exists for *symbol*."""
    return get_weights_path(symbol).is_file()
