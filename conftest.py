"""Root conftest.py — ensures the project root is on sys.path for imports."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so `from src.…` imports work in tests
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
