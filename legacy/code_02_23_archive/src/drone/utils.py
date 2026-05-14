from __future__ import annotations

import numpy as np


def wrap_pi(angle: float) -> float:
    """Map angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def sat_tanh(x: float | np.ndarray, L: float) -> float | np.ndarray:
    """Smooth saturation with level L: sat_L(x) = L * tanh(x / L)."""
    L = float(max(L, 1e-9))
    return L * np.tanh(np.asarray(x, dtype=float) / L)


def sat_vec_tanh(x: np.ndarray, L: float) -> np.ndarray:
    return np.asarray(sat_tanh(x, L), dtype=float)
