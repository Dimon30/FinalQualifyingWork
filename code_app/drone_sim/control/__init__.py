"""Регуляторы согласованного управления."""
from drone_sim.control.common import HighGainParams, DerivativeObserver4
from drone_sim.control.path_following import (
    Ch4PathController,
    W_mat,
    W_inv,
    b_mat,
    _safe_inv4,
)

__all__ = [
    "HighGainParams",
    "DerivativeObserver4",
    "Ch4PathController",
    "W_mat",
    "W_inv",
    "b_mat",
    "_safe_inv4",
]
