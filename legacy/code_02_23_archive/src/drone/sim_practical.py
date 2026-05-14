from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from .dynamics import quad_dynamics_extended, rk4_step
from .practical_controller import PathSpeedController


@dataclass
class SimConfig:
    dt: float = 0.01
    T: float = 25.0
    L: float = 5.0


def simulate(
    ctrl: PathSpeedController,
    x0: np.ndarray,
    cfg: SimConfig,
) -> Dict[str, Any]:
    dt = float(cfg.dt)
    steps = int(round(cfg.T / dt)) + 1

    x = np.asarray(x0, dtype=float).copy()
    xs = np.zeros((steps, x.size), dtype=float)
    Us = np.zeros((steps, 4), dtype=float)
    ts = np.zeros(steps, dtype=float)
    logs = []

    for k in range(steps):
        t = k * dt
        ts[k] = t
        xs[k] = x

        U, info = ctrl.step(x=x, dt=dt)
        Us[k] = U
        logs.append(info)

        x = rk4_step(lambda xx, uu: quad_dynamics_extended(xx, uu, L=cfg.L), x, U, dt)

    return {"t": ts, "x": xs, "U": Us, "log": logs}
