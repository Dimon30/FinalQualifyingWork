from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass(frozen=True)
class TrajPoint:
    p: np.ndarray   # (3,)
    v: np.ndarray   # (3,)
    a: np.ndarray   # (3,)


def spiral_time_traj(r: float = 0.5) -> Callable[[float], TrajPoint]:
    """Траектория из примера (глава 3): x*=r cos(t), y*=r sin(t), z*=t."""
    def f(t: float) -> TrajPoint:
        p = np.array([r*np.cos(t), r*np.sin(t), t], dtype=float)
        v = np.array([-r*np.sin(t), r*np.cos(t), 1.0], dtype=float)
        a = np.array([-r*np.cos(t), -r*np.sin(t), 0.0], dtype=float)
        return TrajPoint(p=p, v=v, a=a)
    return f


def line_s_traj() -> Callable[[float], TrajPoint]:
    """Прямая из примера (глава 4): x=s, y=s, z=s. Здесь берём s=t."""
    def f(t: float) -> TrajPoint:
        p = np.array([t, t, t], dtype=float)
        v = np.array([1.0, 1.0, 1.0], dtype=float)
        a = np.zeros(3, dtype=float)
        return TrajPoint(p=p, v=v, a=a)
    return f


def nearest_point_line_xyz(x: float, y: float, z: float) -> Tuple[float, np.ndarray]:
    """Для линии x=y=z=s ближайшая точка: s* = (x+y+z)/3."""
    s = (x + y + z) / 3.0
    p = np.array([s, s, s], dtype=float)
    return s, p


def spiral_nearest_observer_step(
    zeta: float,
    x: float,
    y: float,
    z: float,
    r: float = 3.0,
    gamma: float = 1.0,
    dt: float = 0.01,
) -> float:
    """Динамический наблюдатель ближайшей точки для спирали x=r cos(zeta), y=r sin(zeta), z=zeta.

    Используется формула вида:
      zeta_dot = -gamma * rho * H(zeta,x,y,z),
      H = r sin(zeta) x - r cos(zeta) y - z + zeta,
      rho = sign(1 + r cos(zeta) x + r sin(zeta) y)
    (см. диссертацию, пример со спиралью).
    """
    H = r*np.sin(zeta)*x - r*np.cos(zeta)*y - z + zeta
    rho = np.sign(1.0 + r*np.cos(zeta)*x + r*np.sin(zeta)*y)
    return zeta + (-gamma * rho * H) * dt
