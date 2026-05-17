"""Геометрия пространственных кривых для согласованного управления (Гл. 4).

Кривая S задаётся параметрически p(s); геометрия определяет:
    α(s) — yaw касательной, β(s) — pitch касательной, ε(s) = dα/ds.
Ошибки в системе Френе: [s_coord, e1, e2] = Ry(β)ᵀ · Rz(α)ᵀ · (p - p_s).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class CurveGeom:
    """Геометрия кривой: p(s), t(s), yaw_star(s), beta(s), eps(s)."""
    p: Callable[[float], np.ndarray]
    t: Callable[[float], np.ndarray]
    yaw_star: Callable[[float], float]
    beta: Callable[[float], float]
    eps: Callable[[float], float]


def Rz(a: float) -> np.ndarray:
    """Матрица поворота вокруг оси Z на угол a."""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def Ry(b: float) -> np.ndarray:
    """Матрица поворота вокруг оси Y на угол b."""
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[ cb, 0.0, sb],
                     [0.0, 1.0, 0.0],
                     [-sb, 0.0, cb]], dtype=float)


def se_from_pose(
    p_xyz: np.ndarray, s: float, curve: CurveGeom
) -> Tuple[float, float, float]:
    """Координаты ошибки (s_local, e1, e2) в системе Френе (ур. 60)."""
    ps = curve.p(s)
    alpha = curve.yaw_star(s)
    beta_val = curve.beta(s)
    v = p_xyz - ps
    q = Ry(beta_val).T @ (Rz(alpha).T @ v)
    return float(q[0]), float(q[1]), float(q[2])


def line_xyz_curve() -> CurveGeom:
    """Прямая x=s, y=s, z=s (45° в 3D), ||t||=√3."""
    def p(s: float) -> np.ndarray:
        return np.array([s, s, s], dtype=float)

    def t(s: float) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0], dtype=float)

    def yaw_star(s: float) -> float:
        return float(np.arctan2(1.0, 1.0))

    def beta(s: float) -> float:
        return float(np.arctan2(1.0, np.sqrt(2.0)))

    def eps(s: float) -> float:
        return 0.0

    return CurveGeom(p=p, t=t, yaw_star=yaw_star, beta=beta, eps=eps)


def spiral_curve(r: float = 3.0) -> CurveGeom:
    """Спираль p(s) = [r·cos(s), r·sin(s), s], ||t||=√(r²+1)."""
    def p(s: float) -> np.ndarray:
        return np.array([r * np.cos(s), r * np.sin(s), s], dtype=float)

    def t(s: float) -> np.ndarray:
        return np.array([-r * np.sin(s), r * np.cos(s), 1.0], dtype=float)

    def yaw_star(s: float) -> float:
        tv = t(s)
        return float(np.arctan2(tv[1], tv[0]))

    def beta(s: float) -> float:
        tv = t(s)
        return float(np.arctan2(tv[2], np.sqrt(tv[0] ** 2 + tv[1] ** 2)))

    def eps(s: float) -> float:
        h = 1e-5
        a1 = yaw_star(s - h)
        a2 = yaw_star(s + h)
        da = np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))
        return float(da / (2 * h))

    return CurveGeom(p=p, t=t, yaw_star=yaw_star, beta=beta, eps=eps)


def nearest_point_line(p_xyz: np.ndarray) -> float:
    """Аналитическая ближайшая точка на прямой x=s,y=s,z=s: ς* = (x+y+z)/3."""
    x, y, z = float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])
    return (x + y + z) / 3.0


def spiral_nearest_observer_step(
    zeta: float,
    p_xyz: np.ndarray,
    r: float = 3.0,
    gamma: float = 1.0,
    dt: float = 0.01,
) -> float:
    """Один шаг наблюдателя ближайшей точки на спирали (Лемма 3): ζ̇ = -γ·ρ·H."""
    x, y, z = float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])
    H = r * np.sin(zeta) * x - r * np.cos(zeta) * y - z + zeta
    rho = np.sign(1.0 + r * np.cos(zeta) * x + r * np.sin(zeta) * y)
    return float(zeta + (-gamma * rho * H) * dt)
