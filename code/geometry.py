"""
Геометрия пространственных кривых для задачи согласованного управления (Глава 4).

Кривая S задаётся параметрически: p(s) = [φx(s), φy(s), φz(s)].
Геометрия определяет:
  - α(s)  — угол рысканья касательного вектора (yaw касательной), (уравнение 66)
  - β(s)  — угол тангажа касательного вектора (pitch касательной), (уравнение 67)
  - ε(s)  — кривизна проекции кривой на плоскость OXY, ε = dα/ds

Координаты ошибки (уравнение 60 диссертации):
  [s_coord, e1, e2] = Ry_β^T Rz_α^T [p - p_s]
где p_s = p(s) — ближайшая точка кривой.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class CurveGeom:
    """Геометрия гладкой пространственной кривой S.

    Атрибуты:
        p(s)        — вектор положения точки кривой
        t(s)        — касательный вектор (не обязательно единичный)
        yaw_star(s) — угол рысканья касательной α(s)
        beta(s)     — угол тангажа касательной β(s)
        eps(s)      — кривизна проекции OXY: ε = dα/ds
    """
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
    """Координаты ошибки в системе касательной/нормалей (уравнение 60).

    Возвращает (s_local, e1, e2):
        s_local — компонента вдоль касательной (≈0 при точном слежении)
        e1, e2  — боковые отклонения от кривой
    """
    ps = curve.p(s)
    alpha = curve.yaw_star(s)
    beta_val = curve.beta(s)
    v = p_xyz - ps
    q = Ry(beta_val).T @ (Rz(alpha).T @ v)
    return float(q[0]), float(q[1]), float(q[2])


# ---------------------------------------------------------------------------
# Прямолинейная траектория x=s, y=s, z=s
# ---------------------------------------------------------------------------

def line_xyz_curve() -> CurveGeom:
    """Кривая x=s, y=s, z=s (прямая линия под углом 45° в 3D)."""
    def p(s: float) -> np.ndarray:
        return np.array([s, s, s], dtype=float)

    def t(s: float) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0], dtype=float)

    def yaw_star(s: float) -> float:
        # Угол рысканья касательной: atan2(ty, tx) = atan2(1,1) = π/4
        return float(np.arctan2(1.0, 1.0))

    def beta(s: float) -> float:
        # Угол тангажа: atan2(tz, sqrt(tx²+ty²)) = atan2(1, √2)
        return float(np.arctan2(1.0, np.sqrt(2.0)))

    def eps(s: float) -> float:
        # Прямая: кривизна нулевая
        return 0.0

    return CurveGeom(p=p, t=t, yaw_star=yaw_star, beta=beta, eps=eps)


# ---------------------------------------------------------------------------
# Спиральная траектория x=r·cos(s), y=r·sin(s), z=s
# ---------------------------------------------------------------------------

def spiral_curve(r: float = 3.0) -> CurveGeom:
    """Спиральная кривая x=r·cos(s), y=r·sin(s), z=s с радиусом r."""
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


# ---------------------------------------------------------------------------
# Алгоритмы поиска ближайшей точки
# ---------------------------------------------------------------------------

def nearest_point_line(p_xyz: np.ndarray) -> float:
    """Аналитическая ближайшая точка на прямой x=s,y=s,z=s.

    ς* = (x+y+z)/3 (из диссертации стр. 41)
    """
    x, y, z = float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])
    return (x + y + z) / 3.0


def spiral_nearest_observer_step(
    zeta: float,
    p_xyz: np.ndarray,
    r: float = 3.0,
    gamma: float = 1.0,
    dt: float = 0.01,
) -> float:
    """Один шаг нелинейного наблюдателя ближайшей точки на спирали (Лемма 3).

    ζ̇ = -γ·ρ·H(ζ,x,y,z)
    H = r·sin(ζ)·x - r·cos(ζ)·y - z + ζ
    ρ = sign(∂H/∂ζ) = sign(1 + r·cos(ζ)·x + r·sin(ζ)·y)

    Параметры (стр. 43-44 диссертации): γ=1
    """
    x, y, z = float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])
    H = r * np.sin(zeta) * x - r * np.cos(zeta) * y - z + zeta
    rho = np.sign(1.0 + r * np.cos(zeta) * x + r * np.sin(zeta) * y)
    return float(zeta + (-gamma * rho * H) * dt)