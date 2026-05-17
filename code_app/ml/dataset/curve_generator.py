"""Числовые утилиты для генерации и проверки кривых датасета.

Полная геометрия (CurveGeom, CurveSpec) — в ``ml/curves/generator.py``.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Tuple

# Допустимый диапазон ||t(s)|| для контроллера Гл. 4
_TNORM_MIN: float = 1.0
_TNORM_MAX: float = 5.0
_TNORM_EPS: float = 1e-6

Curve = Callable[[float], np.ndarray]


def make_line(a: float, b: float, c: float) -> Curve:
    """Прямая p(s) = [a·s, b·s, c·s], ||t|| = sqrt(a²+b²+c²)."""
    d = np.array([a, b, c], dtype=float)
    if np.linalg.norm(d) < 1e-9:
        raise ValueError("Вектор направления (a,b,c) не может быть нулевым")

    def p(s: float) -> np.ndarray:
        return d * float(s)

    return p


def make_circle(r: float) -> Curve:
    """Круг радиуса r в плоскости XY: p(s) = [r·cos(s/r), r·sin(s/r), 0], ||t||=1."""
    if r <= 0:
        raise ValueError(f"Радиус r должен быть > 0, получено {r}")

    def p(s: float) -> np.ndarray:
        theta = float(s) / r
        return np.array([r * np.cos(theta), r * np.sin(theta), 0.0])

    return p


def make_spiral(r: float, k: float) -> Curve:
    """Спираль p(s) = [r·cos(s), r·sin(s), k·s], ||t|| = sqrt(r²+k²)."""
    if r <= 0:
        raise ValueError(f"Радиус r должен быть > 0, получено {r}")

    def p(s: float) -> np.ndarray:
        s_ = float(s)
        return np.array([r * np.cos(s_), r * np.sin(s_), k * s_])

    return p


def sample_curve_points(curve: Curve, s_values: np.ndarray) -> np.ndarray:
    """Вычислить точки кривой в s_values; возвращает массив (N, 3)."""
    s_arr = np.asarray(s_values, dtype=float)
    if s_arr.ndim != 1:
        raise ValueError("s_values должен быть 1-D массивом")
    return np.stack([curve(s) for s in s_arr], axis=0)


def compute_tangent(curve: Curve, s: float, h: float = 1e-5) -> np.ndarray:
    """Касательная p'(s) центральной разностью."""
    return (curve(s + h) - curve(s - h)) / (2.0 * h)


def compute_curvature(curve: Curve, s: float, h: float = 1e-5) -> float:
    """Кривизна κ = ||p' × p''|| / ||p'||³ (формула Френе)."""
    t_vec = (curve(s + h) - curve(s - h)) / (2.0 * h)
    n_vec = (curve(s + h) - 2.0 * curve(s) + curve(s - h)) / (h ** 2)

    t_norm = float(np.linalg.norm(t_vec))
    if t_norm < 1e-12:
        return 0.0

    cross = np.cross(t_vec, n_vec)
    return float(np.linalg.norm(cross)) / (t_norm ** 3)


def validate_curve(
    curve: Curve,
    s_range: Tuple[float, float] = (0.0, 10.0),
    n_check: int = 50,
    tol: float = 0.02,
) -> bool:
    """Кривая допустима, если ||t(s)|| ≈ const (std/mean < tol) и mean ∈ [1, 5]."""
    s_min, s_max = float(s_range[0]), float(s_range[1])
    if s_min >= s_max:
        raise ValueError(f"s_range должен быть (s_min < s_max), получено {s_range}")

    s_vals = np.linspace(s_min, s_max, n_check)
    norms = np.array([
        np.linalg.norm(compute_tangent(curve, s))
        for s in s_vals
    ])

    mean_norm = float(np.mean(norms))
    if mean_norm < 1e-12:
        return False
    relative_std = float(np.std(norms)) / mean_norm
    if relative_std > tol:
        return False
    if not (_TNORM_MIN - _TNORM_EPS <= mean_norm <= _TNORM_MAX + _TNORM_EPS):
        return False

    return True
