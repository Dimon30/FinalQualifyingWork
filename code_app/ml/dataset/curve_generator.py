"""
ml/dataset/curve_generator.py
==============================
Числовые утилиты для генерации и проверки допустимых кривых датасета.

Кривая здесь — просто вызываемый объект ``p: Callable[[float], ndarray]``.
Полная геометрия (CurveGeom, CurveSpec) живёт в ``ml/curves/generator.py``;
этот модуль занимается *числовой* работой с кривой для формирования датасета.

Допустимые типы (контроллер Гл. 4 требует ||t(s)|| = const):
    line   — p(s) = [a·s, b·s, c·s],              ||t|| = sqrt(a²+b²+c²)
    circle — p(s) = [r·cos(s/r), r·sin(s/r), 0],  ||t|| = 1
    spiral — p(s) = [r·cos(s), r·sin(s), k·s],    ||t|| = sqrt(r²+k²)

Публичный API:
    make_line(a, b, c)          -> Callable
    make_circle(r)              -> Callable
    make_spiral(r, k)           -> Callable

    sample_curve_points(p, s_values)  -> ndarray [N×3]
    compute_tangent(p, s, h)          -> ndarray [3]
    compute_curvature(p, s, h)        -> float

    validate_curve(p, s_range, n_check, tol) -> bool
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Tuple

# Допустимый диапазон ||t(s)|| для контроллера Гл. 4
_TNORM_MIN: float = 1.0
_TNORM_MAX: float = 5.0
# Допуск на числовую погрешность при сравнении ||t|| с границами диапазона
_TNORM_EPS: float = 1e-6

# Тип кривой: скаляр → 3-вектор
Curve = Callable[[float], np.ndarray]


# ---------------------------------------------------------------------------
# Фабрики кривых
# ---------------------------------------------------------------------------

def make_line(a: float, b: float, c: float) -> Curve:
    """Прямая: p(s) = [a·s, b·s, c·s].

    ||t|| = sqrt(a²+b²+c²) = const.

    Параметры:
        a, b, c — компоненты вектора направления (не обязательно единичного)

    Пример:
        make_line(1, 1, 1)   →  диагональ, ||t||=√3 ≈ 1.73
        make_line(1, 0, 0)   →  вдоль X,   ||t||=1
    """
    d = np.array([a, b, c], dtype=float)
    if np.linalg.norm(d) < 1e-9:
        raise ValueError("Вектор направления (a,b,c) не может быть нулевым")

    def p(s: float) -> np.ndarray:
        return d * float(s)

    return p


def make_circle(r: float) -> Curve:
    """Нормированный круг радиуса r в плоскости XY.

    p(s) = [r·cos(s/r), r·sin(s/r), 0],  ||t|| = 1 = const.

    Параметры:
        r — радиус (> 0)
    """
    if r <= 0:
        raise ValueError(f"Радиус r должен быть > 0, получено {r}")

    def p(s: float) -> np.ndarray:
        theta = float(s) / r
        return np.array([r * np.cos(theta), r * np.sin(theta), 0.0])

    return p


def make_spiral(r: float, k: float) -> Curve:
    """Спираль: p(s) = [r·cos(s), r·sin(s), k·s].

    ||t|| = sqrt(r² + k²) = const.

    Параметры:
        r — радиус горизонтального круга (> 0)
        k — скорость вертикального подъёма (может быть 0, но тогда ||t||=r)
    """
    if r <= 0:
        raise ValueError(f"Радиус r должен быть > 0, получено {r}")

    def p(s: float) -> np.ndarray:
        s_ = float(s)
        return np.array([r * np.cos(s_), r * np.sin(s_), k * s_])

    return p


# ---------------------------------------------------------------------------
# Числовые утилиты
# ---------------------------------------------------------------------------

def sample_curve_points(
    curve: Curve,
    s_values: np.ndarray,
) -> np.ndarray:
    """Вычислить точки кривой в заданных параметрических значениях.

    Параметры:
        curve    — кривая p: float → ndarray[3]
        s_values — 1-D массив параметрических значений, shape (N,)

    Возвращает:
        ndarray shape (N, 3) — координаты точек кривой
    """
    s_arr = np.asarray(s_values, dtype=float)
    if s_arr.ndim != 1:
        raise ValueError("s_values должен быть 1-D массивом")
    return np.stack([curve(s) for s in s_arr], axis=0)


def compute_tangent(
    curve: Curve,
    s: float,
    h: float = 1e-5,
) -> np.ndarray:
    """Касательный вектор p'(s) численным центральным разностным методом.

    t(s) = (p(s+h) − p(s−h)) / (2h)

    Параметры:
        curve — кривая p: float → ndarray[3]
        s     — параметрическое значение
        h     — шаг конечной разности (по умолчанию 1e-5)

    Возвращает:
        ndarray[3] — касательный вектор (не нормированный)
    """
    return (curve(s + h) - curve(s - h)) / (2.0 * h)


def compute_curvature(
    curve: Curve,
    s: float,
    h: float = 1e-5,
) -> float:
    """Кривизна кривой κ(s) по формуле Френе.

    κ = ||p' × p''|| / ||p'||³

    p'  — центральная разность 1-го порядка
    p'' — центральная разность 2-го порядка

    Параметры:
        curve — кривая p: float → ndarray[3]
        s     — параметрическое значение
        h     — шаг конечной разности

    Возвращает:
        float — кривизна (≥ 0); для прямой возвращает 0.
    """
    # p' ≈ (p(s+h) - p(s-h)) / 2h
    t_vec = (curve(s + h) - curve(s - h)) / (2.0 * h)
    # p'' ≈ (p(s+h) - 2·p(s) + p(s-h)) / h²
    n_vec = (curve(s + h) - 2.0 * curve(s) + curve(s - h)) / (h ** 2)

    t_norm = float(np.linalg.norm(t_vec))
    if t_norm < 1e-12:
        return 0.0

    cross = np.cross(t_vec, n_vec)
    return float(np.linalg.norm(cross)) / (t_norm ** 3)


# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------

def validate_curve(
    curve: Curve,
    s_range: Tuple[float, float] = (0.0, 10.0),
    n_check: int = 50,
    tol: float = 0.02,
) -> bool:
    """Проверить, что кривая допустима для контроллера Гл. 4.

    Критерии:
        1. ||t(s)|| примерно константа: std(norms) / mean(norms) < tol
        2. Среднее ||t(s)|| ∈ [_TNORM_MIN, _TNORM_MAX] = [1, 5]

    Параметры:
        curve   — кривая p: float → ndarray[3]
        s_range — диапазон параметра для проверки (s_min, s_max)
        n_check — число контрольных точек
        tol     — допустимое относительное отклонение нормы касательной

    Возвращает:
        True  — кривая допустима
        False — кривая нарушает хотя бы одно условие
    """
    s_min, s_max = float(s_range[0]), float(s_range[1])
    if s_min >= s_max:
        raise ValueError(f"s_range должен быть (s_min < s_max), получено {s_range}")

    s_vals = np.linspace(s_min, s_max, n_check)
    norms = np.array([
        np.linalg.norm(compute_tangent(curve, s))
        for s in s_vals
    ])

    mean_norm = float(np.mean(norms))

    # 1. Проверка на постоянство ||t||
    if mean_norm < 1e-12:
        return False  # вырожденная кривая
    relative_std = float(np.std(norms)) / mean_norm
    if relative_std > tol:
        return False

    # 2. Проверка диапазона ||t|| (с допуском на числовую погрешность)
    if not (_TNORM_MIN - _TNORM_EPS <= mean_norm <= _TNORM_MAX + _TNORM_EPS):
        return False

    return True
