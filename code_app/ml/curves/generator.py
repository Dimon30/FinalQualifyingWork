"""
ml/curves/generator.py
======================
Генератор допустимых кривых для обучения ML-модели.

Ограничение: контроллер Гл. 4 работает ТОЛЬКО на кривых с ||t(s)|| = const.
Допустимые типы:
    line    — прямая:              p(s) = s * direction,  ||t|| = ||direction||
    circle  — нормированный круг:  p(s) = [r*cos(s/r), r*sin(s/r), 0], ||t|| = 1
    spiral  — спираль:             p(s) = [r*cos(s), r*sin(s), k*s], ||t|| = sqrt(r²+k²)

Для каждой кривой сохраняются:
    - CurveGeom (объект для симуляции)
    - tangent_norm: ||t|| — нужно для gamma_nearest и контроля
    - gamma_nearest: безопасное значение γ при dt=ORACLE_DT
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from drone_sim import make_curve
from drone_sim.geometry.curves import CurveGeom
from ml.config import ORACLE_DT

# Допустимые типы кривых
CurveType = Literal["line", "circle", "spiral"]


@dataclass
class CurveSpec:
    """Спецификация кривой: геометрия + метаданные для ML-пайплайна.

    Атрибуты:
        curve        — геометрия кривой (CurveGeom)
        curve_type   — тип: "line", "circle", "spiral"
        params       — словарь параметров кривой (r, k, direction и пр.)
        tangent_norm — ||t(s)|| = const (гарантировано для допустимых кривых)
        gamma_nearest — безопасное γ: < 2 / (tangent_norm² * dt)
    """
    curve: CurveGeom
    curve_type: CurveType
    params: dict
    tangent_norm: float
    gamma_nearest: float


def _safe_gamma(tangent_norm: float, dt: float = ORACLE_DT) -> float:
    """Вычислить безопасное gamma_nearest.

    Условие устойчивости: γ·dt·||t||² < 2
    Берём с запасом 10× от границы: γ = 0.2 / (||t||² · dt).
    """
    return 0.2 / (tangent_norm**2 * dt)


def make_line_curve(
    direction: Tuple[float, float, float] | np.ndarray,
) -> CurveSpec:
    """Прямая вдоль direction: p(s) = s * direction.

    Параметры:
        direction — вектор направления (не обязательно единичный)
                    ||t|| = ||direction|| = const

    Пример:
        make_line_curve([1.0, 0.0, 0.0])   → прямая вдоль X, ||t||=1
        make_line_curve([1.0, 1.0, 1.0])   → диагональ, ||t||=sqrt(3)
    """
    d = np.asarray(direction, dtype=float)
    tn = float(np.linalg.norm(d))
    if tn < 1e-9:
        raise ValueError("direction не может быть нулевым вектором")

    curve = make_curve(lambda s, _d=d: _d * s)
    return CurveSpec(
        curve=curve,
        curve_type="line",
        params={"direction": d.tolist()},
        tangent_norm=tn,
        gamma_nearest=_safe_gamma(tn),
    )


def make_circle_curve(r: float) -> CurveSpec:
    """Нормированный круг радиуса r в плоскости XY.

    Параметризация: p(s) = [r·cos(s/r), r·sin(s/r), 0].
    ||t|| = 1 = const — безопасна для любого r.

    Параметры:
        r — радиус (> 0)
    """
    if r <= 0:
        raise ValueError(f"Радиус r должен быть > 0, получено r={r}")

    curve = make_curve(lambda s, _r=r: np.array([
        _r * np.cos(s / _r),
        _r * np.sin(s / _r),
        0.0,
    ]))
    return CurveSpec(
        curve=curve,
        curve_type="circle",
        params={"r": r},
        tangent_norm=1.0,
        gamma_nearest=_safe_gamma(1.0),
    )


def make_spiral_curve(r: float, k: float) -> CurveSpec:
    """Спираль: p(s) = [r·cos(s), r·sin(s), k·s].

    ||t|| = sqrt(r² + k²) = const.

    Параметры:
        r — радиус горизонтального круга
        k — скорость вертикального подъёма
    """
    if r <= 0:
        raise ValueError(f"Радиус r должен быть > 0, получено r={r}")

    tn = float(np.sqrt(r**2 + k**2))
    if tn < 1e-9:
        raise ValueError("||t|| ≈ 0: задайте r > 0 или k ≠ 0")

    curve = make_curve(lambda s, _r=r, _k=k: np.array([
        _r * np.cos(s),
        _r * np.sin(s),
        _k * s,
    ]))
    return CurveSpec(
        curve=curve,
        curve_type="spiral",
        params={"r": r, "k": k},
        tangent_norm=tn,
        gamma_nearest=_safe_gamma(tn),
    )


def generate_curve(
    curve_type: CurveType,
    rng: np.random.Generator | None = None,
) -> CurveSpec:
    """Сгенерировать случайную допустимую кривую заданного типа.

    Параметры:
        curve_type — "line", "circle" или "spiral"
        rng        — генератор случайных чисел (numpy); если None — создаётся новый

    Диапазоны параметров подобраны для РАЗНООБРАЗИЯ V_opt в датасете:
        line:   direction ~ случайный единичный вектор (||t||=1, кривизна=0)
                Прямые не ограничивают V* — используются только для обобщения.
        circle: r ~ U[0.5, 5.0]  →  κ = 1/r ∈ [0.2, 2.0]
                Нормированный круг: ||t||=1 при любом r.
                Малые r (тугие повороты) реально ограничивают V*.
        spiral: r ~ U[0.5, 2.5],  k = max(sqrt(1 - r²) + 0.1, U[0.3, 1.5])
                ||t|| = sqrt(r²+k²) ≥ 1.0 гарантируется выбором k.
                Малые r + умеренный k → высокая кривизна → ограничение V*.
    """
    if rng is None:
        rng = np.random.default_rng()

    if curve_type == "line":
        # Случайное направление, нормированное → ||t||=1
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        return make_line_curve(d)

    elif curve_type == "circle":
        # r ∈ [0.5, 5.0]: включает тугие круги (κ=2) для ограничения V*
        r = float(rng.uniform(0.5, 5.0))
        return make_circle_curve(r)

    elif curve_type == "spiral":
        # r ∈ [1.0, 3.0] — ограничение снизу для физической реализуемости.
        # Условие физической реализуемости: beta = atan2(k, r) < 45°
        #   (при beta > arccos(g/(L+g)) ≈ 49° дрон не может компенсировать гравитацию).
        # Безопасное условие: k ≤ 0.9 × r  (beta < 42°, 10%-запас от предела 45°).
        # r ≥ 1.0 гарантирует ||t|| = sqrt(r²+k²) ≥ r ≥ 1.0 при любом k ≥ 0.
        r = float(rng.uniform(1.0, 3.0))
        k_max = 0.9 * r   # beta < 42°
        k_min = 0.1       # минимальный подъём, чтобы спираль не была плоской
        k = float(rng.uniform(k_min, k_max))
        return make_spiral_curve(r, k)

    else:
        raise ValueError(f"Неизвестный тип кривой: {curve_type!r}. "
                         f"Допустимо: 'line', 'circle', 'spiral'")


def generate_dataset_curves(
    n: int = 200,
    seed: int = 42,
    type_weights: Tuple[float, float, float] = (0.1, 0.4, 0.5),
) -> List[CurveSpec]:
    """Сгенерировать список из n допустимых кривых для датасета.

    Параметры:
        n            — число кривых
        seed         — seed для воспроизводимости
        type_weights — веса для ("line", "circle", "spiral").
                       По умолчанию: (0.1, 0.4, 0.5) — мало прямых (они не ограничивают V*),
                       много окружностей и спиралей с тугими поворотами.

    Возвращает:
        List[CurveSpec] длиной n
    """
    rng = np.random.default_rng(seed)
    types: List[CurveType] = ["line", "circle", "spiral"]
    w = np.asarray(type_weights, dtype=float)
    w /= w.sum()

    chosen_types = rng.choice(types, size=n, p=w)  # type: ignore[arg-type]

    curves: List[CurveSpec] = []
    for ct in chosen_types:
        spec = generate_curve(ct, rng=rng)
        curves.append(spec)

    return curves
