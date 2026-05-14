"""
Опорные траектории для задачи следящего управления (Глава 3).

Траектория задаётся как функция времени y*(t) со всеми производными
до 4-го порядка включительно (необходимо для закона управления по выходу).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TrajPoint:
    """Точка опорной траектории со всеми производными."""
    y: np.ndarray   # y*(t)
    y1: np.ndarray  # ẏ*(t)
    y2: np.ndarray  # ÿ*(t)
    y3: np.ndarray  # y*(3)(t)
    y4: np.ndarray  # y*(4)(t)


def spiral_time_ref(r: float = 0.5) -> Callable[[float], TrajPoint]:
    """Спиральная опорная траектория из Главы 3 диссертации (стр. 28).

    x*(t) = r·cos(t),  y*(t) = r·sin(t),  z*(t) = t,  φ*(t) = 0

    Параметры из диссертации: r = 0.5
    """
    def f(t: float) -> TrajPoint:
        y  = np.array([r * np.cos(t),  r * np.sin(t),  t,    0.0], dtype=float)
        y1 = np.array([-r * np.sin(t), r * np.cos(t),  1.0,  0.0], dtype=float)
        y2 = np.array([-r * np.cos(t), -r * np.sin(t), 0.0,  0.0], dtype=float)
        y3 = np.array([r * np.sin(t),  -r * np.cos(t), 0.0,  0.0], dtype=float)
        y4 = np.array([r * np.cos(t),  r * np.sin(t),  0.0,  0.0], dtype=float)
        return TrajPoint(y=y, y1=y1, y2=y2, y3=y3, y4=y4)
    return f