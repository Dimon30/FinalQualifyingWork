"""Фиксированный набор тестовых сценариев для бенчмарка моделей V*.

Каждый сценарий описывает одну кривую с полными параметрами симуляции.
Набор детерминирован — служит воспроизводимым стандартом сравнения.

Использование::

    from ml.evaluation.test_suite import get_test_suite

    for scenario in get_test_suite():
        print(scenario.name, scenario.label)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from drone_sim import make_curve
from drone_sim.geometry.curves import (
    CurveGeom,
    spiral_curve,
    line_xyz_curve,
    nearest_point_line,
)


@dataclass
class TestScenario:
    """Один тестовый сценарий бенчмарка.

    Атрибуты:
        name        — машинное имя (используется в именах файлов)
        label       — человеко-читаемое описание (для графиков и отчёта)
        curve       — геометрия кривой CurveGeom
        x0          — начальный вектор состояния [16]
        cfg_kw      — словарь параметров SimConfig (кроме Vstar/speed_fn/quad_model)
        warmup_time — время прогрева в секундах (NN неактивна)
        vstar_rate  — максимальный темп изменения V* [1/с]
    """
    name: str
    label: str
    curve: CurveGeom
    x0: np.ndarray
    cfg_kw: dict
    warmup_time: float = 5.0
    vstar_rate: float = 0.3


def get_test_suite() -> list[TestScenario]:
    """Вернуть фиксированный тестовый набор из 4 сценариев.

    Сценарии выбраны для покрытия разных геометрических режимов:
    - spiral_r3   : высокая ||t||, стандарт диссертации
    - circle_r3z5 : плоская замкнутая кривая
    - helix_r2    : промежуточная кривизна, ||t||=√5
    - line_diag   : прямая (аналитическая ближайшая точка)
    """
    # --- Спираль r=3 ----------------------------------------------------------
    x0_spiral = np.zeros(16)
    x0_spiral[0:3] = [2.9, 0.0, 0.0]
    spiral_r3 = TestScenario(
        name="spiral_r3",
        label="Спираль $r{=}3$",
        curve=spiral_curve(r=3.0),
        x0=x0_spiral,
        cfg_kw=dict(
            T=40.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0,
        ),
        warmup_time=5.0,
        vstar_rate=0.3,
    )

    # --- Горизонтальная окружность r=3, z=5 -----------------------------------
    x0_circle = np.zeros(16)
    x0_circle[0:3] = [3.0, 0.0, 5.0]
    circle_r3z5 = TestScenario(
        name="circle_r3z5",
        label="Окружность $r{=}3$, $z{=}5$",
        curve=make_curve(lambda s: np.array([3.0 * np.cos(s), 3.0 * np.sin(s), 5.0])),
        x0=x0_circle,
        cfg_kw=dict(
            T=40.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0,
        ),
        warmup_time=5.0,
        vstar_rate=0.3,
    )

    # --- Спираль r=2 ----------------------------------------------------------
    x0_helix = np.zeros(16)
    x0_helix[0:3] = [1.9, 0.0, 0.0]
    helix_r2 = TestScenario(
        name="helix_r2",
        label="Спираль $r{=}2$",
        curve=make_curve(lambda s: np.array([2.0 * np.cos(s), 2.0 * np.sin(s), s])),
        x0=x0_helix,
        cfg_kw=dict(
            T=40.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=3.0, zeta0=0.0,
        ),
        warmup_time=5.0,
        vstar_rate=0.3,
    )

    # --- Прямая x=s, y=s, z=s -------------------------------------------------
    x0_line = np.zeros(16)
    x0_line[0:3] = [0.0, 0.0, 0.0]
    line_diag = TestScenario(
        name="line_diag",
        label="Прямая $x{=}s,y{=}s,z{=}s$",
        curve=line_xyz_curve(),
        x0=x0_line,
        cfg_kw=dict(
            T=30.0, dt=0.005, kappa=100.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0,
            nearest_fn=nearest_point_line,
        ),
        warmup_time=3.0,
        vstar_rate=0.5,
    )

    return [spiral_r3, circle_r3z5, helix_r2, line_diag]
