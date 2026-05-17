"""Фиксированный набор из 4 тестовых сценариев для бенчмарка V*-моделей."""
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
    """Один сценарий бенчмарка: name, label, curve, x0, cfg_kw, warmup_time, vstar_rate."""
    name: str
    label: str
    curve: CurveGeom
    x0: np.ndarray
    cfg_kw: dict
    warmup_time: float = 5.0
    vstar_rate: float = 0.3


def get_test_suite() -> list[TestScenario]:
    """4 фиксированных сценария: spiral_r3, circle_r3z5, helix_r2, line_diag.

    Покрывают разные геометрические режимы: высокая/средняя ||t||, плоская кривая, прямая.
    """
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
