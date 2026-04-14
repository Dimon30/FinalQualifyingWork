"""
drone_sim — пакет симуляции согласованного управления квадрокоптером.

Реализует алгоритмы Главы 4 диссертации Ким С.А. (2024).

Структура:
    drone_sim.models        — QuadModel, динамика (quad_dynamics_16 и др.)
    drone_sim.geometry      — CurveGeom, кривые, фрейм Френе
    drone_sim.control       — HighGainParams, Ch4PathController, W_mat, b_mat
    drone_sim.simulation    — simulate, SimConfig, SimResult, simulate_path_following
    drone_sim.visualization — plot_3d_traj, plot_errors и др.

Быстрый старт::

    import numpy as np
    from drone_sim import make_curve, SimConfig, simulate_path_following

    curve = make_curve(lambda s: np.array([3*np.cos(s), 3*np.sin(s), s]))
    cfg = SimConfig(Vstar=1.0, T=30.0, dt=0.002, kappa=200)
    result = simulate_path_following(curve, cfg)
    result.print_summary()
    result.plot("out_images/my_run")
"""
from drone_sim.models.quad_model import QuadModel
from drone_sim.geometry.curves import CurveGeom
from drone_sim.simulation.path_sim import (
    make_curve,
    NearestPointObserver,
    PathFollowingController,
    SimConfig,
    SimResult,
    simulate_path_following,
)

__all__ = [
    "QuadModel",
    "CurveGeom",
    "make_curve",
    "NearestPointObserver",
    "PathFollowingController",
    "SimConfig",
    "SimResult",
    "simulate_path_following",
]
