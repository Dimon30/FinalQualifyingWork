"""
Глава 3: Следящее управление по выходу — движение по спиральной траектории.

Параметры из диссертации (стр. 28):
    r = 0.5,  x*(t) = r·cos(t),  y*(t) = r·sin(t),  z*(t) = t,  φ*(t) = 0
    κ = 100,  a = (5,10,10,5,1),  γ = (1,4,6,4,4),  L = 5,  ℓ = 0.9

Начальное состояние: нулевое (дрон у земли, нет скоростей).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dynamics import quad_dynamics_16, sat_tanh
from controllers.ch3_output import Ch3OutputTrackingController
from controllers.common import HighGainParams
from trajectories import spiral_time_ref
from sim import simulate
from plotting import ensure_out, plot_3d_traj, plot_errors, plot_angles

OUT = os.path.join(os.path.dirname(__file__), "..", "code_app/out_images", "ch3")
L = 5.0  # уровень насыщения


def main():
    ensure_out(OUT)

    # Параметры из диссертации стр. 28
    params = HighGainParams(
        kappa=100.0,
        a=(5.0, 10.0, 10.0, 5.0, 1.0),
        gamma=(1.0, 4.0, 6.0, 4.0, 4.0),
        L=L,
        ell=0.9,
    )

    ref_fn = spiral_time_ref(r=0.5)
    ctrl = Ch3OutputTrackingController(params=params)

    # Начальное состояние 16D
    x0 = np.zeros(16)
    # Начинаем с нуля

    def dynamics(x, U):
        return quad_dynamics_16(x, U, L=L)

    def step(t, x, Uprev, dt):
        ref = ref_fn(t)
        return ctrl.step(t, x, ref, dt)

    print("Глава 3: симуляция следящего управления по выходу...")
    print("  Параметры: kappa=100, r=0.5, T=40с, dt=0.01")
    res = simulate(dynamics, step, x0, T=40.0, dt=0.01)
    t = res["t"]
    x = res["x"]

    # Опорная траектория
    p_ref = np.stack([ref_fn(tt).y[:3] for tt in t], axis=0)

    # Ошибки по координатам
    e_xyz = x[:, 0:3] - p_ref
    norm_e = np.linalg.norm(e_xyz, axis=1, keepdims=True)
    e_all = np.hstack([e_xyz, norm_e])

    plot_3d_traj(
        p_ref=p_ref,
        p_real=x[:, 0:3],
        outpath=os.path.join(OUT, "ch3_traj_3d.png"),
        title="Глава 3: следящее управление (спираль)"
    )
    plot_errors(
        t, e_all,
        labels=["x-x*", "y-y*", "z-z*", "||e||"],
        outpath=os.path.join(OUT, "ch3_errors_xyz.png"),
        title="Ошибки регулирования",
    )
    plot_angles(
        t, x[:, 6:9],
        outpath=os.path.join(OUT, "ch3_angles.png"),
        title="Угловые координаты"
    )

    print(f"  Финальная ошибка: ||e|| = {np.linalg.norm(e_xyz[-1]):.4f} м")
    print(f"  Результаты сохранены в {OUT}")


if __name__ == "__main__":
    main()
