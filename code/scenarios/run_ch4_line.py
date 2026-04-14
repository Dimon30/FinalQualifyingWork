"""
Глава 4: Согласованное управление — движение вдоль прямолинейной траектории.

Кривая: x=s, y=s, z=s (прямая)
Параметры из диссертации (стр. 41):
    Начальное положение: x0=(1,1,0), V*=1 м/с, φ*=0
    κ=100, a=(5,10,10,5,1), γ=(1,3,5,3,1), γ5=1, L=5, ℓ=0.9
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from drone_sim.models.dynamics import quad_dynamics_16, sat_tanh
from drone_sim.control.path_following import Ch4PathController
from drone_sim.control.common import HighGainParams
from drone_sim.geometry.curves import line_xyz_curve, se_from_pose, nearest_point_line
from drone_sim.simulation.runner import simulate
from drone_sim.visualization.plotting import ensure_out, plot_3d_traj, plot_errors, plot_velocity

OUT = os.path.join(os.path.dirname(__file__), "..", "out_images", "ch4_line")
L = 5.0
Vstar = 1.0


def main():
    ensure_out(OUT)

    curve = line_xyz_curve()

    params = HighGainParams(
        kappa=100.0,
        a=(5.0, 10.0, 10.0, 5.0, 1.0),
        gamma=(1.0, 3.0, 5.0, 3.0, 1.0),
        L=L,
        ell=0.9,
    )

    ctrl = Ch4PathController(
        curve=curve,
        Vstar=Vstar,
        params=params,
        use_spiral_observer=False,
    )

    x0 = np.zeros(16)
    x0[0:3] = np.array([1.0, 1.0, 0.0])

    def dynamics(x, U):
        return quad_dynamics_16(x, U, L=L)

    def step(t, x, Uprev, dt):
        return ctrl.step(t, x, Uprev, dt)

    print("Глава 4 (прямая): симуляция согласованного управления...")
    print(f"  V*={Vstar}, kappa=100, T=40с, dt=0.005")
    res = simulate(dynamics, step, x0, T=40.0, dt=0.005)
    t = res["t"]
    x = res["x"]

    s_arr = np.array([nearest_point_line(x[k, 0:3]) for k in range(len(t))])
    p_ref = np.stack([curve.p(s) for s in s_arr], axis=0)

    tangent_norm_line = np.sqrt(3.0)
    errors = np.zeros((len(t), 3))
    for k in range(len(t)):
        s = s_arr[k]
        _, e1, e2 = se_from_pose(x[k, 0:3], s, curve)
        s_arc = s * tangent_norm_line
        errors[k] = [s_arc - Vstar * t[k], e1, e2]

    vel = np.linalg.norm(x[:, 3:6], axis=1)

    plot_3d_traj(
        p_ref=p_ref, p_real=x[:, 0:3],
        outpath=os.path.join(OUT, "ch4_line_3d.png"),
        title="Глава 4: согласованное управление (прямая)"
    )
    plot_errors(
        t, errors,
        labels=["s - V*t", "e1", "e2"],
        outpath=os.path.join(OUT, "ch4_line_errors.png"),
        title="Ошибки согласованного управления",
    )
    plot_velocity(
        t, vel, Vstar,
        outpath=os.path.join(OUT, "ch4_line_velocity.png"),
    )

    print(f"  Финальная скорость: {vel[-1]:.4f} м/с (цель: {Vstar})")
    print(f"  Финальные ошибки: s={errors[-1,0]:.3f}, e1={errors[-1,1]:.4f}, e2={errors[-1,2]:.4f}")
    print(f"  Результаты сохранены в {OUT}")


if __name__ == "__main__":
    main()
