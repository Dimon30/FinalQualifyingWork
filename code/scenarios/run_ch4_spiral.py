"""
Глава 4: Согласованное управление — движение вдоль спиральной траектории.

Кривая: x=r·cos(s), y=r·sin(s), z=s, r=3
Параметры из диссертации (стр. 43-44):
    Начальное положение: x0=(2.9,0,0), V*=1 м/с, φ*=0
    κ=200, a=(5,10,10,5,1), γ=(1,3,5,3,1), γ5=1, L=5, ℓ=0.9
    γ(наблюдатель ближ. точки) = 1
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from drone_sim.models.dynamics import quad_dynamics_16
from drone_sim.control.path_following import Ch4PathController
from drone_sim.control.common import HighGainParams
from drone_sim.geometry.curves import spiral_curve, se_from_pose, spiral_nearest_observer_step
from drone_sim.simulation.runner import simulate
from drone_sim.visualization.plotting import (
    ensure_out, display_path, plot_3d_traj, plot_errors, plot_velocity, plot_xy
)

OUT = os.path.join(os.path.dirname(__file__), "..", "out_images", "ch4_spiral")
L = 5.0
Vstar = 1.0
R = 3.0


def main():
    ensure_out(OUT)

    curve = spiral_curve(r=R)

    params = HighGainParams(
        kappa=200.0,
        a=(5.0, 10.0, 10.0, 5.0, 1.0),
        gamma=(1.0, 3.0, 5.0, 3.0, 1.0),
        L=L,
        ell=0.9,
    )

    ctrl = Ch4PathController(
        curve=curve,
        Vstar=Vstar,
        params=params,
        use_spiral_observer=True,
        r=R,
        gamma_nearest=1.0,
    )

    x0 = np.zeros(16)
    x0[0:3] = np.array([2.9, 0.0, 0.0])

    def dynamics(x, U):
        return quad_dynamics_16(x, U, L=L)

    def step(t, x, Uprev, dt):
        return ctrl.step(t, x, Uprev, dt)

    print("Глава 4 (спираль): симуляция согласованного управления...")
    print(f"  V*={Vstar}, kappa=200, r={R}, T=40с, dt=0.002")
    res = simulate(dynamics, step, x0, T=40.0, dt=0.002)
    t = res["t"]
    x = res["x"]

    zeta_arr = np.zeros(len(t))
    zeta = 0.0
    for k in range(len(t)):
        zeta = spiral_nearest_observer_step(zeta, x[k, 0:3], r=R, gamma=1.0, dt=0.002)
        zeta_arr[k] = zeta

    p_ref = np.stack([curve.p(z) for z in zeta_arr], axis=0)

    tangent_norm_spiral = np.sqrt(R**2 + 1.0)
    errors = np.zeros((len(t), 3))
    for k in range(len(t)):
        s = zeta_arr[k]
        _, e1, e2 = se_from_pose(x[k, 0:3], s, curve)
        s_arc = s * tangent_norm_spiral
        errors[k] = [s_arc - Vstar * t[k], e1, e2]

    phi_arr = x[:, 6]
    phi_star_arr = np.array([curve.yaw_star(z) for z in zeta_arr])
    dphi_arr = np.arctan2(np.sin(phi_arr - phi_star_arr),
                          np.cos(phi_arr - phi_star_arr))

    vel = np.linalg.norm(x[:, 3:6], axis=1)

    plot_3d_traj(
        p_ref=p_ref, p_real=x[:, 0:3],
        outpath=os.path.join(OUT, "ch4_spiral_3d.png"),
        title="Глава 4: согласованное управление (спираль)"
    )
    plot_xy(
        p_ref=p_ref, p_real=x[:, 0:3],
        outpath=os.path.join(OUT, "ch4_spiral_xy.png"),
        title="Проекция X-Y"
    )
    plot_errors(
        t, errors,
        labels=["s - V*t", "e1", "e2"],
        outpath=os.path.join(OUT, "ch4_spiral_errors.png"),
        title="Ошибки согласованного управления",
    )
    plot_errors(
        t, dphi_arr.reshape(-1, 1),
        labels=["δφ, рад"],
        outpath=os.path.join(OUT, "ch4_spiral_dphi.png"),
        title="Ошибка по углу рысканья",
    )
    plot_velocity(
        t, vel, Vstar,
        outpath=os.path.join(OUT, "ch4_spiral_velocity.png"),
    )

    print(f"  Финальная скорость: {vel[-1]:.4f} м/с (цель: {Vstar})")
    print(f"  Финальные ошибки: s={errors[-1,0]:.3f}, e1={errors[-1,1]:.4f}, e2={errors[-1,2]:.4f}")
    print(f"  Результаты сохранены в {display_path(OUT)}")


if __name__ == "__main__":
    main()
