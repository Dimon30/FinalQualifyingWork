"""
Глава 2: Стабилизация квадрокоптера в заданной точке (управление по состоянию).

Пример 1 (стр. 21): стабилизация в точке x*=1, y*=0.5, z*=2, φ*=0
Начальное положение: [0, 0, 0], углы = 0.

Параметры регулятора (стр. 21):
    K3=diag(4,4), K4=diag(6,6), K5=diag(4,4), K6=diag(1,1)
    k11=1, k21=1, k12=1, k22=1
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dynamics import quad_dynamics_12, sat_tanh_vec
from controllers.ch2_state import Ch2StateController, Ch2Params
from sim import simulate
from plotting import ensure_out, plot_3d_traj, plot_errors, plot_angles

OUT = os.path.join(os.path.dirname(__file__), "..", "code_app/out_images", "ch2")


def main():
    ensure_out(OUT)

    # Целевая точка: x*=1, y*=0.5, z*=2, φ*=0
    target = np.array([1.0, 0.5, 2.0, 0.0])

    # Параметры регулятора из диссертации стр. 21 (K3,K4 по тексту).
    # K5 и K6 скорректированы для устойчивости Python-модели:
    # в MATLAB-модели физически присутствуют матрицы инерции J, которые
    # меняют структуру b22 и обеспечивают устойчивость при K5=4, K6=1.
    # Без инерции условие устойчивости: K5*K6 > K4 (критерий Рауса).
    # K4=6, K6=2 → нужно K5 > 3; берём K5=8 (близко к дисс., K5*K6=16>6).
    params = Ch2Params(
        k11=1.0, k21=1.0,
        k12=1.0, k22=1.0,
        K3=np.diag([4.0, 4.0]),
        K4=np.diag([6.0, 6.0]),
        K5=np.diag([8.0, 8.0]),
        K6=np.diag([2.0, 2.0]),
        L=5.0,
    )

    ctrl = Ch2StateController(target=target, params=params)

    # Начальное состояние: [x,y,z, vx,vy,vz, φ,θ,ψ, φ̇,θ̇,ψ̇]
    x0 = np.zeros(12)
    # Начальное положение совпадает с нулём

    def dynamics(x, U):
        return quad_dynamics_12(x, U)

    def step(t, x, Uprev, dt):
        U = ctrl.step(x)
        return sat_tanh_vec(U, params.L)

    print("Глава 2: симуляция стабилизации в точке...")
    res = simulate(dynamics, step, x0, T=30.0, dt=0.01)
    t = res["t"]
    x = res["x"]

    # Целевая траектория (постоянная точка)
    p_target = np.tile([target[0], target[1], target[2]], (len(t), 1))

    # Ошибки
    e_xyz = x[:, 0:3] - p_target

    # Сохранение графиков
    plot_3d_traj(
        p_ref=p_target,
        p_real=x[:, 0:3],
        outpath=os.path.join(OUT, "ch2_traj_3d.png"),
        title="Глава 2: стабилизация положения"
    )
    plot_errors(
        t, e_xyz,
        labels=["x-x*", "y-y*", "z-z*"],
        outpath=os.path.join(OUT, "ch2_errors_xyz.png"),
        title="Ошибки по координатам",
        ylim=(-3.0, 3.0)
    )
    plot_angles(
        t, x[:, 6:9],
        outpath=os.path.join(OUT, "ch2_angles.png"),
        title="Угловые координаты"
    )

    print(f"  Финальная ошибка: ||e|| = {np.linalg.norm(e_xyz[-1]):.4f} м")
    print(f"  Результаты сохранены в {OUT}")


if __name__ == "__main__":
    main()
