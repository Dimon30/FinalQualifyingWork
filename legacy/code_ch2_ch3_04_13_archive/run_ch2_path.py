"""
Глава 2: Второй пример — движение по ломаной траектории (пример 2, стр. 21).

Маршрут: A(5,5,5) → B(10,10,5) → C(20,5,10) → D(30,20,20)
Начальное положение: (2, 0, 0), желаемый курс φ* = 0.2 рад.
На вход регулятора подаётся последовательность точек в виде линейных функций времени.

Параметры (стр. 21 диссертации):
    K3=diag(100,100), K4=diag(100,100), K5=diag(100,100), K6=diag(10,10)
    k11=100, k21=100, k12=1, k22=1
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dynamics import quad_dynamics_12, sat_tanh_vec
from controllers.ch2_state import Ch2StateController, Ch2Params
from sim import simulate
from plotting import ensure_out, plot_3d_traj, plot_errors, plot_angles

OUT = os.path.join(os.path.dirname(__file__), "..", "code_app/out_images", "ch2_path")

# Маршрутные точки [x*, y*, z*, φ*]
PHI_STAR = 0.2
WAYPOINTS = np.array([
    [5.0,  5.0,  5.0,  PHI_STAR],  # A
    [10.0, 10.0, 5.0,  PHI_STAR],  # B
    [20.0, 5.0,  10.0, PHI_STAR],  # C
    [30.0, 20.0, 20.0, PHI_STAR],  # D
])
# Временны́е метки: A — t=0, D — t=30c, равномерно
SEG_TIMES = np.array([0.0, 10.0, 20.0, 30.0])


def current_target(t: float) -> np.ndarray:
    """Линейная интерполяция между путевыми точками."""
    if t <= SEG_TIMES[0]:
        return WAYPOINTS[0].copy()
    if t >= SEG_TIMES[-1]:
        return WAYPOINTS[-1].copy()
    for i in range(len(SEG_TIMES) - 1):
        if t <= SEG_TIMES[i + 1]:
            alpha = (t - SEG_TIMES[i]) / (SEG_TIMES[i + 1] - SEG_TIMES[i])
            return (1.0 - alpha) * WAYPOINTS[i] + alpha * WAYPOINTS[i + 1]
    return WAYPOINTS[-1].copy()


def main():
    ensure_out(OUT)

    # Параметры из диссертации стр. 21 (пример 2)
    params = Ch2Params(
        k11=100.0, k21=100.0,
        k12=1.0,   k22=1.0,
        K3=np.diag([100.0, 100.0]),
        K4=np.diag([100.0, 100.0]),
        K5=np.diag([100.0, 100.0]),
        K6=np.diag([10.0,  10.0]),
        L=5.0,
    )

    # Инициализируем контроллер с начальной точкой маршрута A
    ctrl = Ch2StateController(target=WAYPOINTS[0], params=params)

    # Начальное состояние: (2, 0, 0), углы и скорости = 0
    x0 = np.zeros(12)
    x0[0:3] = np.array([2.0, 0.0, 0.0])

    def dynamics(x, U):
        return quad_dynamics_12(x, U)

    def step(t, x, Uprev, dt):
        # Обновляем цель по времени (линейные функции времени)
        ctrl.target = current_target(t)
        U = ctrl.step(x)
        return sat_tanh_vec(U, params.L)

    print("Glava 2 (path): simulyaciya dvizheniya po marshrutu A->B->C->D...")
    print("  A(5,5,5)->B(10,10,5)->C(20,5,10)->D(30,20,20), phi*=0.2 rad")
    res = simulate(dynamics, step, x0, T=30.0, dt=0.005)
    t = res["t"]
    x = res["x"]

    # Опорная траектория (для графиков)
    p_ref = np.stack([current_target(tt)[:3] for tt in t], axis=0)

    # Ошибки по координатам
    e_xyz = x[:, 0:3] - p_ref
    norm_e = np.linalg.norm(e_xyz, axis=1, keepdims=True)
    e_all = np.hstack([e_xyz, norm_e])

    plot_3d_traj(
        p_ref=p_ref,
        p_real=x[:, 0:3],
        outpath=os.path.join(OUT, "ch2_path_3d.png"),
        title="Глава 2: движение по ломаной (пример 2)"
    )
    plot_errors(
        t, e_all,
        labels=["x-x*", "y-y*", "z-z*", "||e||"],
        outpath=os.path.join(OUT, "ch2_path_errors.png"),
        title="Ошибки регулирования при движении по ломаной",
    )
    plot_angles(
        t, x[:, 6:9],
        outpath=os.path.join(OUT, "ch2_path_angles.png"),
        title="Угловые координаты (ломаная)"
    )

    # XY проекция
    from plotting import plot_xy
    plot_xy(
        p_ref=p_ref,
        p_real=x[:, 0:3],
        outpath=os.path.join(OUT, "ch2_path_xy.png"),
        title="Проекция X-Y (ломаная)"
    )

    print(f"  Финальная ошибка: ||e|| = {np.linalg.norm(e_xyz[-1]):.4f} м")
    print(f"  Результаты сохранены в {OUT}")


if __name__ == "__main__":
    main()
