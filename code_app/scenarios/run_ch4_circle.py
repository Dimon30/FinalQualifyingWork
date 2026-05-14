"""
Глава 4: Согласованное управление — движение вдоль круговой траектории.

Кривая: circle_r3_z5 — горизонтальный круг радиуса r=3 на высоте z=5.
    p(s) = [3·cos(s/3), 3·sin(s/3), 5.0]
    Параметризация по длине дуги: ||t|| = 1 = const.

Параметры (аналог спирали из диссертации):
    Начальное положение: x0=(3, 0, 5), V*=1 м/с
    κ=200, a=(5,10,10,5,1), γ=(1,3,5,3,1), γ5=1, L=5
    γ(наблюдатель ближ. точки) = 100  (безопасно при ||t||=1, dt=0.002)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from drone_sim import make_curve, SimConfig, QuadModel, simulate_path_following
from drone_sim.visualization.plotting import display_path

OUT = os.path.join(os.path.dirname(__file__), "..", "out_images", "ch4_circle")

R = 3.0
Z = 5.0
Vstar = 1.0

# Нормированный круг на высоте Z: ||t|| = 1 = const
curve = make_curve(lambda s, _r=R, _z=Z: np.array([
    _r * np.cos(s / _r),
    _r * np.sin(s / _r),
    _z,
]))

# Начальное положение — точка на кривой при s=0
x0 = np.zeros(16)
x0[0:3] = np.array([R, 0.0, Z])

cfg = SimConfig(
    quad_model=QuadModel(),
    Vstar=Vstar,
    T=40.0,
    dt=0.002,
    x0=x0,
    kappa=200.0,
    gamma=(1., 3., 5., 3., 1.),
    gamma_nearest=100.0,   # безопасно: 2 / (||t||²·dt) = 2/(1·0.002) = 1000
    zeta0=0.0,
)


def main():
    print("Глава 4 (circle_r3_z5): симуляция согласованного управления...")
    print(f"  Кривая: круг r={R}, z={Z}, ||t||=1")
    print(f"  V*={Vstar}, kappa=200, T=40с, dt=0.002")

    result = simulate_path_following(curve, cfg)
    result.print_summary()
    result.plot(OUT, prefix="circle_r3_z5")

    print(f"  Результаты сохранены в {display_path(OUT)}")


if __name__ == "__main__":
    main()
