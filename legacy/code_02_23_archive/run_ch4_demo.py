import numpy as np
import matplotlib.pyplot as plt

from drone.geometry import line_traj, helix_traj
from drone.controllers import HGParams, Ch4CoordinatedController
from drone.sim import SimConfig, simulate_ch4


def plot_3d(traj, sim, title):
    x = sim["x"]
    t = sim["t"]
    p = x[:,0:3]

    ss = np.linspace(0, max(1.0, t[-1]), 600)
    curve = np.array([traj.p(float(s)) for s in ss])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(curve[:,0], curve[:,1], curve[:,2], label="Траектория S")
    ax.plot(p[:,0], p[:,1], p[:,2], label="Квадрокоптер")
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()


def main():
    # ===== выбери траекторию =====
    traj = helix_traj(r=3.0)    # helix
    # traj = line_traj()        # line x=y=z

    Vstar = 1.0
    params = HGParams(kappa=200.0, L=5.0)
    ctrl = Ch4CoordinatedController(traj=traj, Vstar=Vstar, params=params)

    # состояние: [p(3), v(3), angles(3), rates(3), u1_bar, rho1, u2, rho2]
    x0 = np.zeros(16)
    x0[0:3] = np.array([3.5, 0.0, 0.0])  # стартовая точка
    x0[6:9] = np.array([0.0, 0.0, 0.0])

    cfg = SimConfig(dt=0.01, T=40.0, L=params.L)
    sim = simulate_ch4(traj=traj, ctrl=ctrl, x0=x0, cfg=cfg)

    # ===== графики =====
    plot_3d(traj, sim, title="Глава 4: движение вдоль траектории")

    t = sim["t"]
    x = sim["x"]
    p = x[:,0:3]
    v = x[:,3:6]
    speed = np.linalg.norm(v, axis=1)

    plt.figure()
    plt.plot(t, speed)
    plt.axhline(Vstar, linestyle="--")
    plt.title("Скорость ||v|| и V*")
    plt.xlabel("t, s")
    plt.ylabel("m/s")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
