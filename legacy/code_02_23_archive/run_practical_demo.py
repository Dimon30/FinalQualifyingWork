import numpy as np
import matplotlib.pyplot as plt

from drone.geometry import line_traj, helix_traj
from drone.practical_controller import PracticalGains, PathSpeedController
from drone.sim_practical import SimConfig, simulate


def main():
    traj = helix_traj(r=3.0)
    Vstar = 1.0

    gains = PracticalGains(kp_pos=6.0, kd_vel=4.0, kp_ang=12.0, kd_ang=7.0, u1_limit=5.0)
    ctrl = PathSpeedController(traj=traj, Vstar=Vstar, gains=gains)

    x0 = np.zeros(16)
    x0[0:3] = np.array([3.5, 0.0, 0.0])

    cfg = SimConfig(dt=0.01, T=40.0, L=gains.u1_limit)
    sim = simulate(ctrl, x0=x0, cfg=cfg)

    t = sim["t"]
    x = sim["x"]
    p = x[:,0:3]
    v = x[:,3:6]
    speed = np.linalg.norm(v, axis=1)

    ss = np.linspace(0, Vstar*t[-1] * 1.2, 800)
    curve = np.array([traj.p(float(s)) for s in ss])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(curve[:,0], curve[:,1], curve[:,2], label="Траектория S")
    ax.plot(p[:,0], p[:,1], p[:,2], label="Квадрокоптер")
    ax.legend()
    ax.set_title("Практический контроллер: траектория")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, speed)
    plt.axhline(Vstar, linestyle="--")
    plt.title("Скорость ||v|| и V*")
    plt.xlabel("t, s"); plt.ylabel("m/s")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
