from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.drone.simulink_ch4_port import simulate, Ch4Params


def main():
    p = Ch4Params(
        v_ref=1.0,
        traj_name="helix",  # line | helix | circle
        use_const_spatial_speed=True,  # чтобы скорость была именно V* в м/с
        V_star=2.0  # реальная скорость в пространстве
    )
    out = simulate(t1=30.0, dt=0.002, params=p)

    q = out["q"]
    ref = out["ref"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], label="Траектория (задание)")
    ax.plot(q[:, 0], q[:, 1], q[:, 2], label="Квадрокоптер")
    ax.set_title("Порт Simulink главы 4 (23.slx)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
