from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from controllers.common import HighGainParams
from controllers.ch4_coordinated import Ch4CoordinatedController
from dynamics import quad_dynamics_extended
from integrators import rk4
from geometry import spiral_curve

def main():
    curve = spiral_curve(r=3.0)
    p = HighGainParams(kappa=30.0, a=(5,10,10,5,1), gamma=(1,4,6,4,1), L=5.0)
    ctrl = Ch4CoordinatedController(curve=curve, Vstar=1.0, params=p, use_spiral_observer=True, r=3.0, gamma_np=1.0)

    # state: [p(3), v(3), angles(3), rates(3), u1_bar, rho1, u2, rho2]
    x0 = np.zeros(16, dtype=float)
    x0[0:3] = np.array([3.0, 0.0, 0.0])  # near helix
    T=30.0
    dt=0.002
    L=p.L

    def f(t, x):
        U = ctrl.step(t=t, x=x, Uprev=None, dt=dt)
        return quad_dynamics_extended(x, U, L=L)

    ts, xs = rk4(f, x0, 0.0, T, dt)
    q = xs[:,0:3]

    # reference for plot: r(s_ref) with s_ref=V*t
    s_ref = 1.0*ts
    ref = np.stack([3.0*np.cos(s_ref), 3.0*np.sin(s_ref), s_ref], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref[:,0], ref[:,1], ref[:,2], label="Траектория (задание)")
    ax.plot(q[:,0], q[:,1], q[:,2], label="Квадрокоптер")
    ax.set_title("Глава 4 (патч b_mat rows)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
