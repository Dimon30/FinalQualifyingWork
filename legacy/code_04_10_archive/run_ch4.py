from __future__ import annotations
import os
import numpy as np

from controllers.common import HighGainParams
from controllers.ch4_coordinated import Ch4CoordinatedController
from geometry import line_xyz_curve, spiral_curve
from sim import simulate
from plotting import ensure_out, plot_3d_traj

OUT = "out"

def main():
    ensure_out(OUT)

    # x = [p(3), v(3), angles(3), rates(3), u1_bar, rho1, u2, rho2]
    x0 = np.zeros(16)
    x0[0:3] = np.array([2.0, -1.0, 0.5])   # start off the curve a bit
    x0[6:9] = np.array([0.0, 0.0, 0.0])

    # High-gain observer is very sensitive to dt. Use smaller kappa with dt=0.002.
    p4 = HighGainParams(kappa=20.0, a=(5,10,10,5,1), gamma=(1,3,5,3,1), L=5.0)

    # Choose trajectory here (ONE place):
    curve = spiral_curve(r=3.0)   # or: line_xyz_curve()

    ctrl = Ch4CoordinatedController(curve=curve, Vstar=1.0, params=p4, use_spiral_observer=True, r=3.0)

    def step(t, x, Uprev, dt):
        return ctrl.step(t, x, Uprev, dt)

    res = simulate(step, x0, T=30.0, dt=0.002, L=p4.L)
    t = res["t"]; x = res["x"]
    pref = np.stack([curve.p(tt) for tt in t], axis=0)
    plot_3d_traj(pref, x[:,0:3], os.path.join(OUT, "ch4_traj_3d.png"))

if __name__ == "__main__":
    main()
