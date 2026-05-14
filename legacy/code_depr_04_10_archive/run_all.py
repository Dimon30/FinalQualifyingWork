from __future__ import annotations
import os
import numpy as np

from controllers.common import HighGainParams
from controllers.ch3_tracking import Ch3OutputTrackingController, TrackingRef
from controllers.ch4_coordinated import Ch4CoordinatedController
from trajectories import spiral_time_ref
from geometry import line_xyz_curve, spiral_curve
from sim import simulate
from plotting import ensure_out, plot_3d_traj, plot_errors

OUT = "out"

def main():
    ensure_out(OUT)

    # x = [p(3), v(3), angles(3), rates(3), u1_bar, rho1, u2, rho2]
    x0 = np.zeros(16)
    x0[0:3] = np.array([0.0, 0.0, 0.0])
    x0[6:9] = np.array([0.0, 0.0, 0.0])

    p = HighGainParams(kappa=100.0, a=(5,10,10,5,1), gamma=(1,4,6,4,1), L=5.0)
    ctrl3 = Ch3OutputTrackingController(p)
    ref_fun = spiral_time_ref(r=0.5)

    def step3(t, x, Uprev, dt):
        rp = ref_fun(t)
        ref = TrackingRef(y=rp.y, y1=rp.y1, y2=rp.y2, y3=rp.y3, y4=rp.y4)
        return ctrl3.step(t, x, ref, dt)

    res3 = simulate(step3, x0.copy(), T=40.0, dt=0.01, L=p.L)
    t = res3["t"]; x = res3["x"]
    pref = np.stack([ref_fun(tt).y[:3] for tt in t], axis=0)
    plot_3d_traj(pref, x[:,0:3], os.path.join(OUT, "ch3_spiral_3d.png"))
    e = x[:,0:3] - pref
    plot_errors(t, e, ["ex","ey","ez"], os.path.join(OUT, "ch3_errors_xyz.png"))

    # ===== Глава 4 =====
    curve_line = line_xyz_curve()
    p4 = HighGainParams(kappa=100.0, a=(5,10,10,5,1), gamma=(1,3,5,3,1), L=5.0)
    ctrl4_line = Ch4CoordinatedController(curve=curve_line, Vstar=1.0, params=p4, use_spiral_observer=False)

    def step4_line(t, x, Uprev, dt):
        return ctrl4_line.step(t, x, Uprev, dt)

    x0_line = x0.copy(); x0_line[0:3] = np.array([2.0, -1.0, 0.5])
    res4 = simulate(step4_line, x0_line, T=40.0, dt=0.01, L=p4.L)
    t4 = res4["t"]; x4 = res4["x"]
    sref = t4
    pref4 = np.stack([np.array([s,s,s]) for s in sref], axis=0)
    plot_3d_traj(pref4, x4[:,0:3], os.path.join(OUT, "ch4_line_3d.png"))

    # ===== Глава 4: согласованное вдоль спирали (observer nearest) =====
    curve_sp = spiral_curve(r=3.0)
    ctrl4_sp = Ch4CoordinatedController(curve=curve_sp, Vstar=1.0, params=p4, use_spiral_observer=True, r=3.0)

    def step4_sp(t, x, Uprev, dt):
        return ctrl4_sp.step(t, x, Uprev, dt)

    x0_sp = x0.copy(); x0_sp[0:3] = np.array([2.9, 0.0, 0.0])
    res4s = simulate(step4_sp, x0_sp, T=40.0, dt=0.01, L=p4.L)
    t5 = res4s["t"]; x5 = res4s["x"]
    pref5 = np.stack([curve_sp.p(tt) for tt in t5], axis=0)
    plot_3d_traj(pref5, x5[:,0:3], os.path.join(OUT, "ch4_spiral_3d.png"))

if __name__ == "__main__":
    main()
