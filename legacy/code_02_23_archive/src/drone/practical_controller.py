from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Trajectory, nearest_s_newton
from .dynamics import G
from .utils import wrap_pi, sat_tanh


@dataclass
class PracticalGains:
    kp_pos: float = 4.0
    kd_vel: float = 3.0
    kp_ang: float = 10.0
    kd_ang: float = 6.0
    u1_limit: float = 5.0
    ang_limit: float = np.deg2rad(35.0)
    u_ang_limit: float = 8.0


class PathSpeedController:
    """Практический траекторный контроллер: следуем траектории с заданной скоростью V*.

    Это не high-gain закон (71)-(77), но он:
      - использует ту же модель объекта (ускорения как в Simulink),
      - обеспечивает устойчивое слежение по положению и заданной скорости вдоль касательной.
    """

    def __init__(self, traj: Trajectory, Vstar: float, gains: PracticalGains):
        self.traj = traj
        self.Vstar = float(Vstar)
        self.g = gains
        self.s_prev = 0.0

    def step(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        p = x[0:3]
        v = x[3:6]
        phi, theta, psi = map(float, x[6:9])  # yaw, pitch, roll
        phidot, thetadot, psidot = map(float, x[9:12])

        # closest point on curve
        s = nearest_s_newton(p, self.traj, s0=self.s_prev, iters=8)
        self.s_prev = s
        ps = self.traj.p(s)
        t_hat = self.traj.tangent(s)

        # desired velocity: V* along tangent
        v_ref = self.Vstar * t_hat

        # PD in position + velocity
        a_cmd = (
            self.g.kp_pos * (ps - p)
            + self.g.kd_vel * (v_ref - v)
        )

        # add gravity compensation so that u1=0 corresponds to hover at angles 0
        a_des = a_cmd + np.array([0.0, 0.0, G], dtype=float)

        # desired yaw along tangent projection
        yaw_ref = float(np.arctan2(t_hat[1], t_hat[0]))
        yaw_err = wrap_pi(phi - yaw_ref)

        # compute desired pitch/roll from desired accel and yaw_ref
        cy, sy = np.cos(yaw_ref), np.sin(yaw_ref)
        # rotate desired accel into yaw frame
        ax =  cy * a_des[0] + sy * a_des[1]
        ay = -sy * a_des[0] + cy * a_des[1]
        az = a_des[2]

        # small-angle mapping for pitch/roll
        theta_ref = np.clip(np.arctan2(ax, max(az, 1e-3)), -self.g.ang_limit, self.g.ang_limit)
        psi_ref   = np.clip(np.arctan2(-ay, max(az, 1e-3)), -self.g.ang_limit, self.g.ang_limit)

        # thrust command u1 such that (u1+g)*b ≈ a_cmd + g e3
        # using current angles for b:
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)
        b = np.array([cphi*sth*cps + sphi*sps,
                      sphi*sth*cps - cphi*sps,
                      cth*cps], dtype=float)
        thrust = float(a_des @ b)  # projection
        u1 = np.clip(thrust - G, -self.g.u1_limit, self.g.u1_limit)

        # inner-loop PD for angles -> angular accelerations (u2,u3,u4)
        # remember: in plant φ̈ = u2, θ̈ = u3, ψ̈ = u4
        u2 = np.clip(-self.g.kp_ang * yaw_err - self.g.kd_ang * phidot, -self.g.u_ang_limit, self.g.u_ang_limit)
        u3 = np.clip(self.g.kp_ang * (theta_ref - theta) - self.g.kd_ang * thetadot, -self.g.u_ang_limit, self.g.u_ang_limit)
        u4 = np.clip(self.g.kp_ang * (psi_ref - psi) - self.g.kd_ang * psidot, -self.g.u_ang_limit, self.g.u_ang_limit)

        # we output directly u2,u3,u4 as accelerations, and v1/v2 = 0 (keep u1,u2 without extra integrators)
        # but the plant expects U=[v1,v2,u3,u4] with u1,u2 being double-integrator outputs.
        # We'll keep u1,u2 close to commands by driving the integrators with v1,v2.
        u1_bar = float(x[12])
        rho1 = float(x[13])
        u2_state = float(x[14])
        rho2 = float(x[15])

        # simple second-order tracking for u1,u2 (critically damped)
        wn = 12.0
        v1 = wn**2 * (u1 - sat_tanh(u1_bar, self.g.u1_limit)) - 2*wn * rho1
        v2 = wn**2 * (u2 - u2_state) - 2*wn * rho2

        return np.array([v1, v2, u3, u4], dtype=float), dict(
            s=s, ps=ps, v_ref=v_ref, yaw_ref=yaw_ref, theta_ref=theta_ref, psi_ref=psi_ref, u1=u1, u2=u2
        )
