from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import Trajectory, se_from_pose, nearest_s_newton
from .utils import wrap_pi, sat_vec_tanh
from .dynamics import G


def W_mat(alpha: float, beta: float, eps: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array(
        [
            [ca * cb, sa * cb, sb, 0.0],
            [-sa, ca, 0.0, 0.0],
            [-ca * sb, -sa * sb, cb, 0.0],
            [-eps * ca * cb, -eps * sa * cb, -eps * sb, 1.0],
        ],
        dtype=float,
    )


def W_inv_mat(alpha: float, beta: float, eps: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array(
        [
            [ca * cb, -sa, -ca * sb, 0.0],
            [sa * cb, ca, -sa * sb, 0.0],
            [sb, 0.0, cb, 0.0],
            [eps, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def b_mat_simulink(phi: float, theta: float, psi: float, u1: float) -> np.ndarray:
    """b(θ,ψ,u1,φ) как в perfect_model (chapter4/23.slx, chart_103, flag=full)."""
    cph, sph = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi), np.sin(psi)
    d = float(u1 + G)

    A = np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, cph, -sph],
         [0.0, 0.0, sph, cph]],
        dtype=float,
    )
    B = np.array(
        [[ct*cp, 0.0, -st*cp*d, -ct*sp*d],
         [0.0,   1.0, 0.0,      0.0     ],
         [st*cp, 0.0,  ct*cp*d, -st*sp*d],
         [-sp,   0.0, 0.0,      -cp*d   ]],
        dtype=float,
    )
    return A @ B


def safe_inv4(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    try:
        c = np.linalg.cond(M)
        if not np.isfinite(c) or c > 1e8:
            raise np.linalg.LinAlgError("ill-conditioned")
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.inv(M + eps * np.eye(4))


@dataclass
class HGParams:
    kappa: float = 200.0
    a: Tuple[float, float, float, float, float] = (5.0, 10.0, 10.0, 5.0, 1.0)
    gamma: Tuple[float, float, float, float, float] = (1.0, 4.0, 6.0, 4.0, 1.0)
    L: float = 5.0


class DerivativeObserver4:
    def __init__(self, dim: int, p: HGParams):
        self.dim = int(dim)
        self.p = p
        self.l1 = np.zeros(self.dim)
        self.l2 = np.zeros(self.dim)
        self.l3 = np.zeros(self.dim)
        self.l4 = np.zeros(self.dim)
        self.sigma = np.zeros(self.dim)

    def hat(self):
        return self.l1, self.l2, self.l3, self.l4, self.sigma

    def step(self, y: np.ndarray, y4_model: np.ndarray, dt: float):
        k = float(self.p.kappa)
        a1, a2, a3, a4, a5 = map(float, self.p.a)

        y = np.asarray(y, dtype=float).reshape(self.dim)
        e = y - self.l1

        self.l1 = self.l1 + dt * (self.l2 + k * a1 * e)
        self.l2 = self.l2 + dt * (self.l3 + (k**2) * a2 * e)
        self.l3 = self.l3 + dt * (self.l4 + (k**3) * a3 * e)
        self.l4 = self.l4 + dt * (self.sigma + y4_model + (k**4) * a4 * e)
        self.sigma = self.sigma + dt * ((k**5) * a5 * e)


@dataclass
class Ch4State:
    s_prev: float = 0.0
    eta: np.ndarray = None

    def __post_init__(self):
        if self.eta is None:
            self.eta = np.zeros(4, dtype=float)


class Ch4CoordinatedController:
    """Глава 4: согласованное траекторное управление по выходу (71)-(77)."""

    def __init__(self, traj: Trajectory, Vstar: float, params: HGParams):
        self.traj = traj
        self.Vstar = float(Vstar)
        self.p = params
        self.state = Ch4State()
        self.obs = DerivativeObserver4(dim=4, p=params)

    def _lambda_tilde_1(self, t: float, p_xyz: np.ndarray, phi: float, s: float) -> np.ndarray:
        _s_local, e1, e2 = se_from_pose(p_xyz, s, self.traj)
        phi_star = float(self.traj.yaw_star(s))
        dphi = wrap_pi(phi - phi_star)

        # согласование по скорости: хотим ṡ -> V*, берём s_ref = V* t
        s_ref = self.Vstar * float(t)
        return np.array([s - s_ref, e1, e2, dphi], dtype=float)

    def step(self, t: float, x: np.ndarray, dt: float) -> np.ndarray:
        """Возвращает U = [v1, v2, u3, u4] для модели quad_dynamics_extended."""
        x = np.asarray(x, dtype=float)
        p_xyz = x[0:3]
        phi, theta, psi = map(float, x[6:9])

        # u1_bar — часть состояния объекта (двойной интегратор), u1 = sat_L(u1_bar)
        u1_bar = float(x[12])
        u1 = float(self.p.L * np.tanh(u1_bar / max(self.p.L, 1e-9)))

        # ===== геометрия =====
        s = nearest_s_newton(p_xyz, self.traj, s0=self.state.s_prev, iters=8)
        self.state.s_prev = s
        alpha = float(self.traj.yaw_star(s))
        beta = float(self.traj.beta(s))
        eps = float(self.traj.eps(s))

        W = W_mat(alpha, beta, eps)
        Winv = W_inv_mat(alpha, beta, eps)

        # ===== выход λ̃1 =====
        y = self._lambda_tilde_1(t, p_xyz, phi, s)

        # ===== регулятор (72) + динамика (71) =====
        l1h, l2h, l3h, l4h, sigma = self.obs.hat()
        g1, g2, g3, g4, g5 = map(float, self.p.gamma)

        b = b_mat_simulink(phi=phi, theta=theta, psi=psi, u1=u1)
        binv = safe_inv4(b)

        v = (-sigma - g1*l1h - g2*l2h - g3*l3h - g4*l4h)
        Ubar = sat_vec_tanh(binv @ (Winv @ v), self.p.L)

        # динамическая часть: U = γ5 η + Ubar, η̇ = Ubar
        self.state.eta = self.state.eta + dt * Ubar
        U = sat_vec_tanh(g5 * self.state.eta + Ubar, self.p.L)

        # ===== обновление наблюдателя =====
        y4_model = W @ (b @ Ubar)
        self.obs.step(y=y, y4_model=y4_model, dt=dt)

        return U.astype(float)
