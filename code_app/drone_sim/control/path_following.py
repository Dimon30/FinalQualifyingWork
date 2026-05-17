"""Регулятор согласованного управления по выходу (Гл. 4, уравнения 71-77).

Закон управления:
    U = γ5·η + Ū,  η̇ = Ū
    Ū = sat_L[b^{-1}(θ,ψ,u1,φ)·W^{-1}(-σ - γ1·λ̂1 - γ2·λ̂2 - γ3·λ̂3 - γ4·λ̂4)]

Регулируемые переменные: λ̃1 = col(s_arc - V*t, e1, e2, δφ).
"""
from __future__ import annotations
import numpy as np
from typing import Optional

from drone_sim.models.dynamics import G, sat_tanh, sat_tanh_vec, thrust_direction
from drone_sim.models.quad_model import QuadModel
from drone_sim.control.common import HighGainParams, DerivativeObserver4
from drone_sim.geometry.curves import (
    CurveGeom, se_from_pose, nearest_point_line, spiral_nearest_observer_step
)


def W_mat(alpha: float, beta: float, eps: float) -> np.ndarray:
    """Матрица W(α,β,ε) — преобразование ошибок в систему координат кривой (стр. 38)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array([
        [ca*cb,      sa*cb,      sb,   0.0],
        [-sa,         ca,        0.0,  0.0],
        [-ca*sb,     -sa*sb,     cb,   0.0],
        [-eps*ca*cb, -eps*sa*cb, -eps*sb, 1.0],
    ], dtype=float)


def W_inv(alpha: float, beta: float, eps: float) -> np.ndarray:
    """Обратная W^{-1}(α,β,ε)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array([
        [ca*cb,   -sa,   -ca*sb,  0.0],
        [sa*cb,    ca,   -sa*sb,  0.0],
        [sb,       0.0,   cb,     0.0],
        [eps,      0.0,   0.0,    1.0],
    ], dtype=float)


def b_mat(
    phi: float,
    theta: float,
    psi: float,
    u1: float,
    g: Optional[float] = None,
) -> np.ndarray:
    """Матрица входов b = Rz(φ)·B_inner(θ,ψ,u1) (без перестановки строк, см. CLAUDE.md)."""
    if g is None:
        g = G
    d = float(u1 + g)
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cr, sr = np.cos(psi), np.sin(psi)

    B_inner = np.array([
        [st*cr,   0.0,   d*ct*cr,     -d*st*sr],
        [-sr,     0.0,   0.0,          -d*cr   ],
        [ct*cr,   0.0,  -d*st*cr,     -d*ct*sr],
        [0.0,     1.0,   0.0,           0.0    ],
    ], dtype=float)

    Rz_phi = np.array([
        [cp, -sp, 0.0, 0.0],
        [sp,  cp, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)

    return Rz_phi @ B_inner


def _safe_inv4(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Обращение 4×4 с Тихоновской регуляризацией при cond > 1e8."""
    try:
        cond = np.linalg.cond(M)
        if not np.isfinite(cond) or cond > 1e8:
            raise np.linalg.LinAlgError("плохая обусловленность")
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.inv(M + eps * np.eye(4))


class Ch4PathController:
    """Низкоуровневый регулятор Гл. 4 для прямой и спирали.

    Используется в legacy-сценариях. Для произвольной кривой см.
    drone_sim.simulation.path_sim.PathFollowingController.

    Параметры из диссертации:
        Прямая:  κ=100,  a=(5,10,10,5,1), γ=(1,3,5,3,1), L=5
        Спираль: κ=200,  те же a, γ
    """

    def __init__(
        self,
        curve: CurveGeom,
        Vstar: float,
        params: HighGainParams,
        use_spiral_observer: bool = False,
        r: float = 3.0,
        gamma_nearest: float = 1.0,
        quad_model: Optional[QuadModel] = None,
    ):
        self.curve = curve
        self.Vstar = float(Vstar)
        self.p = params
        self.use_spiral_observer = use_spiral_observer
        self.r = float(r)
        self.gamma_nearest = float(gamma_nearest)
        self._model = quad_model if quad_model is not None else QuadModel()

        self._zeta = 0.0
        self._eta = np.zeros(4, dtype=float)
        self.obs = DerivativeObserver4(dim=4, p=params)

    def _nearest_s(self, p_xyz: np.ndarray, dt: float) -> float:
        if self.use_spiral_observer:
            self._zeta = spiral_nearest_observer_step(
                self._zeta, p_xyz, r=self.r, gamma=self.gamma_nearest, dt=dt
            )
            return float(self._zeta)
        return float(nearest_point_line(p_xyz))

    def _lambda_tilde_1(
        self, t: float, p_xyz: np.ndarray, phi: float, s: float
    ) -> np.ndarray:
        """λ̃1 = col(s_arc - V*t, e1, e2, δφ); s_arc = ζ·||t(ζ)|| (точно при ||t||=const)."""
        _, e1, e2 = se_from_pose(p_xyz, s, self.curve)
        phi_star = float(self.curve.yaw_star(s))
        d_phi = float(np.arctan2(np.sin(phi - phi_star), np.cos(phi - phi_star)))
        tangent_norm = float(np.linalg.norm(self.curve.t(s)))
        s_arc = s * tangent_norm
        s_ref = self.Vstar * float(t)
        return np.array([s_arc - s_ref, e1, e2, d_phi], dtype=float)

    def step(
        self,
        t: float,
        x: np.ndarray,
        Uprev: Optional[np.ndarray],
        dt: float,
    ) -> np.ndarray:
        """Один шаг регулятора. Возвращает U = [v1, v2, u3, u4]."""
        p_xyz = x[0:3]
        phi, theta, psi = float(x[6]), float(x[7]), float(x[8])
        u1_bar = float(x[12])
        u1 = sat_tanh(u1_bar, self.p.L)

        s = self._nearest_s(p_xyz, dt)

        alpha = float(self.curve.yaw_star(s))
        beta_val = float(self.curve.beta(s))
        eps_val = float(self.curve.eps(s))

        W = W_mat(alpha, beta_val, eps_val)
        Winv = W_inv(alpha, beta_val, eps_val)

        lam1 = self._lambda_tilde_1(t, p_xyz, phi, s)

        b = b_mat(phi, theta, psi, u1, g=self._model.g)
        binv = _safe_inv4(b)

        l1h, l2h, l3h, l4h, sigma = self.obs.hat()
        g1, g2, g3, g4 = (
            self.p.gamma[0], self.p.gamma[1],
            self.p.gamma[2], self.p.gamma[3]
        )
        g5 = self.p.gamma[4]

        v = -sigma - g1*l1h - g2*l2h - g3*l3h - g4*l4h
        Ubar = sat_tanh_vec(binv @ (Winv @ v), self.p.L)

        self._eta += dt * Ubar
        U = g5 * self._eta + Ubar

        y4_model = W @ (b @ Ubar)
        self.obs.step(y=lam1, y4_model=y4_model, dt=dt)

        return U.astype(float)

    def reset(self):
        self._zeta = 0.0
        self._eta[:] = 0.0
        self.obs.reset()
