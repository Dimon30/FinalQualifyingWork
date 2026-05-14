from __future__ import annotations

"""Глава 4 — согласованное управление вдоль пространственной кривой.

Здесь реализован итоговый закон управления по выходу из диссертации:
  (4.44)–(4.50) / (71)–(77) — динамический согласованный регулятор +
  расширенный (high‑gain) наблюдатель производных.

Важно по обозначениям (как в коде проекта):
  φ = yaw, θ = pitch, ψ = roll.

Управляющий вектор в расширенной модели:
  U = col(v1, v2, u3, u4),
где u1 и u2 — выходы двойных интеграторов:
  u1̇ = ρ1, ρ̇1 = v1;
  u2̇ = ρ2, ρ̇2 = v2;
а u3=θ¨, u4=ψ¨.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from controllers.common import HighGainParams, DerivativeObserver4, sat_vec_tanh
from dynamics import G
from geometry import CurveGeom, nearest_point_line, spiral_nearest_observer_step, se_from_pose


def W_mat(alpha: float, beta: float, eps: float) -> np.ndarray:
    """Матрица W(α,β) из (4.25)/(64).

    В диссертации ε зависит от кривизны (связана с производной α по s).
    В наших примерах ε берётся из curve.eps(s).
    """
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
    """Матрица W^{-1}(α,β) из (4.26)."""
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


def b_mat(phi: float, theta: float, psi: float, u1: float) -> np.ndarray:
    """Матрица b(θ,ψ,u1,φ) из (4.25)/(64).

    Буквально переносим структуру из диссертации/статьи:
      b = diag(Rz(φ), I2) @ B(θ,ψ,d),  d = u1 + g.

    Важно: в управлении по выходу используется именно b(θ,ψ,u1,φ) (а не полная
    b(λ,φ) с добавкой \tilde b), что и даёт робастность.
    """
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    d = float(u1 + G)

    # [Rz(φ) 0; 0 I2]
    A = np.array(
        [
            [cph, -sph, 0.0, 0.0],
            [sph, cph, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # B(θ,ψ,d) как в (64) — 4×4:
    # строка 1-2: зависят от θ,ψ и d; строка 3-4: (u2̈, u3, u4) части.
    # В «пакете» диссертации матрица записана блочно; здесь записываем явно.
    B = np.array(
        [
            # v1,   v2,                          u3,                          u4
            [sth * cps, 0.0, d * (cth * cps - sth * sps), 0.0],
            [-sps, 0.0, 0.0, d * (-cps)],
            [cth * cps, 0.0, d * (-sth * cps - cth * sps), 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    M = A @ B

    # В диссертации (см. b4(0,φ)) строки упорядочены как:
    #   [v1, v2, u3, u4] -> [..] так, что первые 2 строки соответствуют (v1,v2),
    #   а последние 2 — g-вращению для (u3,u4). В нашей явной сборке A@B эти блоки
    #   оказываются переставлены местами. Исправляем перестановкой строк.
    P = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return P @ M
def safe_inv4(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Устойчивая инверсия 4×4: сначала обычная, при плохой обусловленности — Tikhonov."""
    try:
        c = np.linalg.cond(M)
        if not np.isfinite(c) or c > 1e8:
            raise np.linalg.LinAlgError("ill-conditioned")
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.inv(M + eps * np.eye(4))


@dataclass
class Ch4Internal:
    """Внутренние состояния регулятора главы 4."""

    # оценка ближайшей точки на кривой (для спирали используем динамический наблюдатель)
    zeta: float = 0.0
    # динамический блок η из (4.44)/(71)
    eta: np.ndarray = None

    def __post_init__(self):
        if self.eta is None:
            self.eta = np.zeros(4, dtype=float)


class Ch4CoordinatedController:
    """Согласованное управление по выходу (Глава 4) + high-gain наблюдатель."""

    def __init__(
        self,
        curve: CurveGeom,
        Vstar: float,
        params: HighGainParams,
        use_spiral_observer: bool = False,
        r: float = 3.0,
        gamma_np: float = 1.0,
    ):
        self.curve = curve
        self.Vstar = float(Vstar)
        self.p = params
        self.state = Ch4Internal(zeta=0.0)
        self.use_spiral_observer = bool(use_spiral_observer)
        self.r = float(r)
        self.gamma_np = float(gamma_np)

        # наблюдатель производных для λ̃1 ∈ R^4
        self.obs = DerivativeObserver4(dim=4, p=params)

    def _nearest_s(self, p_xyz: np.ndarray, dt: float) -> float:
        if self.use_spiral_observer:
            self.state.zeta = spiral_nearest_observer_step(
                self.state.zeta, p_xyz, r=self.r, gamma=self.gamma_np, dt=dt
            )
            return float(self.state.zeta)
        return float(nearest_point_line(p_xyz))

    def _lambda_tilde_1(self, t: float, p_xyz: np.ndarray, phi: float, s: float) -> np.ndarray:
        # координаты ошибки в системе касательной/нормалей
        s_local, e1, e2 = se_from_pose(p_xyz, s, self.curve)
        phi_star = float(self.curve.yaw_star(s))
        dphi = float(np.arctan2(np.sin(phi - phi_star), np.cos(phi - phi_star)))

        # цель «согласования» — ṡ → V*  =>  s_ref = V* t
        s_ref = self.Vstar * float(t)
        return np.array([s - s_ref, e1, e2, dphi], dtype=float)

    def step(self, t: float, x: np.ndarray, Uprev: Optional[np.ndarray], dt: float) -> np.ndarray:
        """Один шаг регулятора.

        Возвращает U = [v1, v2, u3, u4] для модели quad_dynamics_extended.
        """
        # ===== измеряемые состояния =====
        p_xyz = x[0:3]
        phi, theta, psi = map(float, x[6:9])
        u1_bar = float(x[12])

        # настоящий u1 (с учётом насыщения) — нужен в b(·)
        u1 = float(self.p.L * np.tanh(u1_bar / max(self.p.L, 1e-9)))

        # ===== геометрия =====
        s = self._nearest_s(p_xyz, dt)
        alpha = float(self.curve.yaw_star(s))
        beta = float(self.curve.beta(s))
        eps = float(self.curve.eps(s))

        W = W_mat(alpha, beta, eps)
        Winv = W_inv_mat(alpha, beta, eps)

        # ===== выход λ̃1 =====
        y = self._lambda_tilde_1(t, p_xyz, phi, s)

        # ===== управление по выходу =====
        l1h, l2h, l3h, l4h, sigma = self.obs.hat()
        g1, g2, g3, g4, g5 = self.p.gamma

        b = b_mat(phi=phi, theta=theta, psi=psi, u1=u1)
        binv = safe_inv4(b)

        # Ubar = sat_L( b^{-1} W^{-1} ( -σ - Σ γi λ̂i ) )
        v = (-sigma - g1 * l1h - g2 * l2h - g3 * l3h - g4 * l4h)
        Ubar = sat_vec_tanh(binv @ (Winv @ v), self.p.L)

        # динамическая часть (γ5 η + Ubar), η̇ = Ubar
        self.state.eta += dt * Ubar
        U = g5 * self.state.eta + Ubar

        # ===== обновление наблюдателя =====
        # y4_model = W b Ubar (как в (4.49))
        y4_model = W @ (b @ Ubar)
        self.obs.step(y=y, y4_model=y4_model, dt=dt)

        return U.astype(float)
