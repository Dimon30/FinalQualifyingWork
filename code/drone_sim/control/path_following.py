"""
path_following.py
=================
Регулятор согласованного управления движением вдоль заданной траектории.

Закон управления по выходу (уравнения 71-77 диссертации Ким С.А., 2024):

    u1 = sat_L(ū1)
    [v̄1, v2, u3, u4] = γ5·η + Ū,   η̇ = Ū
    Ū = sat_L[b^{-1}(θ,ψ,u1,φ)·W^{-1}(-σ - γ1·λ̂1 - γ2·λ̂2 - γ3·λ̂3 - γ4·λ̂4)]

с наблюдателем (73-77):
    λ̂̇1 = λ̂2 + κa1(λ̃1 - λ̂1)
    λ̂̇2 = λ̂3 + κ²a2(λ̃1 - λ̂1)
    λ̂̇3 = λ̂4 + κ³a3(λ̃1 - λ̂1)
    λ̂̇4 = σ + W·b·Ū + κ⁴a4(λ̃1 - λ̂1)
    σ̇   = κ⁵a5(λ̃1 - λ̂1)

Регулируемые переменные (уравнение 60):
    λ̃1 = col(s - V*t, e1, e2, δφ)

где s — параметр ближайшей точки на кривой S,
    e1, e2 — боковые отклонения (из геометрического преобразования),
    δφ = φ - φ*(s) — ошибка рысканья.

Матрица W (уравнение из стр. 38):
    W(α,β) = [[cα·cβ,  sα·cβ,  sβ,   0],
              [-sα,    cα,     0,    0],
              [-cα·sβ, -sα·sβ, cβ,   0],
              [-ε·cα·cβ, -ε·sα·cβ, -ε·sβ, 1]]

Матрица b (уравнение из стр. 38):
    b = Rz(φ) × B_inner(θ,ψ,u1)

где B_inner — 4×4 матрица с колонками [v1, v2, u3, u4].

Параметры (стр. 41 / 44 диссертации):
    Прямая: κ=100, a=(5,10,10,5,1), γ=(1,3,5,3,1), γ5=1, L=5, ℓ=0.9
    Спираль: κ=200, те же a и γ
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
    """Матрица W(α,β,ε) (стр. 38 диссертации).

    W = [[cα·cβ,     sα·cβ,     sβ,   0],
         [-sα,       cα,        0,    0],
         [-cα·sβ,   -sα·sβ,    cβ,   0],
         [-ε·cα·cβ, -ε·sα·cβ, -ε·sβ, 1]]
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array([
        [ca*cb,      sa*cb,      sb,   0.0],
        [-sa,         ca,        0.0,  0.0],
        [-ca*sb,     -sa*sb,     cb,   0.0],
        [-eps*ca*cb, -eps*sa*cb, -eps*sb, 1.0],
    ], dtype=float)


def W_inv(alpha: float, beta: float, eps: float) -> np.ndarray:
    """Обратная матрица W^{-1}(α,β,ε) (стр. 38 диссертации)."""
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
    """Матрица b(θ,ψ,u1,φ) = Rz(φ) × B_inner(θ,ψ,u1) (стр. 38 диссертации).

    B_inner (колонки: v1, v2, u3, u4):
        Col v1: [sθcψ,  -sψ,  cθcψ,  0]
        Col v2: [0,      0,   0,     1]
        Col u3: d·[cθcψ,     0,    -sθcψ,     0]
        Col u4: d·[-sθsψ,   -cψ,   -cθsψ,    0]

    где d = u1 + g.

    Параметры:
        g -- ускорение свободного падения [м/с²]; None → G=9.81 (обратная совместимость)
    """
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
    """Устойчивое обращение 4×4 матрицы (регуляризация Тихонова при плохой обусловленности)."""
    try:
        cond = np.linalg.cond(M)
        if not np.isfinite(cond) or cond > 1e8:
            raise np.linalg.LinAlgError("плохая обусловленность")
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.inv(M + eps * np.eye(4))


class Ch4PathController:
    """Регулятор согласованного управления по выходу (16D модель).

    Движение вдоль гладкой пространственной кривой S со скоростью V*.

    Параметры из диссертации:
        Прямая:  κ=100,  a=(5,10,10,5,1), γ=(1,3,5,3,1), γ5=1, L=5
        Спираль: κ=200,  a=(5,10,10,5,1), γ=(1,3,5,3,1), γ5=1, L=5
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
        """
        Аргументы:
            curve               — геометрия кривой S
            Vstar               — желаемая скорость движения V*
            params              — HighGainParams
            use_spiral_observer — True → использовать динамический наблюдатель
                                  ближайшей точки (для спирали, Лемма 3)
            r                   — радиус спирали (только при use_spiral_observer=True)
            gamma_nearest       — коэффициент γ наблюдателя ближайшей точки
            quad_model          — физические параметры дрона (QuadModel);
                                  None → нормализованная модель (mass=1, J=1, g=9.81)
        """
        self.curve = curve
        self.Vstar = float(Vstar)
        self.p = params
        self.use_spiral_observer = use_spiral_observer
        self.r = float(r)
        self.gamma_nearest = float(gamma_nearest)
        self._model = quad_model if quad_model is not None else QuadModel()

        # Параметр ближайшей точки (оценка)
        self._zeta = 0.0

        # Динамическая компонента η (η̇ = Ū)
        self._eta = np.zeros(4, dtype=float)

        # Наблюдатель производных λ̃1 ∈ R^4
        self.obs = DerivativeObserver4(dim=4, p=params)

    def _nearest_s(self, p_xyz: np.ndarray, dt: float) -> float:
        """Оценить параметр ближайшей точки на кривой."""
        if self.use_spiral_observer:
            self._zeta = spiral_nearest_observer_step(
                self._zeta, p_xyz, r=self.r, gamma=self.gamma_nearest, dt=dt
            )
            return float(self._zeta)
        return float(nearest_point_line(p_xyz))

    def _lambda_tilde_1(
        self, t: float, p_xyz: np.ndarray, phi: float, s: float
    ) -> np.ndarray:
        """Вектор регулируемых переменных λ̃1 = col(s_arc - V*t, e1, e2, δφ).

        Уравнение (60) и стр. 37 диссертации.
        s_arc = ζ · ||t(ζ)|| — длина дуги от начала (для кривых с постоянной
        нормой касательной: прямая ||t||=√3, спираль ||t||=√(r²+1)).
        Условие (56): ṡ → V*, где ṡ = unit_tangent · v = d(s_arc)/dt.
        """
        _, e1, e2 = se_from_pose(p_xyz, s, self.curve)
        phi_star = float(self.curve.yaw_star(s))
        d_phi = float(np.arctan2(np.sin(phi - phi_star), np.cos(phi - phi_star)))
        # Масштабирование параметра в длину дуги
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
        """Один шаг регулятора.

        Аргументы:
            t      — текущее время
            x      — вектор состояния 16D
            Uprev  — предыдущий вектор управления (не используется)
            dt     — шаг времени

        Возвращает U = [v1, v2, u3, u4].
        """
        p_xyz = x[0:3]
        phi, theta, psi = float(x[6]), float(x[7]), float(x[8])
        u1_bar = float(x[12])
        u1 = sat_tanh(u1_bar, self.p.L)

        # Ближайшая точка на кривой
        s = self._nearest_s(p_xyz, dt)

        # Геометрия кривой в точке s
        alpha = float(self.curve.yaw_star(s))
        beta_val = float(self.curve.beta(s))
        eps_val = float(self.curve.eps(s))

        W = W_mat(alpha, beta_val, eps_val)
        Winv = W_inv(alpha, beta_val, eps_val)

        # Измеряемый выход λ̃1
        lam1 = self._lambda_tilde_1(t, p_xyz, phi, s)

        # Матрица входов b (с g из QuadModel)
        b = b_mat(phi, theta, psi, u1, g=self._model.g)
        binv = _safe_inv4(b)

        # Оценки наблюдателя
        l1h, l2h, l3h, l4h, sigma = self.obs.hat()
        g1, g2, g3, g4 = (
            self.p.gamma[0], self.p.gamma[1],
            self.p.gamma[2], self.p.gamma[3]
        )
        g5 = self.p.gamma[4]

        # Закон управления Ū (уравнение 72):
        # Ū = sat_L[b^{-1} W^{-1} (-σ - γ1·λ̂1 - γ2·λ̂2 - γ3·λ̂3 - γ4·λ̂4)]
        v = -sigma - g1*l1h - g2*l2h - g3*l3h - g4*l4h
        Ubar = sat_tanh_vec(binv @ (Winv @ v), self.p.L)

        # Динамическая компонента (уравнение 71): U = γ5·η + Ū,  η̇ = Ū
        self._eta += dt * Ubar
        U = g5 * self._eta + Ubar

        # Обновление наблюдателя (уравнение 76):
        # y4_model = W · b · Ū
        y4_model = W @ (b @ Ubar)
        self.obs.step(y=lam1, y4_model=y4_model, dt=dt)

        return U.astype(float)

    def reset(self):
        """Сбросить внутреннее состояние регулятора."""
        self._zeta = 0.0
        self._eta[:] = 0.0
        self.obs.reset()
