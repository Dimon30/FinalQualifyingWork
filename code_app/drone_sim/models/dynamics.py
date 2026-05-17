"""Динамическая модель квадрокоптера (уравнения 52-55 диссертации).

Обозначение углов: φ = рысканье (yaw), θ = тангаж (pitch), ψ = крен (roll).
Вектор состояния 16D: [x,y,z, vx,vy,vz, φ,θ,ψ, φ̇,θ̇,ψ̇, u1_bar,ρ1, u2,ρ2].
"""
from __future__ import annotations
import numpy as np

from drone_sim.models.quad_model import QuadModel

G = 9.81  # м/с² — константа для обратной совместимости (старые сценарии Гл. 2)


def thrust_direction(phi: float, theta: float, psi: float) -> np.ndarray:
    """Направление тяги b(φ,θ,ψ) из уравнения (53) диссертации."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cr, sr = np.cos(psi), np.sin(psi)
    return np.array([
        cp * st * cr + sp * sr,
        sp * st * cr - cp * sr,
        ct * cr,
    ], dtype=float)


def sat_tanh(x: float, L: float) -> float:
    """Гладкое насыщение sat_L(x) = L·tanh(x/L)."""
    return float(L * np.tanh(x / max(L, 1e-9)))


def sat_tanh_vec(x: np.ndarray, L: float) -> np.ndarray:
    """Покомпонентное гладкое насыщение."""
    return L * np.tanh(x / max(L, 1e-9))


def quad_dynamics_12(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    """12D-модель Гл. 2: x = [p, v, φθψ, φ̇θ̇ψ̇], U = [u1, u2, u3, u4]."""
    phi, theta, psi = x[6], x[7], x[8]
    u1, u2, u3, u4 = U

    b = thrust_direction(phi, theta, psi)
    a = b * (u1 + G) - np.array([0.0, 0.0, G])

    xdot = np.zeros(12, dtype=float)
    xdot[0:3] = x[3:6]
    xdot[3:6] = a
    xdot[6] = x[9]
    xdot[7] = x[10]
    xdot[8] = x[11]
    xdot[9] = u2
    xdot[10] = u3
    xdot[11] = u4
    return xdot


def quad_dynamics_16(
    x: np.ndarray,
    U: np.ndarray,
    L: float = 5.0,
    model=None,
) -> np.ndarray:
    """16D-модель Глав 3-4 с двойными интеграторами тяги и рысканья.

    U = [v1, v2, u3, u4]; реальная тяга u1 = sat_tanh(u1_bar, L).
    model: QuadModel или None (нормализованная модель mass=1).
    """
    if model is None:
        model = QuadModel()
    g = model.g

    phi, theta, psi = x[6], x[7], x[8]
    u1_bar, rho1, u2, rho2 = x[12], x[13], x[14], x[15]
    v1, v2, u3, u4 = U

    u1 = sat_tanh(u1_bar, L)
    b = thrust_direction(phi, theta, psi)
    a = (b * (u1 + g) - np.array([0.0, 0.0, g])) / model.mass

    # Квадратичное аэродинамическое сопротивление: a_drag = -drag * v * ||v||.
    # Обеспечивает терминальную скорость и предотвращает безграничный разгон
    # при насыщении тяги. Защита от деления на ноль через порог 1e-12.
    drag_coef = getattr(model, "drag", 0.0)
    if drag_coef > 0.0:
        v = x[3:6]
        v_norm = float(np.linalg.norm(v))
        if v_norm > 1e-12:
            a = a - drag_coef * v * v_norm

    xdot = np.zeros(16, dtype=float)
    xdot[0:3] = x[3:6]
    xdot[3:6] = a
    xdot[6] = x[9]
    xdot[7] = x[10]
    xdot[8] = x[11]
    xdot[9] = u2 / model.J_phi
    xdot[10] = u3 / model.J_theta
    xdot[11] = u4 / model.J_psi
    xdot[12] = rho1
    xdot[13] = v1
    xdot[14] = rho2
    xdot[15] = v2
    return xdot
