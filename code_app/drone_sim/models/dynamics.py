"""
Динамическая модель квадрокоптера.

Уравнения движения в соответствии с диссертацией (уравнения 52-55):

    ṗ = v
    v̇ = (b(φ,θ,ψ) * (u1 + g) - [0, 0, g]) / mass
    φ̈ = u2 / J_phi
    θ̈ = u3 / J_theta
    ψ̈ = u4 / J_psi

Физические параметры задаются через QuadModel (models/quad_model.py).
При QuadModel() (defaults: mass=1, J=1, g=9.81) уравнения совпадают
с нормализованной моделью диссертации.

Соглашение об обозначении углов (нестандартное, как в диссертации):
    φ = рысканье (yaw)
    θ = тангаж (pitch)
    ψ = крен (roll)

Вектор состояния расширенной модели (16-мерный):
    x = [x, y, z, vx, vy, vz, φ, θ, ψ, φ̇, θ̇, ψ̇, u1_bar, ρ1, u2, ρ2]

u1 и u2 — выходы двойных интеграторов:
    u̇1 = ρ1,  ρ̇1 = v1
    u̇2 = ρ2,  ρ̇2 = v2

Физическая тяга: u1_phys = sat_L(u1_bar)
Управляющий вектор: U = [v1, v2, u3, u4]
"""
from __future__ import annotations
import numpy as np

from drone_sim.models.quad_model import QuadModel

G = 9.81  # ускорение свободного падения, м/с² (константа для обратной совместимости)


def thrust_direction(phi: float, theta: float, psi: float) -> np.ndarray:
    """Направление тяги b(φ,θ,ψ) из уравнения (53) диссертации.

    b = [cφ·sθ·cψ + sφ·sψ,
         sφ·sθ·cψ - cφ·sψ,
         cθ·cψ]

    где φ=рысканье, θ=тангаж, ψ=крен.
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cr, sr = np.cos(psi), np.sin(psi)
    return np.array([
        cp * st * cr + sp * sr,
        sp * st * cr - cp * sr,
        ct * cr,
    ], dtype=float)


def sat_tanh(x: float, L: float) -> float:
    """Гладкое насыщение: sat_L(x) = L·tanh(x/L)."""
    return float(L * np.tanh(x / max(L, 1e-9)))


def sat_tanh_vec(x: np.ndarray, L: float) -> np.ndarray:
    """Покомпонентное гладкое насыщение."""
    return L * np.tanh(x / max(L, 1e-9))


# ---------------------------------------------------------------------------
# Модель для Главы 2: 12-мерное состояние (без двойных интеграторов)
# x = [x, y, z, vx, vy, vz, φ, θ, ψ, φ̇, θ̇, ψ̇]
# U = [u1, u2, u3, u4]  (u1 — виртуальная тяга без насыщения)
# ---------------------------------------------------------------------------

def quad_dynamics_12(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    """ОДУ квадрокоптера, 12D. Для Главы 2 (управление по состоянию).

    x = [x,y,z, vx,vy,vz, φ,θ,ψ, φ̇,θ̇,ψ̇]
    U = [u1, u2, u3, u4]
    """
    phi, theta, psi = x[6], x[7], x[8]
    u1, u2, u3, u4 = U

    b = thrust_direction(phi, theta, psi)
    a = b * (u1 + G) - np.array([0.0, 0.0, G])

    xdot = np.zeros(12, dtype=float)
    xdot[0:3] = x[3:6]        # ṗ = v
    xdot[3:6] = a              # v̇ = b*(u1+g) - [0,0,g]
    xdot[6] = x[9]             # φ̇
    xdot[7] = x[10]            # θ̇
    xdot[8] = x[11]            # ψ̇
    xdot[9] = u2               # φ̈ = u2
    xdot[10] = u3              # θ̈ = u3
    xdot[11] = u4              # ψ̈ = u4
    return xdot


# ---------------------------------------------------------------------------
# Модель для Глав 3-4: 16-мерное состояние (с двойными интеграторами)
# x = [x,y,z, vx,vy,vz, φ,θ,ψ, φ̇,θ̇,ψ̇, u1_bar,ρ1, u2,ρ2]
# U = [v1, v2, u3, u4]
# ---------------------------------------------------------------------------

def quad_dynamics_16(
    x: np.ndarray,
    U: np.ndarray,
    L: float = 5.0,
    model=None,  # QuadModel | None
) -> np.ndarray:
    """ОДУ квадрокоптера, 16D. Согласованное управление по выходу.

    x = [x,y,z, vx,vy,vz, φ,θ,ψ, φ̇,θ̇,ψ̇, u1_bar,ρ1, u2,ρ2]
    U = [v1, v2, u3, u4]
    L     -- уровень насыщения
    model -- физические параметры дрона (QuadModel); None → нормализованная модель
    """
    if model is None:
        model = QuadModel()
    g = model.g

    phi, theta, psi = x[6], x[7], x[8]
    u1_bar, rho1, u2, rho2 = x[12], x[13], x[14], x[15]
    v1, v2, u3, u4 = U

    u1 = sat_tanh(u1_bar, L)    # физическая тяга (с насыщением)
    b = thrust_direction(phi, theta, psi)
    a = (b * (u1 + g) - np.array([0.0, 0.0, g])) / model.mass

    xdot = np.zeros(16, dtype=float)
    xdot[0:3] = x[3:6]                    # ṗ = v
    xdot[3:6] = a                          # v̇ = (b*(u1+g) - [0,0,g]) / mass
    xdot[6] = x[9]                         # φ̇
    xdot[7] = x[10]                        # θ̇
    xdot[8] = x[11]                        # ψ̇
    xdot[9] = u2 / model.J_phi             # φ̈ = u2 / J_phi
    xdot[10] = u3 / model.J_theta          # θ̈ = u3 / J_theta
    xdot[11] = u4 / model.J_psi            # ψ̈ = u4 / J_psi
    xdot[12] = rho1                        # u̇1_bar = ρ1
    xdot[13] = v1                          # ρ̇1 = v1
    xdot[14] = rho2                        # u̇2 = ρ2
    xdot[15] = v2                          # ρ̇2 = v2
    return xdot
