"""
Глава 2: Алгоритм стабилизации квадрокоптера в заданной точке пространства
(управление по состоянию).

Закон управления (уравнения 30-36 / итоговый закон стр. 20 диссертации):

    u1 = sat_L[(cθcψ)^{-1}(g - k11·ξ̃11 - k21·ξ̃21) - g]
    u2 = -k12·ξ̃12 - k22·ξ̃22
    [u3, u4] = b22^{-1}[q2 + b21·u2 - K3·ξ̃3 - K4·ξ̃4 - K5·ξ̃5 - K6·ξ̃6]

Переменные ошибки (уравнения 23-25):
    ξ̃1 = [z-z*, φ-φ*],  ξ̃2 = [vz, φ̇]
    ξ̃3 = [x-x*, y-y*],  ξ̃4 = [vx, vy]
    ξ̃5 = g·Rz(φ)·[sθcψ, -sψ]   (ускорение в горизонтали)
    ξ̃6 = g·Rz(φ)·(Jac)·[φ̇, ψ̇, θ̇]

Параметры (пример 1 из диссертации стр. 21):
    K3=diag(4,4), K4=diag(6,6), K5=diag(4,4), K6=diag(1,1)
    k11=1, k21=1, k12=1, k22=1
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from dynamics import G, sat_tanh, thrust_direction


@dataclass
class Ch2Params:
    """Параметры регулятора Главы 2."""
    k11: float = 1.0   # коэффициент PD по z (пропорциональный)
    k21: float = 1.0   # коэффициент PD по z (производная)
    k12: float = 1.0   # коэффициент PD по φ
    k22: float = 1.0   # коэффициент PD по φ
    K3: np.ndarray = None   # diag(k31, k32) — коэф. по [x-x*, y-y*]
    K4: np.ndarray = None   # diag(k41, k42) — коэф. по [vx, vy]
    K5: np.ndarray = None   # diag(k51, k52) — коэф. по ξ̃5
    K6: np.ndarray = None   # diag(k61, k62) — коэф. по ξ̃6
    L: float = 5.0           # уровень насыщения для u1
    alpha: float = 0.9       # параметр α (L = α·g) для условия (35)

    def __post_init__(self):
        if self.K3 is None:
            self.K3 = np.diag([4.0, 4.0])
        if self.K4 is None:
            self.K4 = np.diag([6.0, 6.0])
        if self.K5 is None:
            self.K5 = np.diag([4.0, 4.0])
        if self.K6 is None:
            self.K6 = np.diag([1.0, 1.0])


def _b21(phi: float, psi: float, theta: float) -> np.ndarray:
    """Матрица b21(φ,ψ,θ) из уравнения стр. 21 диссертации.

    b21 = g·Rz(φ) @ [[sθcψ], [-sψ]]
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cr, sr = np.cos(psi), np.sin(psi)

    Rz = np.array([[cp, -sp], [sp, cp]], dtype=float)
    inner = np.array([[st * cr], [-sr]], dtype=float)  # 2×1
    # b21 — это 2×1 вектор → но он используется как 2×2 с u2 → u3/u4
    # В диссертации b21(ξ) — матрица 2×1 (только один столбец для u2)
    return G * (Rz @ inner)  # 2×1


def _b22(phi: float, psi: float, theta: float) -> np.ndarray:
    """Матрица b22(φ,ψ,θ) из уравнения стр. 21 диссертации.

    b22 = g·Rz(φ) @ [[cθcψ, -sθsψ], [0, -cψ]]
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cr, sr = np.cos(psi), np.sin(psi)

    Rz = np.array([[cp, -sp], [sp, cp]], dtype=float)
    inner = np.array([[ct * cr, -st * sr],
                      [0.0,     -cr     ]], dtype=float)
    return G * (Rz @ inner)  # 2×2


def _q2(phi: float, psi: float, theta: float,
        phidot: float, psidot: float, thetadot: float) -> np.ndarray:
    """Нелинейный дрейф q2(φ,ψ,θ,φ̇,ψ̇,θ̇) из уравнения стр. 21."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cr, sr = np.cos(psi), np.sin(psi)

    Rz = np.array([[cp, -sp], [sp, cp]], dtype=float)

    M1 = np.array([[-st * cr,    0,          2 * cr   ],
                   [sr,          2 * ct * cr, -2 * st * sr]], dtype=float)
    rates1 = np.array([phidot**2, thetadot * phidot, psidot * phidot], dtype=float)

    M2 = np.array([[-st * cr, -2 * ct * sr, -st * cr],
                   [0,         0,            -cr     ]], dtype=float)
    rates2 = np.array([thetadot**2, thetadot * psidot, psidot**2], dtype=float)

    return G * (Rz @ (M1 @ rates1 + M2 @ rates2))


class Ch2StateController:
    """Регулятор Главы 2: стабилизация в точке (управление по состоянию).

    Измеряет полное состояние x = [x,y,z, vx,vy,vz, φ,θ,ψ, φ̇,θ̇,ψ̇].
    Возвращает U = [u1, u2, u3, u4] для quad_dynamics_12.
    """

    def __init__(self, target: np.ndarray, params: Ch2Params):
        """
        Аргументы:
            target — целевая точка [x*, y*, z*, φ*]
            params — Ch2Params
        """
        self.target = np.array(target, dtype=float)  # [x*, y*, z*, φ*]
        self.p = params

    def step(self, x: np.ndarray) -> np.ndarray:
        """Вычислить управление для текущего состояния x.

        x = [x,y,z, vx,vy,vz, φ,θ,ψ, φ̇,θ̇,ψ̇]
        Возвращает U = [u1, u2, u3, u4]
        """
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        phidot, thetadot, psidot = x[9], x[10], x[11]

        xstar, ystar, zstar = self.target[0], self.target[1], self.target[2]
        phistar = self.target[3] if len(self.target) > 3 else 0.0

        # Переменные ошибки (уравнения 23-25)
        xi1_tilde = np.array([pz - zstar, phi - phistar])   # ξ̃1
        xi2_tilde = np.array([vz, phidot])                   # ξ̃2
        xi3_tilde = np.array([px - xstar, py - ystar])       # ξ̃3
        xi4_tilde = np.array([vx, vy])                       # ξ̃4

        # ξ̃5 = g·Rz(φ)·[sθcψ, -sψ]^T
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cr, sr = np.cos(psi), np.sin(psi)
        Rz2 = np.array([[cp, -sp], [sp, cp]])
        xi5_tilde = G * (Rz2 @ np.array([st * cr, -sr]))

        # ξ̃6 = g·Rz(φ)·Jac·[φ̇, ψ̇, θ̇]
        # Jac из диссертации стр. 20: [[sψ, -sθsψ, cθcψ], [-sθcψ, -cψ, 0]]
        Jac = np.array([[sr,       -st * sr,  ct * cr],
                        [-st * cr, -cr,       0.0    ]], dtype=float)
        xi6_tilde = G * (Rz2 @ (Jac @ np.array([phidot, psidot, thetadot])))

        # u1: контур высоты z и рысканья φ (уравнение 34)
        ct_cr = np.cos(theta) * np.cos(psi)
        ct_cr = max(abs(ct_cr), 1e-3) * np.sign(ct_cr) if abs(ct_cr) > 1e-9 else 1e-3
        raw_u1 = (G - self.p.k11 * xi1_tilde[0] - self.p.k21 * xi2_tilde[0]) / ct_cr - G
        u1 = sat_tanh(raw_u1, self.p.L)

        # u2: рысканье φ (уравнение 33)
        u2 = -self.p.k12 * xi1_tilde[1] - self.p.k22 * xi2_tilde[1]

        # [u3, u4]: горизонтальное движение (уравнение 36)
        b21_val = _b21(phi, psi, theta)  # 2×1
        b22_val = _b22(phi, psi, theta)  # 2×2
        q2_val = _q2(phi, psi, theta, phidot, psidot, thetadot)  # 2×1

        rhs = (q2_val.flatten()
               + (b21_val * u2).flatten()
               - self.p.K3 @ xi3_tilde
               - self.p.K4 @ xi4_tilde
               - self.p.K5 @ xi5_tilde
               - self.p.K6 @ xi6_tilde)

        try:
            u34 = np.linalg.solve(b22_val, rhs)
        except np.linalg.LinAlgError:
            u34 = np.linalg.lstsq(b22_val, rhs, rcond=None)[0]

        return np.array([u1, u2, float(u34[0]), float(u34[1])], dtype=float)
