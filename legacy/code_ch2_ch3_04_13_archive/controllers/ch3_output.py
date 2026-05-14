"""
Глава 3: Динамический алгоритм следящего управления по выходу.

Закон управления (уравнения 48-49 диссертации):

    U = sat_L[b4(φ,ψ,θ,u1)^{-1}(-σ - γ1·ζ̂1 - γ2·ζ̂2 - γ3·ζ̂3 - γ4·ζ̂4)]

с наблюдателем производных (49):
    ζ̂̇1 = ζ̂2 + κa1(ζ1 - ζ̂1)
    ...
    ζ̂̇4 = σ + b4·U + κ⁴a4(ζ1 - ζ̂1)
    σ̇   = κ⁵a5(ζ1 - ζ̂1)

Выходная переменная: ζ1 = col(x, y, z, φ) (положение + рысканье).
Опорная траектория: y*(t) — время-параметрическая (например, спираль).

Вектор состояния: 16D расширенная модель (двойные интеграторы для u1, u2).
Управляющий вектор: U = [v1, v2, u3, u4].

Параметры из диссертации (стр. 28):
    r=0.5, κ=100, a=(5,10,10,5,1), γ=(1,4,6,4,4), L=5, ℓ=0.9
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

from dynamics import G, sat_tanh, sat_tanh_vec, thrust_direction
from controllers.common import HighGainParams, DerivativeObserver4
from trajectories import TrajPoint


def _b4_matrix(phi: float, theta: float, psi: float, u1: float) -> np.ndarray:
    """Матрица b4(φ,θ,ψ,u1) входов 4-й производной выхода (уравнение 47).

    При θ=0, ψ=0:
        b4(0,φ) = [[1, 0,      0,       0    ],
                   [0, 1,      0,       0    ],
                   [0, 0,  g·cφ,    g·sφ],
                   [0, 0,  g·sφ,   -g·cφ]]

    Общая формула вычисляется численно дифференцированием 4-й производной
    выхода y=(x,y,z,φ) по входам U=(v1,v2,u3,u4) через Якобиан.
    Используем аналитическую формулу из структуры модели.

    Структура (из уравнений 46-47):
        Строки 0-2: вклад в (ẍ,ÿ,z̈) через b(φ,θ,ψ)·(u1+g)
        Строка  3:  вклад в φ̈ = u2 через u̇2=ρ2, ρ̈2=v2

    Более точно: 4-я производная выхода аффинна по U:
        ζ̂̇4 = q4(ζ,φ) + b4(ζ,φ)·U

    b4 аналитически вычисляется как Якобиан 4-й производной по U.
    """
    d = u1 + G
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cr, sr = np.cos(psi), np.sin(psi)

    # 4-я производная [x,y,z] по U = [v1,v2,u3,u4]:
    # ẍ = b(φ,θ,ψ)·(u1+g) - g*e3  → 4-я производная по v1:
    #    d^4x/dv1^4 = b · (d^4 u1 / dt^4 частично по v1)
    # У u1_bar: ü1_bar = v1 → u1 = sat(u1_bar) → du1/du1_bar ≈ 1 (малые углы)
    # Для общего случая используем структуру (см. диссертацию, уравнение 46-47):

    # b4 из Утверждения 3 (формула 47):
    # b4(0,φ) — при θ=0,ψ=0; для произвольных углов - численно

    # Аналитически: вклад в позицию (строки 0-2):
    # d^2(v) / d(v1)^2 = b*1 (u1 дважды интегрируется → ρ→u1→v̇=b*(u1+g))
    # Два двойных интегратора: d^4 x / d(v1)^4 через цепочку:
    # v1 → ρ̇1 → u̇1_bar → ... → ẍ, через 2 шага интеграции
    # Итого: d^4[x,y,z]/dv1 = b(φ,θ,ψ)
    #        d^4φ/dv2 = 1 (двойной интегратор u2: v2→ρ2→u̇2→φ̈=u2)
    #        d^4[x,y,z]/du3, du4 через угловые ускорения

    # Полная b4 (аналитическая форма из диссертации стр. 25-26):
    b_thrust = thrust_direction(phi, theta, psi)  # [bx, by, bz]

    # Производные b по θ (для u3=θ̈)
    # ∂b/∂θ:
    db_dtheta = np.array([
        cp * ct * cr,
        sp * ct * cr,
        -st * cr,
    ], dtype=float)

    # Производные b по ψ (для u4=ψ̈)
    # ∂b/∂ψ:
    db_dpsi = np.array([
        cp * st * (-sr) + sp * cr,
        sp * st * (-sr) - cp * cr,
        ct * (-sr),
    ], dtype=float)

    # b4 = [[b_x, 0, d*db_dθ_x, d*db_dψ_x],
    #        [b_y, 0, d*db_dθ_y, d*db_dψ_y],
    #        [b_z, 0, d*db_dθ_z, d*db_dψ_z],
    #        [0,   1, 0,          0         ]]
    B = np.zeros((4, 4), dtype=float)
    B[0:3, 0] = b_thrust          # столбец v1
    B[3, 1] = 1.0                  # столбец v2 → φ̈
    B[0:3, 2] = d * db_dtheta     # столбец u3 → θ̈ → ẍ
    B[0:3, 3] = d * db_dpsi       # столбец u4 → ψ̈ → ẍ

    return B


def _q4_drift(state: np.ndarray, L: float) -> np.ndarray:
    """Нелинейный дрейф q4(ζ,φ) в 4-й производной выхода.

    Вычисляется численным дифференцированием: q4 = d^4y/dt^4 - b4·U
    при U=0. Использует конечные разности для оценки третьей производной
    тяговой функции.
    """
    phi, theta, psi = state[6], state[7], state[8]
    phidot, thetadot, psidot = state[9], state[10], state[11]
    u1_bar, rho1, u2, rho2 = state[12], state[13], state[14], state[15]

    u1 = sat_tanh(u1_bar, L)
    d = u1 + G

    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cr, sr = np.cos(psi), np.sin(psi)

    b = thrust_direction(phi, theta, psi)

    # ḃ = ∂b/∂φ·φ̇ + ∂b/∂θ·θ̇ + ∂b/∂ψ·ψ̇
    db_dphi   = np.array([-sp*st*cr + cp*sr,  cp*st*cr + sp*sr, 0.0])
    db_dtheta = np.array([ cp*ct*cr,           sp*ct*cr,        -st*cr])
    db_dpsi   = np.array([-cp*st*sr + sp*cr,  -sp*st*sr - cp*cr, -ct*sr])

    bdot = db_dphi*phidot + db_dtheta*thetadot + db_dpsi*psidot

    # ḋ = ρ1 (производная u1_bar → u1: ρ̇1 * sat'(u1_bar))
    sat_prime = 1.0 / np.cosh(u1_bar / max(L, 1e-9))**2
    u1dot = rho1 * sat_prime

    # b̈ — вторая производная (учитываем только члены без U)
    # Упрощённо через конечные разности b по углам
    h = 1e-4
    def b_at(phi_h, theta_h, psi_h):
        return thrust_direction(phi_h, theta_h, psi_h)

    bddot_pos = np.zeros(3)  # вклад от угловых скоростей
    for i, (ang, angdot, h_vec) in enumerate(zip(
        [phi, theta, psi],
        [phidot, thetadot, psidot],
        [(h, 0, 0), (0, h, 0), (0, 0, h)]
    )):
        bp = b_at(phi + h_vec[0], theta + h_vec[1], psi + h_vec[2])
        bm = b_at(phi - h_vec[0], theta - h_vec[1], psi - h_vec[2])
        d2b_di2 = (bp - 2*b + bm) / h**2
        bddot_pos += d2b_di2 * angdot**2

    bddot = bddot_pos  # перекрёстные члены малы (пренебрегаем при θ,ψ малых)

    # Дрейф 4-й производной (члены без U):
    # d^4[x,y,z]/dt^4 = bⁿ·(u1+g) + ... ≈ b̈·d + 2·ḃ·ḋ + b·ü1
    # Для φ: d^4φ/dt^4 = d²u2/dt² = ρ̇2 (но v2=0) → 0
    # Полная формула сложная, для практики используем обнуление (наблюдатель компенсирует)
    q = np.zeros(4, dtype=float)
    q[0:3] = bddot * d + 2.0 * bdot * u1dot  # приближение
    q[3] = 0.0  # рысканье: φ̈ = u2, дрейф в u2-канале → 0 при v2=0

    return q


class Ch3OutputTrackingController:
    """Регулятор Главы 3: следящее управление по выходу (16D модель).

    Параметры (стр. 28 диссертации):
        r=0.5, κ=100, a=(5,10,10,5,1), γ=(1,4,6,4,4), L=5
    """

    def __init__(self, params: HighGainParams):
        self.p = params
        self.obs = DerivativeObserver4(dim=4, p=params)

    def step(
        self, t: float, state: np.ndarray, ref: TrajPoint, dt: float
    ) -> np.ndarray:
        """Один шаг регулятора.

        Аргументы:
            t      — текущее время
            state  — вектор состояния 16D
            ref    — опорная точка TrajPoint
            dt     — шаг времени

        Возвращает U = [v1, v2, u3, u4].
        """
        # Измеряемый выход ζ1 = [x, y, z, φ]
        y = np.array([state[0], state[1], state[2], state[6]], dtype=float)

        # Матрица входов b4 и дрейф q4
        phi, theta, psi = state[6], state[7], state[8]
        u1_bar = state[12]
        u1 = sat_tanh(u1_bar, self.p.L)

        b4 = _b4_matrix(phi, theta, psi, u1)

        # Сначала получаем оценки наблюдателя (от предыдущего шага)
        z1h, z2h, z3h, z4h, sigma = self.obs.hat()

        # Закон управления (уравнение 48):
        # U = sat_L[b4^{-1}(-σ - γ1(ζ̂1-y*) - γ2(ζ̂2-ẏ*) - γ3(ζ̂3-ÿ*) - γ4(ζ̂4-y*⁽³⁾))]
        g1, g2, g3, g4 = self.p.gamma[0], self.p.gamma[1], self.p.gamma[2], self.p.gamma[3]
        v = (-sigma
             - g1 * (z1h - ref.y)
             - g2 * (z2h - ref.y1)
             - g3 * (z3h - ref.y2)
             - g4 * (z4h - ref.y3))

        try:
            U_raw = np.linalg.solve(b4, v)
        except np.linalg.LinAlgError:
            U_raw = np.linalg.lstsq(b4, v, rcond=None)[0]

        U = sat_tanh_vec(U_raw, self.p.L)

        # Обновление наблюдателя (уравнение 49): y4_model = b4·U (известный член)
        # σ адаптируется к дрейфу q4(ζ,φ) — неизвестной нелинейности
        self.obs.step(y=y, y4_model=b4 @ U, dt=dt)

        return U
