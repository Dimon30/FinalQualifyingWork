"""
path_sim.py
===========
Согласованное управление квадрокоптером по произвольной гладкой кривой.

Реализует алгоритм Главы 4 диссертации Ким С.А. (2024) в виде удобного API.
Основная идея: дрон движется вдоль пространственной кривой S с желаемой
параметрической скоростью V* и нулевыми боковыми ошибками e1, e2.

Математика (уравнения 56-77 диссертации):
    Регулируемые переменные:
        lambda_1 = col(s_arc - V*t, e1, e2, delta_phi)
        s_arc = zeta * ||t(zeta)||  -- длина дуги (точно для ||t|| = const)
        e1, e2  -- боковые ошибки в системе координат Френе
        delta_phi = phi - phi*(zeta) -- ошибка рысканья

    Наблюдатель ближайшей точки (обобщение Леммы 3):
        H(zeta, x) = (p(zeta) - x_drone) . t(zeta)
        zeta_dot = -gamma * sign(dH/dzeta) * H(zeta, x)

    Закон управления (уравнения 71-72):
        U_bar = sat_L[ b^{-1} W^{-1} (-sigma - sum_i gamma_i * lambda_hat_i) ]
        U = gamma5 * eta + U_bar,  eta_dot = U_bar

    Наблюдатель производных 4-го порядка (уравнения 73-77):
        lambda_hat_dot_1 = lambda_hat_2 + kappa*a1*(lambda_1 - lambda_hat_1)
        ...
        sigma_dot = kappa^5 * a5 * (lambda_1 - lambda_hat_1)

Пример использования::

    import numpy as np
    from drone_sim import make_curve, SimConfig, simulate_path_following

    # Эллиптическая спираль: a=3, b=2, подъём 0.5 рад/м
    curve = make_curve(lambda s: np.array([3*np.cos(s), 2*np.sin(s), 0.5*s]))

    cfg = SimConfig(Vstar=1.0, T=25.0, dt=0.002, kappa=200)
    result = simulate_path_following(curve, cfg)

    result.print_summary()
    result.plot("out_images/elliptic_spiral")
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Optional

from drone_sim.models.dynamics import quad_dynamics_16, sat_tanh, sat_tanh_vec, G
from drone_sim.models.quad_model import QuadModel
from drone_sim.control.common import HighGainParams, DerivativeObserver4
from drone_sim.control.path_following import W_mat, W_inv, b_mat, _safe_inv4
from drone_sim.geometry.curves import CurveGeom, se_from_pose
from drone_sim.visualization.plotting import display_path
from drone_sim.simulation.runner import simulate as _simulate_raw

__all__ = [
    "QuadModel",
    "make_curve",
    "NearestPointObserver",
    "PathFollowingController",
    "SimConfig",
    "SimResult",
    "simulate_path_following",
]


# ===========================================================================
# 1. Построение геометрии кривой из параметрической функции
# ===========================================================================

def make_curve(p_fn: Callable[[float], np.ndarray], h: float = 1e-5) -> CurveGeom:
    """Создать геометрию кривой из параметрической функции p(s).

    Параметры:
        p_fn  -- функция p(s) -> np.ndarray[3], задающая кривую S.
                 Принимает скаляр s (параметр), возвращает точку в R^3.
                 Примеры:
                   lambda s: np.array([np.cos(s), np.sin(s), 0.2*s])  # спираль
                   lambda s: np.array([s, s**2, s])                    # парабола
        h     -- шаг для численного дифференцирования (по умолчанию 1e-5)

    Возвращает:
        CurveGeom -- объект геометрии кривой для регулятора и визуализации

    Вычисляемые геометрические характеристики:
        t(s)       = dp/ds  (касательный вектор, центральная разность)
        yaw_star(s)= atan2(ty, tx)  (угол рысканья касательной)
        beta(s)    = atan2(tz, sqrt(tx^2+ty^2))  (угол тангажа касательной)
        eps(s)     = d(yaw_star)/ds  (кривизна проекции XY)

    Замечание:
        Функция p_fn должна быть дважды непрерывно дифференцируемой.
        Разрывы или резкие изломы кривой приведут к ошибочным eps(s).
    """
    def p(s: float) -> np.ndarray:
        return np.asarray(p_fn(float(s)), dtype=float)

    def t(s: float) -> np.ndarray:
        """Касательный вектор dp/ds (центральная конечная разность)."""
        return (p(s + h) - p(s - h)) / (2.0 * h)

    def yaw_star(s: float) -> float:
        """Угол рысканья касательной alpha(s) = atan2(ty, tx)."""
        tv = t(s)
        return float(np.arctan2(tv[1], tv[0]))

    def beta(s: float) -> float:
        """Угол тангажа касательной beta(s) = atan2(tz, sqrt(tx^2+ty^2))."""
        tv = t(s)
        return float(np.arctan2(tv[2], np.sqrt(tv[0]**2 + tv[1]**2)))

    def eps(s: float) -> float:
        """Кривизна проекции кривой на XY: eps(s) = d(alpha)/ds."""
        a1 = yaw_star(s - h)
        a2 = yaw_star(s + h)
        # Нормализованная разность углов (избегаем 2pi-разрывов)
        da = float(np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1)))
        return da / (2.0 * h)

    return CurveGeom(p=p, t=t, yaw_star=yaw_star, beta=beta, eps=eps)


# ===========================================================================
# 2. Наблюдатель ближайшей точки на произвольной кривой
# ===========================================================================

class NearestPointObserver:
    """Динамический наблюдатель ближайшей точки на произвольной кривой.

    Обобщение Леммы 3 (диссертация, стр. 43) на произвольные кривые.

    Условие ближайшей точки:  H(zeta, x) = (p(zeta) - x) . t(zeta) = 0
    Закон наблюдателя:  zeta_dot = -gamma * rho * H(zeta, x)
                        rho = sign(dH/dzeta)
    dH/dzeta = ||t(zeta)||^2 + (p(zeta) - x) . t'(zeta)

    При numerical_grad=False используется приближение dH/dzeta ~ ||t||^2,
    что точно вблизи кривой (второй член мал) и даёт rho = +1.

    Параметры:
        curve          -- геометрия кривой (CurveGeom)
        gamma          -- коэффициент сходимости (gamma > 0)
        zeta0          -- начальное значение параметра
        numerical_grad -- True: точное dH/dzeta (численно)
                          False: быстрое приближение ~||t||^2
        h_grad         -- шаг для численного градиента (только при numerical_grad=True)
    """

    def __init__(
        self,
        curve: CurveGeom,
        gamma: float = 1.0,
        zeta0: float = 0.0,
        numerical_grad: bool = False,
        h_grad: float = 1e-4,
    ):
        self.curve = curve
        self.gamma = float(gamma)
        self._zeta = float(zeta0)
        self.numerical_grad = numerical_grad
        self._h = float(h_grad)

    @property
    def zeta(self) -> float:
        """Текущая оценка параметра ближайшей точки."""
        return self._zeta

    def step(self, p_xyz: np.ndarray, dt: float) -> float:
        """Один шаг интегрирования наблюдателя (явный метод Эйлера).

        Аргументы:
            p_xyz -- положение дрона [x, y, z]
            dt    -- шаг времени

        Возвращает:
            zeta -- обновлённая оценка параметра ближайшей точки
        """
        z = self._zeta
        p_c = self.curve.p(z)
        t_c = self.curve.t(z)

        H = float(np.dot(p_c - p_xyz, t_c))

        if self.numerical_grad:
            h = self._h
            H_p = float(np.dot(self.curve.p(z + h) - p_xyz, self.curve.t(z + h)))
            H_m = float(np.dot(self.curve.p(z - h) - p_xyz, self.curve.t(z - h)))
            dH = (H_p - H_m) / (2.0 * h)
        else:
            # Приближение: dH/dzeta ~ ||t||^2 (доминирует вблизи кривой)
            dH = float(np.dot(t_c, t_c))

        rho = float(np.sign(dH)) if abs(dH) > 1e-12 else 1.0
        self._zeta = float(z - self.gamma * rho * H * dt)
        return self._zeta

    def reset(self, zeta0: float = 0.0) -> None:
        """Сбросить состояние наблюдателя."""
        self._zeta = float(zeta0)


# ===========================================================================
# 3. Контроллер согласованного управления для произвольной кривой
# ===========================================================================

class PathFollowingController:
    """Регулятор согласованного управления по выходу для произвольной кривой.

    Реализует алгоритм Главы 4 (уравнения 71-77 диссертации) в виде,
    совместимом с любой кривой, созданной через make_curve() или CurveGeom.

    Использует:
        - NearestPointObserver (по умолчанию) или nearest_fn для ближайшей точки
        - DerivativeObserver4 для оценки производных вектора ошибок
        - b_mat, W_mat/W_inv из control/path_following.py
        - 16-мерную модель квадрокоптера

    Параметры:
        curve             -- геометрия кривой S
        Vstar             -- желаемая скорость (параметрическая); начальное значение V_ref
        params            -- HighGainParams (kappa, a, gamma, L, ell)
        gamma_nearest     -- коэффициент наблюдателя ближайшей точки (NearestPointObserver)
        zeta0             -- начальное значение параметра
        use_numerical_grad -- True: точный gradH для NearestPointObserver
        nearest_fn        -- опциональная аналитическая функция ближайшей точки:
                             nearest_fn(p_xyz: np.ndarray) -> float
                             Если задана — используется вместо NearestPointObserver.
                             Пример: nearest_fn=nearest_point_line для прямой x=s,y=s,z=s.
        speed_fn          -- опциональная функция адаптивного выбора скорости:
                             speed_fn(state: np.ndarray, s: float) -> float
                             Если None — используется постоянная скорость Vstar (fallback).
                             Пример: lambda x, s: predictor.predict(feature_vector(x, curve, s=s))
    """

    def __init__(
        self,
        curve: CurveGeom,
        Vstar: float,
        params: HighGainParams,
        gamma_nearest: float = 1.0,
        zeta0: float = 0.0,
        use_numerical_grad: bool = False,
        nearest_fn: Optional[Callable] = None,
        quad_model: Optional[QuadModel] = None,
        speed_fn: Optional[Callable] = None,
        warmup_time: float = 5.0,
    ):
        self.curve = curve
        self.Vstar = float(Vstar)          # текущий V_ref (обновляется если speed_fn задана)
        self.p = params
        self._nearest_fn = nearest_fn
        self._model = quad_model if quad_model is not None else QuadModel()
        self._speed_fn = speed_fn          # None → постоянная скорость (fallback)
        self._prev_V: float = float(Vstar) # предыдущий V_ref для ограничения ускорения
        self._Vstar_base: float = float(Vstar)  # baseline (безопасный) уровень скорости
        self._warmup_time: float = float(warmup_time)    # пока t < warmup_time — NN не вызывается
        self._vstar_max_rate: float = 0.3               # переопределяется через SimConfig
        self._cooldown_steps: int = 0    # шаги восстановления после аварии (Level 3)

        self._nearest = NearestPointObserver(
            curve=curve,
            gamma=gamma_nearest,
            zeta0=zeta0,
            numerical_grad=use_numerical_grad,
        )
        self._eta = np.zeros(4, dtype=float)
        self.obs = DerivativeObserver4(dim=4, p=params)

        # Накопленная длина дуги: s_arc = ∫₀^ζ ||t(τ)|| dτ
        # Корректна для любой параметризации (в т.ч. неравномерной эллипс, парабола…).
        # Вычисляется инкрементально методом средней точки O(dζ²) за шаг.
        self._s_arc: float = 0.0
        self._prev_zeta: float = float(zeta0)

        # Ссылочная длина дуги: s_ref = ∫₀ᵗ V*(τ) dτ (интегратор V*).
        # Используется вместо V*·t — обеспечивает плавность lam1[0] при изменении V*.
        # При постоянном V*: s_ref(t) = V* · t (точно, обратная совместимость).
        self._s_ref: float = 0.0

    def _lambda_tilde_1(
        self, p_xyz: np.ndarray, phi: float, s: float
    ) -> np.ndarray:
        """Вектор регулируемых переменных lambda_tilde_1.

        lambda_tilde_1 = col(s_arc - s_ref, e1, e2, delta_phi)
        s_ref = ∫₀ᵗ V*(τ) dτ — ссылочная длина дуги (плавная при изменении V*).
        При постоянном V*: s_ref = V* · t (обратная совместимость).
        """
        _, e1, e2 = se_from_pose(p_xyz, s, self.curve)
        phi_star = float(self.curve.yaw_star(s))
        d_phi = float(np.arctan2(np.sin(phi - phi_star), np.cos(phi - phi_star)))
        return np.array([self._s_arc - self._s_ref, e1, e2, d_phi], dtype=float)

    def step(
        self,
        t: float,
        x: np.ndarray,
        Uprev: Optional[np.ndarray],
        dt: float,
    ) -> np.ndarray:
        """Один шаг регулятора. Возвращает U = [v1, v2, u3, u4]."""
        p_xyz = x[0:3]
        phi = float(x[6])
        theta = float(x[7])
        psi = float(x[8])
        u1_bar = float(x[12])
        u1 = sat_tanh(u1_bar, self.p.L)

        # Шаг 1: ближайшая точка на кривой
        if self._nearest_fn is not None:
            s = float(self._nearest_fn(p_xyz))
            self._nearest._zeta = s
        else:
            s = self._nearest.step(p_xyz, dt)

        # Адаптивное обновление V_ref (если задан speed_fn)
        # Fallback: speed_fn is None → self.Vstar остаётся константой
        if self._speed_fn is not None and t >= self._warmup_time:
            try:
                V_nn = float(self._speed_fn(x, s))
                if not np.isfinite(V_nn):
                    V_nn = self._prev_V

                # --- Диагностические величины ---
                _, _, e2_curr = se_from_pose(p_xyz, s, self.curve)
                e2_ratio = abs(e2_curr) / max(self._model.lateral_error_limit, 1e-9)
                # Рассогласование дуг: s_arc (∫||t||dζ) и s_ref (∫V*dt) — обе в метрах.
                # В равновесии s_arc ≈ s_ref. Отрицательное → дрон отстаёт.
                sarc_err = self._s_arc - self._s_ref

                # --- Четырёхуровневая защита ---

                # Уровень 0: физический предел на основе геометрии кривой.
                # В установившемся режиме: v_body ≈ V* (arc-speed), ζ̇ = V*/||t(ζ)||.
                # Ограничиваем V* так, чтобы ζ̇ ≤ max_speed/||t||²:
                #   max_safe_vstar = max_speed / ||t|| × 1.1
                # Для спирали r=3 (||t||=√10, max_speed=10): cap ≈ 3.47 ≈ vstar_cap=3.5 ✓
                # Для helix r=2 (||t||=3): cap ≈ 3.67 (безопаснее прежнего cap=4.0 ✓)
                # Для прямой (||t||=√3): cap ≈ 6.35 (без ограничения, ≡ cap=None ✓)
                # Использует max_speed (=10 для всех моделей), НЕ max_velocity_norm
                # (max_velocity_norm у RL-моделей=6.0 из обучения, что дало бы cap=2.09).
                t_norm_curr = float(np.linalg.norm(self.curve.t(s)))
                max_safe_vstar = self._model.max_speed / max(t_norm_curr, 1e-3) * 1.1
                V_nn = min(V_nn, max_safe_vstar)

                # Уровень 1: значительное отставание дуги → вернуть к baseline.
                # Дрон физически не может держать скорость. Порог 1.5 м выбран так:
                # при rate=0.3 м/с², переходный процесс даёт дефицит ~0.3-0.4 м (не срабатывает).
                # При настоящей нестабильности дефицит растёт до >1 м и быстро нарастает.
                if sarc_err < -1.5:
                    V_nn = min(V_nn, self._Vstar_base)

                # Уровень 2: умеренная поперечная ошибка → прижать к baseline
                if e2_ratio > 0.4:
                    V_nn = min(V_nn, self._Vstar_base)

                # Уровень 3: аварийный (только по e2) — сброс η-интегратора + cooldown.
                # η-windup: при нестабильном V* η накапливает большой U.
                # Cooldown: запрещает NN увеличивать V* выше baseline в течение
                # нескольких секунд после аварии — даёт системе восстановиться.
                # ВАЖНО: наблюдатель obs НЕ сбрасывается — сброс sigma вызывает взрыв.
                # ВАЖНО: emergency гейтируется cooldown_steps == 0, чтобы НЕ перезапускаться
                # каждый шаг во время cooldown (повторный взрыв V*=min при временных ошибках).
                emergency = False
                if e2_ratio > 0.7 and self._cooldown_steps == 0:
                    V_nn = self._model.min_speed
                    emergency = True
                    self._eta[:] = 0.0          # сброс η-интегратора
                    self._cooldown_steps = 2500  # ≈5 с при dt=0.002 (период восстановления)

                # Cooldown: не позволяем NN выйти выше baseline пока система восстанавливается
                if not emergency and self._cooldown_steps > 0:
                    V_nn = min(V_nn, self._Vstar_base)
                    self._cooldown_steps -= 1

                # --- Rate limiter ---
                # Рост: ограничен (не более vstar_max_rate в секунду).
                # Снижение: в 20× быстрее — быстрая реакция на ухудшение.
                # Аварийное снижение: мгновенное (bypass).
                if emergency:
                    V_ref = float(np.clip(V_nn, self._model.min_speed, max_safe_vstar))
                else:
                    max_dv_up = self._vstar_max_rate * dt
                    max_dv_dn = self._vstar_max_rate * dt * 20.0
                    delta = V_nn - self._prev_V
                    delta = min(delta, max_dv_up) if delta > 0 else max(delta, -max_dv_dn)
                    V_ref = float(np.clip(self._prev_V + delta, self._model.min_speed, max_safe_vstar))

                self.Vstar = V_ref
                self._prev_V = V_ref
            except Exception as _exc:
                import warnings
                warnings.warn(f"speed_fn exception: {_exc}", RuntimeWarning, stacklevel=2)

        # Обновление накопленной длины дуги: s_arc = ∫₀^ζ ||t(τ)|| dτ
        # Метод средней точки (O(dζ²)): корректен для любой параметризации кривой.
        dz = s - self._prev_zeta
        if abs(dz) > 1e-12:
            mid_z = (self._prev_zeta + s) / 2.0
            self._s_arc += abs(dz) * float(np.linalg.norm(self.curve.t(mid_z)))
        self._prev_zeta = s

        # Шаг 2: геометрия в точке s
        alpha = float(self.curve.yaw_star(s))
        beta_val = float(self.curve.beta(s))
        eps_val = float(self.curve.eps(s))

        W = W_mat(alpha, beta_val, eps_val)
        Winv = W_inv(alpha, beta_val, eps_val)

        # Шаг 3: вектор ошибок
        lam1 = self._lambda_tilde_1(p_xyz, phi, s)

        # Шаг 4: матрица входов b
        b = b_mat(phi, theta, psi, u1, g=self._model.g)
        binv = _safe_inv4(b)

        # Шаг 5: оценки наблюдателя (от предыдущего шага)
        l1h, l2h, l3h, l4h, sigma = self.obs.hat()
        g1, g2, g3, g4 = (
            self.p.gamma[0], self.p.gamma[1],
            self.p.gamma[2], self.p.gamma[3],
        )
        g5 = self.p.gamma[4]

        # Шаг 6: закон управления U_bar (уравнение 72)
        v = -sigma - g1*l1h - g2*l2h - g3*l3h - g4*l4h
        Ubar = sat_tanh_vec(binv @ (Winv @ v), self.p.L)

        # Шаг 7: eta-расширение (уравнение 71)
        self._eta += dt * Ubar
        U = g5 * self._eta + Ubar

        # Шаг 8: обновление наблюдателя (уравнение 76)
        y4_model = W @ (b @ Ubar)
        self.obs.step(y=lam1, y4_model=y4_model, dt=dt)

        # Шаг 9: обновление ссылочной длины дуги s_ref = ∫₀ᵗ V*(τ) dτ.
        # ВАЖНО: обновляем В КОНЦЕ шага, чтобы при t=0 lam1[0] = s_arc - s_ref = 0,
        # что согласуется с warm-start (obs.x1[0] = 0). Только тогда e = 0 на первом шаге,
        # и высокоусиленный наблюдатель (kappa=200) не взрывается из-за начальной ошибки.
        self._s_ref += self.Vstar * dt

        return U.astype(float)

    def reset(self, zeta0: float = 0.0) -> None:
        """Сбросить внутреннее состояние регулятора к начальному."""
        self._nearest.reset(zeta0)
        self._eta[:] = 0.0
        self.obs.reset()
        self._s_arc = 0.0
        self._s_ref = 0.0
        self._prev_zeta = zeta0

    @property
    def zeta(self) -> float:
        """Текущая оценка параметра ближайшей точки."""
        return self._nearest.zeta


# ===========================================================================
# 4. Конфигурация и результаты моделирования
# ===========================================================================

@dataclass
class SimConfig:
    """Параметры симуляции согласованного управления.

    Значения по умолчанию соответствуют параметрам диссертации (стр. 44)
    для сценария со спиралью (kappa=200, dt=0.002).

    Атрибуты:
        Vstar          -- желаемая параметрическая скорость V* [м/с]
        T              -- время симуляции [с]
        dt             -- шаг интегрирования RK4 [с]
                          Важно: kappa=100 -> dt<=0.01; kappa=200 -> dt<=0.005
        x0             -- начальное состояние 16D (None -> старт в p(zeta0))
        kappa          -- коэффициент усиления наблюдателя производных
        a              -- коэффициенты полинома наблюдателя (5-кортеж)
        gamma          -- коэффициенты регулятора (5-кортеж):
                          gamma1..gamma4 -- обратная связь по производным ошибки
                          gamma5 -- вес eta-интегратора (динамическое расширение)
        L              -- уровень насыщения sat_tanh(., L)
        ell            -- параметр из условия (63): 0 < ell < 1
        gamma_nearest  -- коэффициент наблюдателя ближайшей точки (Лемма 3)
        zeta0          -- начальное значение параметра кривой
        use_numerical_grad -- True: точный dH/dzeta для наблюдателя ближайшей точки
        nearest_fn     -- аналитическая функция ближайшей точки (опционально):
                          nearest_fn(p_xyz: np.ndarray) -> float
                          Пример: nearest_fn=nearest_point_line для прямой x=s,y=s,z=s
        quad_model     -- физические параметры дрона; None → нормализованная модель
        speed_fn       -- адаптивная функция скорости (опционально):
                          speed_fn(state: np.ndarray, s: float) -> float
                          None → использовать постоянную скорость Vstar (поведение по умолчанию)
                          Пример:
                            from ml.dataset.features import feature_vector
                            speed_fn = lambda x, s: predictor.predict(feature_vector(x, curve, s=s))
    """
    Vstar: float = 1.0
    T: float = 30.0
    dt: float = 0.002
    x0: Optional[np.ndarray] = None

    kappa: float = 200.0
    a: tuple = (5.0, 10.0, 10.0, 5.0, 1.0)
    gamma: tuple = (1.0, 3.0, 5.0, 3.0, 1.0)
    L: float = 5.0
    ell: float = 0.9

    gamma_nearest: float = 1.0
    zeta0: float = 0.0
    use_numerical_grad: bool = False

    nearest_fn: Optional[Callable] = None
    quad_model: Optional[QuadModel] = None
    speed_fn:   Optional[Callable] = None
    warmup_time: float = 5.0       # первые N сек: NN отключена, используется константная Vstar
    vstar_max_rate: float = 0.3    # макс. скорость изменения V* [параметр/с] при NN-управлении


@dataclass
class SimResult:
    """Результаты симуляции согласованного управления.

    Атрибуты:
        t        -- массив времени [n]
        x        -- траектория состояния [n x 16]
        zeta     -- параметр ближайшей точки [n]
        p_ref    -- опорная траектория (ближайшие точки на кривой) [n x 3]
        errors   -- ошибки регулирования [n x 4]:
                    col 0: s_arc - V*t [м]
                    col 1: e1 [м]  (боковая ошибка 1)
                    col 2: e2 [м]  (боковая ошибка 2)
                    col 3: delta_phi [рад]  (ошибка рысканья)
        velocity -- норма скорости ||v|| [n]
        curve    -- геометрия кривой
        cfg      -- конфигурация симуляции
    """
    t: np.ndarray
    x: np.ndarray
    zeta: np.ndarray
    p_ref: np.ndarray
    errors: np.ndarray
    velocity: np.ndarray
    curve: CurveGeom
    cfg: SimConfig

    def print_summary(self) -> None:
        """Вывести сводку финальных ошибок и скорости."""
        import sys
        out = sys.stdout
        # На Windows консоль может использовать cp866/cp1251 — принудительно UTF-8
        try:
            if hasattr(out, "reconfigure"):
                out.reconfigure(encoding="utf-8")
        except Exception:
            pass

        e = self.errors[-1]
        v = self.velocity[-1]
        print(f"  Результаты симуляции (t = {self.cfg.T} с):")
        print(f"    s_arc - V*t  = {e[0]:+.4f} м")
        print(f"    e1           = {e[1]:+.4f} м")
        print(f"    e2           = {e[2]:+.4f} м")
        print(f"    delta_phi    = {e[3]:+.4f} рад")
        print(f"    ||v||        = {v:.4f} м/с  (V* = {self.cfg.Vstar})")
        print(f"    zeta_final   = {self.zeta[-1]:.4f}")

    def plot(self, out_dir: str, prefix: str = "sim") -> None:
        """Сохранить 6 графиков результатов в директорию out_dir.

        Файлы: {prefix}_traj_3d.png, _traj_xy.png, _errors.png,
               _yaw_error.png, _velocity.png, _angles.png
        """
        os.makedirs(out_dir, exist_ok=True)

        _plot_3d_traj(
            p_ref=self.p_ref, p_real=self.x[:, 0:3],
            outpath=os.path.join(out_dir, f"{prefix}_traj_3d.png"),
            title="Согласованное управление: 3D траектория",
        )
        _plot_xy(
            p_ref=self.p_ref, p_real=self.x[:, 0:3],
            outpath=os.path.join(out_dir, f"{prefix}_traj_xy.png"),
            title="Проекция X-Y",
        )
        _plot_errors(
            t=self.t, e=self.errors[:, :3],
            labels=["s_arc - V*t, м", "e1, м", "e2, м"],
            outpath=os.path.join(out_dir, f"{prefix}_errors.png"),
            title="Ошибки слежения за траекторией",
        )
        _plot_errors(
            t=self.t, e=self.errors[:, 3:4],
            labels=["delta_phi, рад"],
            outpath=os.path.join(out_dir, f"{prefix}_yaw_error.png"),
            title="Ошибка по углу рысканья",
        )
        _plot_velocity(
            t=self.t, vel=self.velocity, Vstar=self.cfg.Vstar,
            outpath=os.path.join(out_dir, f"{prefix}_velocity.png"),
        )
        _plot_angles(
            t=self.t, angles=self.x[:, 6:9],
            outpath=os.path.join(out_dir, f"{prefix}_angles.png"),
        )

        print(f"  Графики сохранены в {display_path(out_dir)}")


# ===========================================================================
# 5. Главная функция симуляции
# ===========================================================================

def simulate_path_following(
    curve: CurveGeom,
    cfg: SimConfig,
) -> SimResult:
    """Симулировать согласованное управление квадрокоптером вдоль кривой.

    Параметры:
        curve  -- геометрия кривой (из make_curve() или geometry.spiral_curve() и т.п.)
        cfg    -- параметры симуляции (SimConfig)

    Возвращает:
        SimResult -- результаты: траектории, ошибки, скорость, опорная кривая
    """
    params = HighGainParams(
        kappa=cfg.kappa,
        a=tuple(cfg.a),
        gamma=tuple(cfg.gamma),
        L=cfg.L,
        ell=cfg.ell,
    )

    if cfg.x0 is None:
        x0 = np.zeros(16, dtype=float)
        x0[0:3] = curve.p(cfg.zeta0)
    else:
        x0 = np.asarray(cfg.x0, dtype=float).copy()
        if len(x0) != 16:
            raise ValueError(
                f"x0 должен быть 16-мерным вектором состояния, получено {len(x0)}."
            )

    model = cfg.quad_model if cfg.quad_model is not None else QuadModel()

    ctrl = PathFollowingController(
        curve=curve,
        Vstar=cfg.Vstar,
        params=params,
        gamma_nearest=cfg.gamma_nearest,
        zeta0=cfg.zeta0,
        use_numerical_grad=cfg.use_numerical_grad,
        nearest_fn=cfg.nearest_fn,
        quad_model=model,
        speed_fn=cfg.speed_fn,
        warmup_time=cfg.warmup_time,
    )
    ctrl._vstar_max_rate = cfg.vstar_max_rate

    # Тёплый старт наблюдателя производных (предотвращает взрыв sigma при perturbation).
    #
    # Проблема: при ненулевых начальных возмущениях x0, e = lam1_0 - x1_obs ≠ 0.
    # Явный Эйлер:  x2_new += dt × k²·a2·e  →  10^4·10·0.01 × e = 1000·e (в 10× больше реального dlam1/dt).
    # sigma_new += dt × k^5·a5·e → 10^8 × e (гигантское значение, управление насыщается).
    #
    # Решение: инициализировать x1_obs = lam1_0 и x2_obs = dlam1/dt|₀ ≈ Frenet-проекция скорости.
    # Тогда e = 0 на первом шаге и observer корректно отслеживает далее.
    # Для x0=0 (обычные сценарии): lam1_0 ≈ 0, vel ≈ 0 → obs ≈ 0 (обратная совместимость).
    from drone_sim.geometry.curves import Rz as _Rz, Ry as _Ry
    _zeta0 = float(cfg.zeta0)
    lam1_0 = ctrl._lambda_tilde_1(x0[0:3], float(x0[6]), _zeta0)
    ctrl.obs.x1 = lam1_0.copy()
    # x2_obs ≈ dlam1/dt: Frenet-проекция начальной скорости
    # Компоненты e1, e2 изменяются со скоростью проекции vel_0 на оси Frenet-frame.
    # Компонента s_arc-s_ref: ds_arc/dt≈0, ds_ref/dt=V* → d(lam1[0])/dt ≈ -V*
    # Компонента d_phi: phi_dot = x0[9], yaw_star не меняется мгновенно → d(lam1[3])/dt ≈ phi_dot
    _alpha0 = float(curve.yaw_star(_zeta0))
    _beta0 = float(curve.beta(_zeta0))
    _vel0 = x0[3:6]
    _q_vel = _Ry(_beta0).T @ (_Rz(_alpha0).T @ _vel0)
    lam2_0 = np.array([
        -float(cfg.Vstar),   # d(s_arc - V*t)/dt ≈ -V* (drone at rest initially)
        float(_q_vel[1]),    # de1/dt ≈ vel component along e1 direction
        float(_q_vel[2]),    # de2/dt ≈ vel component along e2 direction
        float(x0[9]),        # d(d_phi)/dt ≈ phi_dot
    ], dtype=float)
    ctrl.obs.x2 = lam2_0.copy()

    def dynamics(x: np.ndarray, U: np.ndarray) -> np.ndarray:
        return quad_dynamics_16(x, U, L=cfg.L, model=model)

    # runner.py вызывает step() ДВАЖДЫ при t=0: сначала перед циклом (для определения
    # размерности U), потом снова в цикле при k=0. В старом коде Vstar*t=0 при t=0,
    # поэтому двойной вызов был безвреден. В новом коде s_ref += Vstar*dt при каждом
    # вызове → наблюдатель получает ошибку и sigma взрывается.
    # Решение: после pre-loop вызова восстанавливаем состояние (warm-start повторяется).
    _pre_loop_done = [False]
    _vstar_log: list[float] = []    # история V* по шагам (для post-hoc s_ref)

    def step(t: float, x: np.ndarray, Uprev, dt: float) -> np.ndarray:
        U = ctrl.step(t, x, Uprev, dt)
        if not _pre_loop_done[0]:
            # Это был pre-loop вызов → сбрасываем состояние до начального
            ctrl.reset(cfg.zeta0)
            ctrl._prev_V = cfg.Vstar
            ctrl.obs.x1 = lam1_0.copy()
            ctrl.obs.x2 = lam2_0.copy()
            _pre_loop_done[0] = True
        else:
            _vstar_log.append(ctrl.Vstar)
        return U

    raw = _simulate_raw(dynamics, step, x0, T=cfg.T, dt=cfg.dt)
    t_arr = raw["t"]
    x_arr = raw["x"]
    n = len(t_arr)

    zeta_arr = _recompute_zeta(x_arr, curve, cfg)
    p_ref = np.stack([curve.p(z) for z in zeta_arr], axis=0)

    # Длина дуги как интеграл ∫₀^ζ ||t(τ)|| dτ (метод средней точки).
    # Корректна для любой параметризации; для ||t||=const совпадает с ζ·||t||.
    s_arc_arr = np.zeros(n, dtype=float)
    for k in range(1, n):
        dz = zeta_arr[k] - zeta_arr[k - 1]
        if abs(dz) > 1e-12:
            mid_z = (zeta_arr[k - 1] + zeta_arr[k]) / 2.0
            s_arc_arr[k] = s_arc_arr[k - 1] + abs(dz) * float(np.linalg.norm(curve.t(mid_z)))
        else:
            s_arc_arr[k] = s_arc_arr[k - 1]

    # s_ref = ∫₀ᵗ V*(τ) dτ — реконструируем из истории V* (накопленной в _vstar_log).
    # _vstar_log[k] = V* на шаге k (начиная с k=0 первого реального шага цикла).
    # Длина _vstar_log = n-1 (последний шаг не записывается, т.к. runner не вызывает step).
    # Для шага k=0 s_ref=0, далее s_ref[k] = sum(V*[0..k-1]) * dt.
    s_ref_arr = np.zeros(n, dtype=float)
    vlog = _vstar_log if len(_vstar_log) == n - 1 else None
    if vlog is not None:
        for k in range(1, n):
            s_ref_arr[k] = s_ref_arr[k - 1] + float(vlog[k - 1]) * cfg.dt
    else:
        # Fallback: при speed_fn=None или несоответствии длин
        s_ref_arr = cfg.Vstar * t_arr

    errors = np.zeros((n, 4), dtype=float)
    for k in range(n):
        z = zeta_arr[k]
        _, e1, e2 = se_from_pose(x_arr[k, 0:3], z, curve)
        phi = float(x_arr[k, 6])
        phi_star = float(curve.yaw_star(z))
        d_phi = float(np.arctan2(np.sin(phi - phi_star), np.cos(phi - phi_star)))
        errors[k] = [s_arc_arr[k] - s_ref_arr[k], e1, e2, d_phi]

    velocity = np.linalg.norm(x_arr[:, 3:6], axis=1)

    return SimResult(
        t=t_arr, x=x_arr, zeta=zeta_arr, p_ref=p_ref,
        errors=errors, velocity=velocity, curve=curve, cfg=cfg,
    )


def _recompute_zeta(
    x_arr: np.ndarray,
    curve: CurveGeom,
    cfg: SimConfig,
) -> np.ndarray:
    """Прогнать наблюдатель ближайшей точки на записанной траектории."""
    n = len(x_arr)
    zeta_arr = np.zeros(n, dtype=float)

    if cfg.nearest_fn is not None:
        for k in range(n):
            zeta_arr[k] = float(cfg.nearest_fn(x_arr[k, 0:3]))
    else:
        obs = NearestPointObserver(
            curve=curve,
            gamma=cfg.gamma_nearest,
            zeta0=cfg.zeta0,
            numerical_grad=cfg.use_numerical_grad,
        )
        for k in range(n):
            zeta_arr[k] = obs.step(x_arr[k, 0:3], cfg.dt)

    return zeta_arr


# ===========================================================================
# 6. Утилиты визуализации (автономные)
# ===========================================================================

def _plot_3d_traj(p_ref, p_real, outpath, title=""):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    if p_ref is not None:
        ax.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2],
                "--r", linewidth=1.5, label="Заданная кривая")
    ax.plot(p_real[:, 0], p_real[:, 1], p_real[:, 2],
            color=(0.0078, 0.447, 0.741), linewidth=2.0,
            label="Траектория квадрокоптера")
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м"); ax.set_zlabel("z, м")
    ax.legend(); ax.grid(True)
    if title:
        ax.set_title(title)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def _plot_xy(p_ref, p_real, outpath, title="Проекция X-Y"):
    fig, ax = plt.subplots(figsize=(6, 6))
    if p_ref is not None:
        ax.plot(p_ref[:, 0], p_ref[:, 1], "--r", linewidth=1.5, label="Заданная")
    ax.plot(p_real[:, 0], p_real[:, 1],
            color=(0.0078, 0.447, 0.741), linewidth=2.0, label="Квадрокоптер")
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
    ax.grid(True, linestyle="--"); ax.legend()
    ax.set_title(title); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def _plot_errors(t, e, labels, outpath, title=""):
    fig, ax = plt.subplots(figsize=(10, 4))
    styles = ["--", "-.", "-", ":"]
    colors = [(0.466, 0.674, 0.188), (0.929, 0.694, 0.125),
              (0.0078, 0.447, 0.741), "r"]
    for i, lab in enumerate(labels):
        ax.plot(t, e[:, i], linewidth=2.0, label=lab,
                linestyle=styles[i % len(styles)],
                color=colors[i % len(colors)])
    ax.set_xlabel("t, с"); ax.grid(True, linestyle="--"); ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def _plot_velocity(t, vel, Vstar, outpath):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, vel, color=(0.0078, 0.447, 0.741), linewidth=2.0, label="||v||, м/с")
    ax.axhline(Vstar, color="r", linestyle="--", linewidth=1.5, label=f"V* = {Vstar}")
    ax.set_xlabel("t, с"); ax.grid(True, linestyle="--"); ax.legend()
    ax.set_title("Линейная скорость")
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def _plot_angles(t, angles, outpath):
    fig, ax = plt.subplots(figsize=(8, 4))
    lbls = ["phi (рысканье)", "theta (тангаж)", "psi (крен)"]
    styles = ["-", "--", "-."]
    for i in range(min(3, angles.shape[1])):
        ax.plot(t, angles[:, i], linewidth=2.0, label=lbls[i], linestyle=styles[i])
    ax.set_xlabel("t, с"); ax.set_ylabel("рад")
    ax.grid(True, linestyle="--"); ax.legend()
    ax.set_title("Угловые координаты")
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)
