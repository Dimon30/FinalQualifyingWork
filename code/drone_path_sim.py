"""
drone_path_sim.py
=================
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

Замечание по параметризации:
    Для точности метрики s_arc рекомендуется использовать равномерную (дуговую)
    параметризацию кривой, при которой ||dp/ds|| = const. В этом случае
    s_arc = zeta * ||t|| точно совпадает с длиной дуги и ṡ_arc -> V*.

Пример использования::

    import numpy as np
    from drone_path_sim import make_curve, SimConfig, simulate_path_following

    # Эллиптическая спираль: a=3, b=2, подъём 0.5 рад/м
    curve = make_curve(lambda s: np.array([3*np.cos(s), 2*np.sin(s), 0.5*s]))

    cfg = SimConfig(Vstar=1.0, T=25.0, dt=0.002, kappa=200)
    result = simulate_path_following(curve, cfg)

    result.print_summary()
    result.plot("out_images/elliptic_spiral")
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Путь к модулям проекта: поддержка запуска как из code/, так и из корня
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dynamics import quad_dynamics_16, sat_tanh, sat_tanh_vec, G
from quad_model import QuadModel
from controllers.common import HighGainParams, DerivativeObserver4
from controllers.path_following import W_mat, W_inv, b_mat, _safe_inv4
from geometry import CurveGeom, se_from_pose
from sim import simulate as _simulate_raw

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
        - b_mat, W_mat/W_inv из ch4_path.py
        - 16-мерную модель квадрокоптера

    Параметры:
        curve             -- геометрия кривой S
        Vstar             -- желаемая скорость (параметрическая)
        params            -- HighGainParams (kappa, a, gamma, L, ell)
        gamma_nearest     -- коэффициент наблюдателя ближайшей точки (NearestPointObserver)
        zeta0             -- начальное значение параметра
        use_numerical_grad -- True: точный gradH для NearestPointObserver
        nearest_fn        -- опциональная аналитическая функция ближайшей точки:
                             nearest_fn(p_xyz: np.ndarray) -> float
                             Если задана — используется вместо NearestPointObserver.
                             Пример: nearest_fn=nearest_point_line для прямой x=s,y=s,z=s.
                             Позволяет избежать лага динамического наблюдателя для
                             кривых с известным аналитическим решением.
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
    ):
        self.curve = curve
        self.Vstar = float(Vstar)
        self.p = params
        self._nearest_fn = nearest_fn  # аналитическая функция (если есть)
        self._model = quad_model if quad_model is not None else QuadModel()

        self._nearest = NearestPointObserver(
            curve=curve,
            gamma=gamma_nearest,
            zeta0=zeta0,
            numerical_grad=use_numerical_grad,
        )
        self._eta = np.zeros(4, dtype=float)
        self.obs = DerivativeObserver4(dim=4, p=params)

    def _lambda_tilde_1(
        self, t: float, p_xyz: np.ndarray, phi: float, s: float
    ) -> np.ndarray:
        """Вектор регулируемых переменных lambda_tilde_1.

        lambda_tilde_1 = col(s_arc - V*t, e1, e2, delta_phi)

        s_arc = zeta * ||t(zeta)|| -- приближение длины дуги (точно при ||t|| = const)
        e1, e2 -- боковые ошибки в системе координат Френе (уравнение 60)
        delta_phi = phi - phi*(zeta) -- ошибка рысканья (нормализована в [-pi, pi])
        """
        _, e1, e2 = se_from_pose(p_xyz, s, self.curve)
        phi_star = float(self.curve.yaw_star(s))
        d_phi = float(np.arctan2(np.sin(phi - phi_star), np.cos(phi - phi_star)))
        tangent_norm = float(np.linalg.norm(self.curve.t(s)))
        s_arc = s * tangent_norm
        return np.array([s_arc - self.Vstar * float(t), e1, e2, d_phi], dtype=float)

    def step(
        self,
        t: float,
        x: np.ndarray,
        Uprev: Optional[np.ndarray],
        dt: float,
    ) -> np.ndarray:
        """Один шаг регулятора.

        Аргументы:
            t     -- текущее время [с]
            x     -- вектор состояния 16D
                     [x,y,z, vx,vy,vz, phi,theta,psi, phidot,thetadot,psidot,
                      u1_bar,rho1, u2,rho2]
            Uprev -- предыдущее управление (не используется, для совместимости с sim.py)
            dt    -- шаг времени [с]

        Возвращает:
            U = [v1, v2, u3, u4] -- управляющий вектор
        """
        p_xyz = x[0:3]
        phi = float(x[6])
        theta = float(x[7])
        psi = float(x[8])
        u1_bar = float(x[12])
        u1 = sat_tanh(u1_bar, self.p.L)

        # Шаг 1: оценить ближайшую точку на кривой
        if self._nearest_fn is not None:
            s = float(self._nearest_fn(p_xyz))
            self._nearest._zeta = s   # синхронизация для свойства .zeta
        else:
            s = self._nearest.step(p_xyz, dt)

        # Шаг 2: геометрия кривой в точке s
        alpha = float(self.curve.yaw_star(s))
        beta_val = float(self.curve.beta(s))
        eps_val = float(self.curve.eps(s))

        W = W_mat(alpha, beta_val, eps_val)
        Winv = W_inv(alpha, beta_val, eps_val)

        # Шаг 3: вектор ошибок lambda_tilde_1
        lam1 = self._lambda_tilde_1(t, p_xyz, phi, s)

        # Шаг 4: матрица входов b (уравнение из стр. 38, g из QuadModel)
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

        # Шаг 7: динамическое eta-расширение (уравнение 71)
        self._eta += dt * Ubar
        U = g5 * self._eta + Ubar

        # Шаг 8: обновление наблюдателя (уравнение 76)
        # y4_model = W * b * U_bar  (известная часть 4-й производной)
        y4_model = W @ (b @ Ubar)
        self.obs.step(y=lam1, y4_model=y4_model, dt=dt)

        return U.astype(float)

    def reset(self, zeta0: float = 0.0) -> None:
        """Сбросить внутреннее состояние регулятора к начальному."""
        self._nearest.reset(zeta0)
        self._eta[:] = 0.0
        self.obs.reset()

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
                          (для arc-length параметризации = скорость вдоль кривой)
        T              -- время симуляции [с]
        dt             -- шаг интегрирования RK4 [с]
                          Важно: kappa=100 -> dt<=0.01; kappa=200 -> dt<=0.005
        x0             -- начальное состояние 16D (None -> старт в p(zeta0))
        kappa          -- коэффициент усиления наблюдателя производных
        a              -- коэффициенты полинома наблюдателя (5-кортеж)
                          Гурвицев полином: p^5 + a1*p^4 + ... + a5
        gamma          -- коэффициенты регулятора (5-кортеж):
                          gamma1..gamma4 -- обратная связь по производным ошибки
                          gamma5 -- вес eta-интегратора (динамическое расширение)
        L              -- уровень насыщения sat_tanh(., L)
        ell            -- параметр из условия (63): 0 < ell < 1
        gamma_nearest  -- коэффициент наблюдателя ближайшей точки (Лемма 3)
        zeta0          -- начальное значение параметра кривой
        use_numerical_grad -- True: точный dH/dzeta для наблюдателя ближайшей точки
    """
    Vstar: float = 1.0
    T: float = 30.0
    dt: float = 0.002
    x0: Optional[np.ndarray] = None

    # Параметры регулятора (из диссертации стр. 44, сценарий спираль)
    kappa: float = 200.0
    a: tuple = (5.0, 10.0, 10.0, 5.0, 1.0)
    gamma: tuple = (1.0, 3.0, 5.0, 3.0, 1.0)
    L: float = 5.0
    ell: float = 0.9

    # Параметры наблюдателя ближайшей точки
    gamma_nearest: float = 1.0
    zeta0: float = 0.0
    use_numerical_grad: bool = False

    # Аналитическая функция ближайшей точки (опционально)
    # Если задана — заменяет NearestPointObserver.
    # Принимает p_xyz: np.ndarray[3], возвращает float (параметр ζ).
    # Пример для прямой x=s,y=s,z=s:
    #   from geometry import nearest_point_line
    #   cfg = SimConfig(..., nearest_fn=nearest_point_line)
    nearest_fn: Optional[Callable] = None

    # Физические параметры квадрокоптера
    # None → QuadModel() — нормализованная модель диссертации (mass=1, J=1, g=9.81)
    quad_model: Optional[QuadModel] = None


@dataclass
class SimResult:
    """Результаты симуляции согласованного управления.

    Атрибуты:
        t        -- массив времени [n]
        x        -- траектория состояния [n x 16]
                    Индексы: [x,y,z]=0:3, [vx,vy,vz]=3:6,
                             [phi,theta,psi]=6:9, [phidot,thetadot,psidot]=9:12
        zeta     -- параметр ближайшей точки (восстановленный) [n]
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
        """Сохранить графики результатов в директорию out_dir.

        Сохраняемые файлы:
            {prefix}_traj_3d.png   -- 3D траектория
            {prefix}_traj_xy.png   -- проекция X-Y
            {prefix}_errors.png    -- ошибки s_arc-V*t, e1, e2
            {prefix}_yaw_error.png -- ошибка рысканья delta_phi
            {prefix}_velocity.png  -- скорость vs V*
            {prefix}_angles.png    -- угловые координаты phi, theta, psi

        Параметры:
            out_dir -- директория для сохранения (создаётся автоматически)
            prefix  -- префикс имён файлов (по умолчанию "sim")
        """
        os.makedirs(out_dir, exist_ok=True)

        _plot_3d_traj(
            p_ref=self.p_ref,
            p_real=self.x[:, 0:3],
            outpath=os.path.join(out_dir, f"{prefix}_traj_3d.png"),
            title="Согласованное управление: 3D траектория",
        )
        _plot_xy(
            p_ref=self.p_ref,
            p_real=self.x[:, 0:3],
            outpath=os.path.join(out_dir, f"{prefix}_traj_xy.png"),
            title="Проекция X-Y",
        )
        _plot_errors(
            t=self.t,
            e=self.errors[:, :3],
            labels=["s_arc - V*t, м", "e1, м", "e2, м"],
            outpath=os.path.join(out_dir, f"{prefix}_errors.png"),
            title="Ошибки слежения за траекторией",
        )
        _plot_errors(
            t=self.t,
            e=self.errors[:, 3:4],
            labels=["delta_phi, рад"],
            outpath=os.path.join(out_dir, f"{prefix}_yaw_error.png"),
            title="Ошибка по углу рысканья",
        )
        _plot_velocity(
            t=self.t,
            vel=self.velocity,
            Vstar=self.cfg.Vstar,
            outpath=os.path.join(out_dir, f"{prefix}_velocity.png"),
        )
        _plot_angles(
            t=self.t,
            angles=self.x[:, 6:9],
            outpath=os.path.join(out_dir, f"{prefix}_angles.png"),
        )

        print(f"  Графики сохранены в {out_dir}/")


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

    Алгоритм:
        1. Создать PathFollowingController с параметрами из cfg
        2. Задать начальное состояние (из cfg.x0 или точка p(zeta0) на кривой)
        3. Выполнить численное интегрирование RK4 (sim.py)
        4. Восстановить параметры ближайшей точки zeta (прогон наблюдателя)
        5. Вычислить ошибки и скорость
        6. Вернуть SimResult

    Пример:
        >>> import numpy as np
        >>> from drone_path_sim import make_curve, SimConfig, simulate_path_following
        >>> curve = make_curve(lambda s: np.array([3*np.cos(s), 3*np.sin(s), s]))
        >>> cfg = SimConfig(Vstar=1.0, T=20.0, dt=0.002)
        >>> result = simulate_path_following(curve, cfg)
        >>> result.print_summary()
        >>> result.plot("out_images/my_spiral")
    """
    # Собрать HighGainParams
    params = HighGainParams(
        kappa=cfg.kappa,
        a=tuple(cfg.a),
        gamma=tuple(cfg.gamma),
        L=cfg.L,
        ell=cfg.ell,
    )

    # Начальное состояние
    if cfg.x0 is None:
        x0 = np.zeros(16, dtype=float)
        x0[0:3] = curve.p(cfg.zeta0)  # положение на кривой в точке zeta0
    else:
        x0 = np.asarray(cfg.x0, dtype=float).copy()
        if len(x0) != 16:
            raise ValueError(
                f"x0 должен быть 16-мерным вектором состояния, получено {len(x0)}. "
                f"Структура: [x,y,z, vx,vy,vz, phi,theta,psi, "
                f"phidot,thetadot,psidot, u1_bar,rho1, u2,rho2]"
            )

    # Физические параметры дрона (явные или нормализованная модель)
    model = cfg.quad_model if cfg.quad_model is not None else QuadModel()

    # Создать контроллер
    ctrl = PathFollowingController(
        curve=curve,
        Vstar=cfg.Vstar,
        params=params,
        gamma_nearest=cfg.gamma_nearest,
        zeta0=cfg.zeta0,
        use_numerical_grad=cfg.use_numerical_grad,
        nearest_fn=cfg.nearest_fn,
        quad_model=model,
    )

    def dynamics(x: np.ndarray, U: np.ndarray) -> np.ndarray:
        return quad_dynamics_16(x, U, L=cfg.L, model=model)

    def step(t: float, x: np.ndarray, Uprev, dt: float) -> np.ndarray:
        return ctrl.step(t, x, Uprev, dt)

    # Запуск симуляции
    raw = _simulate_raw(dynamics, step, x0, T=cfg.T, dt=cfg.dt)
    t_arr = raw["t"]
    x_arr = raw["x"]
    n = len(t_arr)

    # Восстановить zeta (прогон наблюдателя ближайшей точки на записанной траектории)
    zeta_arr = _recompute_zeta(x_arr, curve, cfg)

    # Опорная траектория (ближайшие точки на кривой)
    p_ref = np.stack([curve.p(z) for z in zeta_arr], axis=0)

    # Вычислить ошибки регулирования
    errors = np.zeros((n, 4), dtype=float)
    for k in range(n):
        z = zeta_arr[k]
        _, e1, e2 = se_from_pose(x_arr[k, 0:3], z, curve)
        phi = float(x_arr[k, 6])
        phi_star = float(curve.yaw_star(z))
        d_phi = float(np.arctan2(np.sin(phi - phi_star), np.cos(phi - phi_star)))
        t_norm = float(np.linalg.norm(curve.t(z)))
        s_arc = z * t_norm
        errors[k] = [s_arc - cfg.Vstar * t_arr[k], e1, e2, d_phi]

    # Скорость дрона
    velocity = np.linalg.norm(x_arr[:, 3:6], axis=1)

    return SimResult(
        t=t_arr,
        x=x_arr,
        zeta=zeta_arr,
        p_ref=p_ref,
        errors=errors,
        velocity=velocity,
        curve=curve,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Вспомогательная функция: восстановление zeta после симуляции
# ---------------------------------------------------------------------------

def _recompute_zeta(
    x_arr: np.ndarray,
    curve: CurveGeom,
    cfg: SimConfig,
) -> np.ndarray:
    """Прогнать наблюдатель ближайшей точки на записанной траектории.

    Если cfg.nearest_fn задана — используется точная аналитическая формула.
    Иначе — NearestPointObserver интегрируется на записанной траектории.
    Возвращает массив параметров zeta[k] для каждого шага симуляции.
    """
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
# 6. Утилиты визуализации (автономные, без зависимости от plotting.py)
# ===========================================================================

def _plot_3d_traj(
    p_ref: np.ndarray,
    p_real: np.ndarray,
    outpath: str,
    title: str = "",
) -> None:
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
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_xy(
    p_ref: np.ndarray,
    p_real: np.ndarray,
    outpath: str,
    title: str = "Проекция X-Y",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if p_ref is not None:
        ax.plot(p_ref[:, 0], p_ref[:, 1], "--r", linewidth=1.5, label="Заданная")
    ax.plot(p_real[:, 0], p_real[:, 1],
            color=(0.0078, 0.447, 0.741), linewidth=2.0, label="Квадрокоптер")
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
    ax.grid(True, linestyle="--"); ax.legend()
    ax.set_title(title); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_errors(
    t: np.ndarray,
    e: np.ndarray,
    labels: list,
    outpath: str,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    styles = ["--", "-.", "-", ":"]
    colors = [
        (0.466, 0.674, 0.188),
        (0.929, 0.694, 0.125),
        (0.0078, 0.447, 0.741),
        "r",
    ]
    for i, lab in enumerate(labels):
        ax.plot(t, e[:, i], linewidth=2.0, label=lab,
                linestyle=styles[i % len(styles)],
                color=colors[i % len(colors)])
    ax.set_xlabel("t, с"); ax.grid(True, linestyle="--")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_velocity(
    t: np.ndarray,
    vel: np.ndarray,
    Vstar: float,
    outpath: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, vel, color=(0.0078, 0.447, 0.741), linewidth=2.0, label="||v||, м/с")
    ax.axhline(Vstar, color="r", linestyle="--", linewidth=1.5,
               label=f"V* = {Vstar}")
    ax.set_xlabel("t, с"); ax.grid(True, linestyle="--")
    ax.legend()
    ax.set_title("Линейная скорость")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_angles(
    t: np.ndarray,
    angles: np.ndarray,
    outpath: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    lbls = ["phi (рысканье)", "theta (тангаж)", "psi (крен)"]
    styles = ["-", "--", "-."]
    for i in range(min(3, angles.shape[1])):
        ax.plot(t, angles[:, i], linewidth=2.0,
                label=lbls[i], linestyle=styles[i])
    ax.set_xlabel("t, с"); ax.set_ylabel("рад")
    ax.grid(True, linestyle="--"); ax.legend()
    ax.set_title("Угловые координаты")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ===========================================================================
# 7. Быстрая проверка (python drone_path_sim.py)
# ===========================================================================

if __name__ == "__main__":
    import sys

    # Принудительно UTF-8 на Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== drone_path_sim: быстрая проверка ===")
    print("Кривая: спираль r=3 (как в диссертации, Гл. 4)")

    # Воспроизводим тест из run_ch4_spiral.py через новый API
    curve = make_curve(lambda s: np.array([3.0*np.cos(s), 3.0*np.sin(s), s]))

    # Начальное состояние: x0=(2.9, 0, 0) как в диссертации
    x0 = np.zeros(16)
    x0[0:3] = np.array([2.9, 0.0, 0.0])

    cfg = SimConfig(
        Vstar=1.0,
        T=20.0,       # укорочено для быстрой проверки
        dt=0.002,
        x0=x0,
        kappa=200.0,
        gamma=(1.0, 3.0, 5.0, 3.0, 1.0),
        gamma_nearest=1.0,
        zeta0=0.0,
    )

    print(f"  T={cfg.T}с, dt={cfg.dt}, kappa={cfg.kappa}, V*={cfg.Vstar}")
    result = simulate_path_following(curve, cfg)
    result.print_summary()
    result.plot("out_images/drone_path_sim_test", prefix="spiral")
    print("=== Готово ===")
