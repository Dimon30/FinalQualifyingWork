"""
ml/dataset/simulator_wrapper.py
=================================
Обёртка над симулятором проекта для ML-пайплайна.

Использует существующий API проекта:
    simulate_path_following(curve: CurveGeom, cfg: SimConfig) -> SimResult

Все пороговые значения (макс. скорость, лимиты ошибок, порог разлёта)
берутся из QuadModel (drone_sim.models.quad_model), а не из констант.
Параметры поиска скорости — из OracleConfig (ml.config).

Публичный API:
    rollout_with_speed(state, curve, V, horizon, drone) -> RolloutMetrics
    is_stable(metrics, drone)                           -> bool
    find_optimal_speed(state, curve, drone, oracle_cfg) -> float

Примечания:
    V         — параметрическая скорость V* (не дуговая!);
                принудительно ограничивается: V = min(V, drone.max_speed)
    state     — 16-мерный вектор состояния; None → старт с кривой при zeta=0
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional, TypedDict, Union

from drone_sim import CurveGeom, SimConfig, make_curve, simulate_path_following
from drone_sim.models.quad_model import QuadModel
from ml.config import ORACLE_DT, ORACLE_KAPPA, OracleConfig


# ---------------------------------------------------------------------------
# Тип метрик
# ---------------------------------------------------------------------------

class RolloutMetrics(TypedDict):
    """Метрики одного ролаута симуляции.

    Ключи:
        max_e2            — максимальное |e2| за весь ролаут [м]
        final_e2          — |e2| в конечный момент времени [м]
        nan_detected      — True, если в состоянии обнаружен NaN/Inf
        velocity_exploded — True, если ||v|| > drone.max_velocity_norm
    """
    max_e2: float
    final_e2: float
    nan_detected: bool
    velocity_exploded: bool


# ---------------------------------------------------------------------------
# Ролаут
# ---------------------------------------------------------------------------

def rollout_with_speed(
    state: Optional[np.ndarray],
    curve: Union[CurveGeom, Callable[[float], np.ndarray]],
    V: float,
    horizon: int,
    drone: Optional[QuadModel] = None,
    dt: float = ORACLE_DT,
    kappa: float = ORACLE_KAPPA,
    zeta0: float = 0.0,
) -> RolloutMetrics:
    """Выполнить ролаут симуляции с заданной скоростью V*.

    Параметры:
        state   — начальное состояние квадрокоптера (16-мерный вектор)
                  или None (старт с точки кривой при zeta=zeta0)
        curve   — CurveGeom или Callable p(s) -> ndarray[3]
                  (Callable оборачивается через make_curve автоматически)
        V       — желаемая параметрическая скорость V*;
                  ПРИНУДИТЕЛЬНО ограничивается: V = min(V, drone.max_speed)
        horizon — число шагов RK4; длительность T = horizon × dt
        drone   — параметры дрона (QuadModel); None → нормализованная модель
        dt      — шаг интегрирования [с] (по умолчанию ORACLE_DT)
        kappa   — коэффициент усиления наблюдателя
        zeta0   — начальное значение параметра кривой для NearestPointObserver;
                  должно совпадать с положением дрона на кривой, иначе
                  наблюдатель будет сходиться долго и даст ложные ошибки

    Возвращает:
        RolloutMetrics — метрики ролаута
    """
    if drone is None:
        drone = QuadModel()

    # 1. Ограничение скорости — берём предел из объекта дрона
    V = min(float(V), drone.max_speed)

    # 2. Преобразование curve к CurveGeom если передана обычная функция
    if not isinstance(curve, CurveGeom):
        curve = make_curve(curve)

    # 3. Параметры симуляции
    T = horizon * dt
    cfg = SimConfig(
        Vstar=V,
        T=T,
        dt=dt,
        x0=np.asarray(state, dtype=float) if state is not None else None,
        kappa=kappa,
        quad_model=drone,
        zeta0=zeta0,
    )

    # 4. Запуск; исключение → NaN-результат
    try:
        result = simulate_path_following(curve, cfg)
    except Exception:
        return RolloutMetrics(
            max_e2=float("inf"),
            final_e2=float("inf"),
            nan_detected=True,
            velocity_exploded=False,
        )

    # 5. Метрики из SimResult
    e2_series = result.errors[:, 2]   # errors[:, 2] = e2
    velocity = result.velocity

    nan_detected = bool(
        np.any(~np.isfinite(result.x)) or np.any(~np.isfinite(e2_series))
    )

    # Порог разлёта из объекта дрона
    velocity_exploded = bool(np.any(velocity > drone.max_velocity_norm))

    if nan_detected:
        max_e2 = float("inf")
        final_e2 = float("inf")
    else:
        max_e2 = float(np.max(np.abs(e2_series)))
        final_e2 = float(np.abs(e2_series[-1]))

    return RolloutMetrics(
        max_e2=max_e2,
        final_e2=final_e2,
        nan_detected=nan_detected,
        velocity_exploded=velocity_exploded,
    )


# ---------------------------------------------------------------------------
# Критерий стабильности
# ---------------------------------------------------------------------------

def is_stable(metrics: RolloutMetrics, drone: Optional[QuadModel] = None) -> bool:
    """Проверить стабильность ролаута по параметрам дрона.

    Критерии (все должны выполняться):
        1. drone.nan_is_failure and nan_detected → нестабильно
        2. max_e2 > drone.lateral_error_limit    → нестабильно
        3. velocity_exploded                      → нестабильно

    Параметры:
        metrics — RolloutMetrics из rollout_with_speed
        drone   — QuadModel с лётными ограничениями; None → нормализованная модель

    Возвращает:
        True если стабильно, False иначе
    """
    if drone is None:
        drone = QuadModel()

    if drone.nan_is_failure and metrics["nan_detected"]:
        return False

    if metrics["max_e2"] > drone.lateral_error_limit:
        return False

    if metrics["velocity_exploded"]:
        return False

    return True


# ---------------------------------------------------------------------------
# Поиск оптимальной скорости
# ---------------------------------------------------------------------------

def find_optimal_speed(
    state: Optional[np.ndarray],
    curve: Union[CurveGeom, Callable[[float], np.ndarray]],
    drone: Optional[QuadModel] = None,
    oracle_cfg: Optional[OracleConfig] = None,
    coarse_to_fine: bool = False,
    dt: float = ORACLE_DT,
    kappa: float = ORACLE_KAPPA,
    zeta0: float = 0.0,
) -> float:
    """Найти максимальную стабильную параметрическую скорость V*.

    Диапазон: V ∈ [drone.min_speed, drone.max_speed].
    Возвращаемое значение никогда не превышает drone.max_speed.
    Если ни одно V не стабильно — возвращает drone.min_speed.
    Функция детерминирована: при одинаковых входах всегда даёт одинаковый результат.

    Режимы поиска (управляются флагом coarse_to_fine):

    coarse_to_fine=False (по умолчанию):
        Линейный перебор с шагом oracle_cfg.speed_step (= 0.3).
        Проверяет все V по возрастанию, запоминает последнее стабильное.

    coarse_to_fine=True:
        Двухпроходный поиск:
          1. Грубый проход: шаг oracle_cfg.coarse_step (= 0.5) по всему диапазону.
             Находит наибольшую стабильную скорость best_coarse.
          2. Точный проход: шаг oracle_cfg.fine_step (= 0.1) в окне
             [best_coarse − coarse_step, best_coarse + coarse_step] ∩ [min, max].
             Возвращает наибольшую стабильную скорость из этого окна.

    Параметры:
        state          — начальное состояние (16D) или None (старт с кривой)
        curve          — CurveGeom или Callable p(s) -> ndarray[3]
        drone          — QuadModel; None → умолчания QuadModel()
        oracle_cfg     — OracleConfig; None → умолчания OracleConfig()
        coarse_to_fine — False: шаг speed_step; True: двухпроходный 0.5/0.1
        dt, kappa      — параметры интегратора

    Возвращает:
        float — оптимальная V* ∈ [drone.min_speed, drone.max_speed]
    """
    if drone is None:
        drone = QuadModel()
    if oracle_cfg is None:
        oracle_cfg = OracleConfig()

    # Кривую оборачиваем один раз — все ролаутам используют один объект
    if not isinstance(curve, CurveGeom):
        curve = make_curve(curve)

    def _rollout(V: float) -> bool:
        """Выполнить ролаут и вернуть True если стабильно."""
        V = min(float(V), drone.max_speed)
        m = rollout_with_speed(state, curve, V, oracle_cfg.rollout_horizon,
                               drone=drone, dt=dt, kappa=kappa, zeta0=zeta0)
        return is_stable(m, drone)

    def _clipped_arange(lo: float, hi: float, step: float) -> np.ndarray:
        """np.arange(lo, hi, step), обрезанный по hi включительно."""
        eps = step * 0.5
        arr = np.arange(lo, hi + eps, step)
        return arr[arr <= hi + eps]

    # ------------------------------------------------------------------
    # Режим 1: линейный перебор с шагом speed_step
    # ------------------------------------------------------------------
    if not coarse_to_fine:
        speeds = _clipped_arange(drone.min_speed, drone.max_speed,
                                 oracle_cfg.speed_step)
        best = drone.min_speed
        for V in speeds:
            if _rollout(V):
                best = float(V)
        return min(best, drone.max_speed)

    # ------------------------------------------------------------------
    # Режим 2: coarse-to-fine (шаг 0.5 → окно → шаг 0.1)
    # ------------------------------------------------------------------

    # --- Грубый проход ---
    coarse_speeds = _clipped_arange(drone.min_speed, drone.max_speed,
                                    oracle_cfg.coarse_step)
    best_coarse: Optional[float] = None
    for V in coarse_speeds:
        if _rollout(V):
            best_coarse = float(V)

    if best_coarse is None:
        return drone.min_speed

    # --- Точный проход в окне вокруг best_coarse ---
    fine_lo = max(drone.min_speed, best_coarse - oracle_cfg.coarse_step)
    fine_hi = min(drone.max_speed, best_coarse + oracle_cfg.coarse_step)
    fine_speeds = _clipped_arange(fine_lo, fine_hi, oracle_cfg.fine_step)

    best_fine = drone.min_speed
    for V in fine_speeds:
        if _rollout(V):
            best_fine = float(V)

    return min(best_fine, drone.max_speed)
