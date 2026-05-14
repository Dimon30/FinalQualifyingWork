"""Simulation helpers used by the ML pipeline."""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional, TypedDict, Union

from drone_sim import CurveGeom, SimConfig, make_curve, simulate_path_following
from drone_sim.models.quad_model import QuadModel
from ml.config import ORACLE_DT, ORACLE_KAPPA, OracleConfig


class RolloutMetrics(TypedDict):
    """Typed result of one rollout."""

    max_e2: float    # максимальная |e2| (используется is_stable)
    max_lateral: float  # максимальная sqrt(e1²+e2²) — полная поперечная ошибка
    final_e2: float
    nan_detected: bool
    velocity_exploded: bool


def rollout_with_speed(
    state: Optional[np.ndarray],
    curve: Union[CurveGeom, Callable[[float], np.ndarray]],
    V: float,
    horizon: int,
    drone: Optional[QuadModel] = None,
    dt: float = ORACLE_DT,
    kappa: float = ORACLE_KAPPA,
    zeta0: float = 0.0,
    gamma_nearest: float = 1.0,
) -> RolloutMetrics:
    """Run one rollout with a fixed target speed.

    Параметры:
        gamma_nearest  — коэффициент наблюдателя ближайшей точки.
                         Должен совпадать с CurveSpec.gamma_nearest для данной кривой:
                         γ = 0.2 / (||t||² · dt).  Дефолт 1.0 слишком мал для кривых с ||t||=1
                         при dt=0.01 — наблюдатель отстаёт на ~V*/γ, что вносит систематическую
                         ошибку курса и дестабилизирует oracle-ролаут.
    """
    if drone is None:
        drone = QuadModel()

    # Speed clamp.
    V = min(float(V), drone.max_speed)

    # Curve normalization.
    if not isinstance(curve, CurveGeom):
        curve = make_curve(curve)

    # Simulation config.
    T = horizon * dt
    cfg = SimConfig(
        Vstar=V,
        T=T,
        dt=dt,
        x0=np.asarray(state, dtype=float) if state is not None else None,
        kappa=kappa,
        quad_model=drone,
        zeta0=zeta0,
        gamma_nearest=gamma_nearest,
    )

    # Execution. Exceptions are converted to failure metrics.
    try:
        result = simulate_path_following(curve, cfg)
    except Exception:
        return RolloutMetrics(
            max_e2=float("inf"),
            max_lateral=float("inf"),
            final_e2=float("inf"),
            nan_detected=True,
            velocity_exploded=False,
        )

    # Metrics from SimResult.
    e1_series = result.errors[:, 1]   # Tangential/radial error.
    e2_series = result.errors[:, 2]   # Out-of-plane lateral error.
    velocity = result.velocity

    nan_detected = bool(
        np.any(~np.isfinite(result.x))
        or np.any(~np.isfinite(e2_series))
        or np.any(~np.isfinite(e1_series))
    )

    # Velocity limit from the drone model.
    velocity_exploded = bool(np.any(velocity > drone.max_velocity_norm))

    if nan_detected:
        max_e2 = float("inf")
        max_lateral = float("inf")
        final_e2 = float("inf")
    else:
        # Пропустить первые 10% шагов (переходный процесс наблюдателя).
        # При правильном warm-start (x1, x2 инициализированы из начального состояния)
        # наблюдатель сходится за ~5 шагов × dt=0.005 ≈ 0.025 с.
        # 10%-skip ≈ 0.1×horizon×dt даёт дополнительный запас для любых нелинейных эффектов.
        warmup = max(1, len(e2_series) // 4)   # 25% ≈ 5s для horizon=4000, совпадает с warmup_time=5s
        e1_s = e1_series[warmup:]
        e2_s = e2_series[warmup:]
        if len(e2_s) == 0:
            max_e2 = float("inf")
            max_lateral = float("inf")
        else:
            max_e2 = float(np.max(np.abs(e2_s)))
            # Полная поперечная ошибка sqrt(e1²+e2²): для кривых вида circle
            # радиальная нестабильность проявляется в e1, а e2 — вертикальная.
            lateral_series = np.sqrt(e1_s ** 2 + e2_s ** 2)
            max_lateral = float(np.max(lateral_series))
        final_e2 = float(np.abs(e2_series[-1]))

    return RolloutMetrics(
        max_e2=max_e2,
        max_lateral=max_lateral,
        final_e2=final_e2,
        nan_detected=nan_detected,
        velocity_exploded=velocity_exploded,
    )


def is_stable(metrics: RolloutMetrics, drone: Optional[QuadModel] = None) -> bool:
    """Return ``True`` if rollout metrics satisfy the configured limits.

    Проверяются:
    1. NaN/Inf в состоянии
    2. Полная поперечная ошибка sqrt(e1²+e2²) > lateral_error_limit.
       Для горизонтальных кривых (circle, line) e2 = вертикальная ошибка,
       а e1 = радиальная; нужно проверять обе.
    3. Взрыв скорости ||v|| > max_velocity_norm
    """
    if drone is None:
        drone = QuadModel()

    if drone.nan_is_failure and metrics["nan_detected"]:
        return False

    # Полная поперечная ошибка: учитывает и e1 (радиальную), и e2 (вертикальную).
    if metrics["max_lateral"] > drone.lateral_error_limit:
        return False

    if metrics["velocity_exploded"]:
        return False

    return True


def find_optimal_speed(
    state: Optional[np.ndarray],
    curve: Union[CurveGeom, Callable[[float], np.ndarray]],
    drone: Optional[QuadModel] = None,
    oracle_cfg: Optional[OracleConfig] = None,
    coarse_to_fine: bool = False,
    dt: float = ORACLE_DT,
    kappa: float = ORACLE_KAPPA,
    zeta0: float = 0.0,
    gamma_nearest: float = 1.0,
) -> float:
    """Return the largest stable target speed in the configured range."""
    if drone is None:
        drone = QuadModel()
    if oracle_cfg is None:
        oracle_cfg = OracleConfig()

    # Reuse one curve object across rollouts.
    if not isinstance(curve, CurveGeom):
        curve = make_curve(curve)

    def _rollout(V: float) -> bool:
        """Return ``True`` if the rollout at ``V`` is stable."""
        V = min(float(V), drone.max_speed)
        m = rollout_with_speed(
            state,
            curve,
            V,
            oracle_cfg.rollout_horizon,
            drone=drone,
            dt=dt,
            kappa=kappa,
            zeta0=zeta0,
            gamma_nearest=gamma_nearest,
        )
        return is_stable(m, drone)

    def _clipped_arange(lo: float, hi: float, step: float) -> np.ndarray:
        """Inclusive ``np.arange`` helper."""
        eps = step * 0.5
        arr = np.arange(lo, hi + eps, step)
        return arr[arr <= hi + eps]

    # Linear search.
    if not coarse_to_fine:
        speeds = _clipped_arange(drone.min_speed, drone.max_speed, oracle_cfg.speed_step)
        best = drone.min_speed
        for V in speeds:
            if _rollout(V):
                best = float(V)
        return min(best, drone.max_speed)

    # Coarse-to-fine search.
    coarse_speeds = _clipped_arange(
        drone.min_speed,
        drone.max_speed,
        oracle_cfg.coarse_step,
    )
    best_coarse: Optional[float] = None
    for V in coarse_speeds:
        if _rollout(V):
            best_coarse = float(V)

    if best_coarse is None:
        return drone.min_speed

    # Fine pass around the coarse optimum.
    fine_lo = max(drone.min_speed, best_coarse - oracle_cfg.coarse_step)
    fine_hi = min(drone.max_speed, best_coarse + oracle_cfg.coarse_step)
    fine_speeds = _clipped_arange(fine_lo, fine_hi, oracle_cfg.fine_step)

    best_fine = drone.min_speed
    for V in fine_speeds:
        if _rollout(V):
            best_fine = float(V)

    return min(best_fine, drone.max_speed)
