"""
ml/dataset/features.py
========================
Извлечение признаков для baseline MLP-модели предсказания оптимальной скорости V*.

Принцип отбора признаков: простые, физически осмысленные, вычисляются из
текущего состояния и локальной геометрии кривой.

Публичный API:
    extract_features(state, curve, drone, s) -> dict[str, float]

Вспомогательные функции:
    normalize_feature(value, scale, clip)
    compute_heading_error(velocity, s, curve)
    compute_kappa(s, curve, h)
    compute_kappa_max_lookahead(s, curve, lookahead_ds, n_points, h)
    compute_de2_dt(velocity, s, curve)

---------------------------------------------------------------------------
Почему t_norm и s НЕ включены в признаки baseline-модели:

    t_norm — норма касательного вектора ||t(s)||. В текущей постановке допустимы
    только кривые с ||t|| = const (прямая, нормированный круг, спираль). Для них
    t_norm практически константа, и добавление этого признака не даёт модели
    полезной информации. Оставлено для диагностики, но не подаётся в модель.

    s — абсолютное значение параметра вдоль траектории. Если подать s как признак,
    модель может запомнить «на каком участке конкретной кривой» нужна та или иная
    скорость, вместо того чтобы учить физически обобщаемые закономерности.
    Это приводит к переобучению на отдельные траектории из датасета.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from drone_sim.geometry.curves import CurveGeom, Rz, Ry, se_from_pose
from drone_sim.models.quad_model import QuadModel

# Масштабный коэффициент для кривизны κ: для допустимых кривых κ ∈ [0, ~0.5];
# при scale=1.0 нормированные значения остаются в разумном диапазоне.
_KAPPA_SCALE: float = 1.0

# Шаг по параметру для lookahead-окна (единицы параметра s).
# Для типичных кривых с ||t||≈1..3 один шаг ≈ 0.2..0.6 м дуговой длины.
_LOOKAHEAD_DS: float = 1.0
_LOOKAHEAD_N:  int   = 5    # число точек в lookahead-окне

# Шаг конечных разностей для численного дифференцирования кривой
_H: float = 1e-5


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def normalize_feature(
    value: float,
    scale: float,
    clip: bool = True,
) -> float:
    """Нормализовать признак: value / scale, опционально clip в [-1, 1].

    Параметры:
        value — исходное значение признака
        scale — масштаб (ожидаемый физический предел)
        clip  — True: ограничить результат в [-1, 1]

    Возвращает:
        float в [-1, 1] (если clip=True) или value/scale
    """
    if scale < 1e-12:
        return 0.0
    normed = value / scale
    if clip:
        normed = float(np.clip(normed, -1.0, 1.0))
    return float(normed)


def compute_heading_error(
    velocity: np.ndarray,
    s: float,
    curve: CurveGeom,
) -> float:
    """Угол между вектором скорости дрона и касательной к кривой [рад].

    heading_error ∈ [0, π]:
        0  — дрон летит точно вдоль касательной
        π  — дрон летит в противоположном направлении

    Реализация: arccos(dot(v̂, t̂)), устойчивый к нулевой скорости.

    Параметры:
        velocity — вектор скорости дрона [vx, vy, vz]
        s        — текущий параметр ближайшей точки на кривой
        curve    — геометрия кривой

    Возвращает:
        float — угол рассогласования [рад]
    """
    v_norm = float(np.linalg.norm(velocity))
    if v_norm < 1e-9:
        # Дрон не движется — угловая ошибка неопределена; возвращаем 0
        return 0.0

    t_vec = curve.t(s)
    t_norm = float(np.linalg.norm(t_vec))
    if t_norm < 1e-9:
        return 0.0

    v_hat = velocity / v_norm
    t_hat = t_vec / t_norm
    dot = float(np.clip(np.dot(v_hat, t_hat), -1.0, 1.0))
    return float(np.arccos(dot))


def compute_kappa(
    s: float,
    curve: CurveGeom,
    h: float = _H,
) -> float:
    """3D кривизна кривой в точке s: κ = ||p' × p''|| / ||p'||³.

    Использует уже вычисленный tangent t(s) = p'(s) и конечную разность
    для p''(s) ≈ (t(s+h) - t(s-h)) / (2h).

    Параметры:
        s     — параметрическое значение
        curve — геометрия кривой (CurveGeom)
        h     — шаг конечной разности

    Возвращает:
        float — кривизна (≥ 0); для прямой = 0
    """
    t1 = curve.t(s)
    t2 = (curve.t(s + h) - curve.t(s - h)) / (2.0 * h)  # ≈ p''(s)

    t1_norm = float(np.linalg.norm(t1))
    if t1_norm < 1e-12:
        return 0.0

    cross = np.cross(t1, t2)
    return float(np.linalg.norm(cross)) / (t1_norm ** 3)


def compute_kappa_max_lookahead(
    s: float,
    curve: CurveGeom,
    lookahead_ds: float = _LOOKAHEAD_DS,
    n_points: int = _LOOKAHEAD_N,
    h: float = _H,
) -> float:
    """Максимальная кривизна на lookahead-окне [s, s + lookahead_ds].

    Берёт n_points равномерно распределённых точек вперёд по параметру
    и возвращает наибольшую кривизну среди них. Позволяет модели «видеть»
    предстоящие повороты и заблаговременно снижать скорость.

    Параметры:
        s           — текущий параметр ближайшей точки
        curve       — геометрия кривой
        lookahead_ds — длина окна в единицах параметра
        n_points    — число точек выборки (включая начало окна)
        h           — шаг конечной разности для compute_kappa

    Возвращает:
        float — max κ в [s, s + lookahead_ds]
    """
    points = np.linspace(s, s + lookahead_ds, n_points)
    kappas = [compute_kappa(sp, curve, h) for sp in points]
    return float(np.max(kappas))


def compute_de2_dt(
    velocity: np.ndarray,
    s: float,
    curve: CurveGeom,
) -> float:
    """Производная поперечной ошибки e2 по времени: de2/dt.

    Вычисляется аналитически из вектора скорости дрона без конечных разностей.

    Вывод:
        e2 — третья компонента q = Ry(β)ᵀ Rz(α)ᵀ (p - p_s).
        При малом изменении α, β по времени:
            de2/dt ≈ [Ry(β)ᵀ Rz(α)ᵀ v]₂  (проекция скорости на ось e2)

    Параметры:
        velocity — вектор скорости [vx, vy, vz]
        s        — параметр ближайшей точки на кривой
        curve    — геометрия кривой

    Возвращает:
        float — скорость изменения поперечной ошибки [м/с]
    """
    alpha    = curve.yaw_star(s)
    beta_val = curve.beta(s)
    # Проекция скорости в систему координат Френе (то же преобразование, что в se_from_pose)
    q = Ry(beta_val).T @ (Rz(alpha).T @ velocity)
    return float(q[2])


# ---------------------------------------------------------------------------
# Основная функция извлечения признаков
# ---------------------------------------------------------------------------

def extract_features(
    state: np.ndarray,
    curve: CurveGeom,
    drone: Optional[QuadModel] = None,
    s: float = 0.0,
) -> dict[str, float]:
    """Извлечь признаки для baseline MLP-модели предсказания V*.

    Признаки нормированы примерно в [-1, 1] для устойчивого обучения MLP.
    Все признаки — локальные (текущая точка + ближайшее lookahead-окно),
    без истории и sequence-зависимостей.

    Параметры:
        state — 16-мерный вектор состояния квадрокоптера:
                [x,y,z, vx,vy,vz, φ,θ,ψ, φ̇,θ̇,ψ̇, u1_bar,ρ1,u2,ρ2]
        curve — геометрия кривой (CurveGeom)
        drone — QuadModel с лётными ограничениями; None → умолчания
        s     — параметр ближайшей точки на кривой (из NearestPointObserver
                или аналитической nearest_fn); по умолчанию 0.0

    Возвращает:
        dict[str, float] — нормированные признаки:
            e1                 — тангенциальная ошибка
            e2                 — поперечная ошибка  ← наиболее важная
            de2_dt             — производная поперечной ошибки
            v_norm             — нормированная скорость
            heading_error      — угловое рассогласование [0, 1]
            kappa              — кривизна в текущей точке
            kappa_max_lookahead— максимальная кривизна в lookahead-окне
    """
    if drone is None:
        drone = QuadModel()

    state = np.asarray(state, dtype=float)

    # --- Извлечение компонент из вектора состояния ---
    p_xyz    = state[0:3]   # положение дрона
    velocity = state[3:6]   # линейная скорость [vx, vy, vz]

    # --- Ошибки в системе координат Френе ---
    # se_from_pose → (s_local, e1, e2); e1 — тангенциальная, e2 — поперечная
    _, e1_raw, e2_raw = se_from_pose(p_xyz, s, curve)

    # e1: тангенциальная ошибка — насколько дрон отстаёт/опережает вдоль кривой
    # Нормируем через tangential_error_limit; clip — защита от выбросов
    e1 = normalize_feature(e1_raw, drone.tangential_error_limit)

    # e2: поперечная ошибка — ключевой признак; при большом e2 скорость надо снижать
    e2 = normalize_feature(e2_raw, drone.lateral_error_limit)

    # de2_dt: растёт ли поперечная ошибка или уменьшается?
    # При одинаковом e2 это разные ситуации: растущая ошибка требует меньшей скорости
    # Нормируем через max_velocity_norm — физический максимум de2/dt
    de2_dt_raw = compute_de2_dt(velocity, s, curve)
    de2_dt = normalize_feature(de2_dt_raw, drone.max_velocity_norm)

    # v_norm: текущая скорость дрона, нормированная через max_speed
    # Помогает модели различать «уже летим быстро» vs «только разгоняемся»
    v_norm_raw = float(np.linalg.norm(velocity))
    v_norm = normalize_feature(v_norm_raw, drone.max_speed)

    # heading_error: угол между вектором скорости и касательной к кривой
    # Если дрон летит «не туда» — нужно снижать скорость; важен для безопасности
    # Нормируем через π → результат в [0, 1]
    heading_error_rad = compute_heading_error(velocity, s, curve)
    heading_error = normalize_feature(heading_error_rad, np.pi, clip=False)

    # kappa: кривизна в текущей точке траектории
    # На крутых поворотах допустимая скорость ниже
    # Нормируем через _KAPPA_SCALE = 1.0; clip в [0, 1] (κ ≥ 0)
    kappa_raw = compute_kappa(s, curve)
    kappa = float(np.clip(kappa_raw / _KAPPA_SCALE, 0.0, 1.0))

    # kappa_max_lookahead: максимальная кривизна в окне [s, s + lookahead_ds]
    # Предупреждает модель о предстоящих поворотах — важно для упреждающего снижения V
    kappa_la_raw = compute_kappa_max_lookahead(s, curve)
    kappa_max_lookahead = float(np.clip(kappa_la_raw / _KAPPA_SCALE, 0.0, 1.0))

    return {
        "e1":                  e1,
        "e2":                  e2,
        "de2_dt":              de2_dt,
        "v_norm":              v_norm,
        "heading_error":       heading_error,
        "kappa":               kappa,
        "kappa_max_lookahead": kappa_max_lookahead,
    }


def feature_vector(
    state: np.ndarray,
    curve: CurveGeom,
    drone: Optional[QuadModel] = None,
    s: float = 0.0,
) -> np.ndarray:
    """Вернуть признаки как numpy-вектор (для подачи в MLP).

    Порядок признаков фиксирован: e1, e2, de2_dt, v_norm,
    heading_error, kappa, kappa_max_lookahead.

    Параметры:
        state, curve, drone, s — те же, что в extract_features

    Возвращает:
        ndarray shape (7,) — нормированные признаки
    """
    d = extract_features(state, curve, drone=drone, s=s)
    return np.array([
        d["e1"],
        d["e2"],
        d["de2_dt"],
        d["v_norm"],
        d["heading_error"],
        d["kappa"],
        d["kappa_max_lookahead"],
    ], dtype=float)
