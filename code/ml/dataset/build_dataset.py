"""
ml/dataset/build_dataset.py
============================
Генерация датасета для обучения baseline MLP-модели предсказания V*.

Пайплайн на одну кривую:
    1. Сгенерировать кривую через ml.curves.generator
    2. Проверить validate_curve (||t|| = const, ||t|| in [1, 5])
    3. Выбрать N точек вдоль параметра s
    4. Для каждой точки:
         a. Сформировать начальное состояние дрона на кривой
         b. Извлечь признаки (extract_features)
         c. Найти оптимальную скорость (find_optimal_speed)
         d. Верифицировать стабильность при V_opt
         e. Сохранить запись в буфер
    5. Сохранить все записи в CSV

Формат CSV:
    e1, e2, de2_dt, v_norm, heading_error, kappa, kappa_max_lookahead, s, t_norm, V_opt

    Модельные признаки (входы MLP):
        e1                 -- тангенциальная ошибка (норм.)
        e2                 -- поперечная ошибка (норм.)
        de2_dt             -- производная поперечной ошибки (норм.)
        v_norm             -- норма скорости (норм.)
        heading_error      -- ошибка направления [0, 1]
        kappa              -- кривизна в текущей точке (норм.)
        kappa_max_lookahead-- макс. кривизна в lookahead-окне (норм.)
    Диагностические поля (не подаются в модель):
        s                  -- параметрическое значение
        t_norm             -- норма касательной ||t(s)||
    Целевая переменная:
        V_opt              -- оптимальная скорость V* ∈ [min_speed, max_speed]

Публичный API:
    generate_dataset(num_curves, out_path, seed, n_samples_per_curve,
                     s_start, s_end, drone, oracle_cfg) -> str
"""
from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from drone_sim.models.quad_model import QuadModel
from ml.config import OracleConfig, ORACLE_DT, ORACLE_KAPPA
from ml.curves.generator import generate_curve, CurveSpec
from ml.dataset.curve_generator import validate_curve
from ml.dataset.features import extract_features
from ml.dataset.simulator_wrapper import (
    find_optimal_speed,
    rollout_with_speed,
    is_stable,
)

# ---------------------------------------------------------------------------
# Настройка логирования
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Колонки CSV в фиксированном порядке.
# Первые 7 — модельные признаки (входы MLP).
# s, t_norm — диагностика (не подаются в модель).
# V_opt — целевая переменная.
_CSV_COLUMNS = [
    "e1", "e2", "de2_dt", "v_norm", "heading_error",
    "kappa", "kappa_max_lookahead",
    "s", "t_norm",
    "V_opt",
]

# Диапазон параметра s для выборки точек на кривой по умолчанию
_DEFAULT_S_START: float = 0.0
_DEFAULT_S_END:   float = 15.0

# Путь к CSV по умолчанию (относительно корня проекта)
_DEFAULT_OUT_PATH: str = "code/ml/data/dataset.csv"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _make_state_on_curve(spec: CurveSpec, s: float) -> np.ndarray:
    """Создать 16-мерное состояние: дрон на кривой в точке s, скорость = 0.

    Параметры:
        spec — CurveSpec с геометрией кривой
        s    — параметрическое значение точки

    Возвращает:
        ndarray shape (16,) — начальное состояние
    """
    x0 = np.zeros(16, dtype=float)
    x0[0:3] = spec.curve.p(s)   # положение = точка на кривой
    # x0[3:6] = 0 — нулевая начальная скорость
    # Остальные поля (углы, состояния интеграторов) = 0
    return x0


def _build_record(
    feats: dict,
    s: float,
    t_norm: float,
    V_opt: float,
) -> dict:
    """Собрать одну строку CSV из признаков и V_opt.

    feats — словарь из extract_features (7 модельных признаков).
    s, t_norm — диагностические поля, не входы модели.
    V_opt — целевая переменная.
    """
    return {
        "e1":                  feats["e1"],
        "e2":                  feats["e2"],
        "de2_dt":              feats["de2_dt"],
        "v_norm":              feats["v_norm"],
        "heading_error":       feats["heading_error"],
        "kappa":               feats["kappa"],
        "kappa_max_lookahead": feats["kappa_max_lookahead"],
        "s":                   round(s, 6),
        "t_norm":              round(t_norm, 6),
        "V_opt":               round(V_opt, 6),
    }


def _save_csv(records: list[dict], out_path: str) -> None:
    """Сохранить список записей в CSV-файл."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def generate_dataset(
    num_curves: int = 20,
    out_path: str = _DEFAULT_OUT_PATH,
    seed: int = 42,
    n_samples_per_curve: int = 5,
    s_start: float = _DEFAULT_S_START,
    s_end:   float = _DEFAULT_S_END,
    drone:      Optional[QuadModel]  = None,
    oracle_cfg: Optional[OracleConfig] = None,
    coarse_to_fine: bool = False,
) -> str:
    """Сгенерировать датасет и сохранить в CSV.

    Пайплайн:
        Для каждой из num_curves кривых:
            1. Сгенерировать кривую (line / circle / spiral)
            2. Проверить validate_curve — если невалидна, пропустить
            3. Выбрать n_samples_per_curve точек s в [s_start, s_end]
            4. Для каждого s:
                a. Сформировать начальное состояние на кривой
                b. Извлечь признаки (extract_features)
                c. Найти V_opt (find_optimal_speed)
                d. Верифицировать: откатить ролаут при V_opt, проверить is_stable
                e. Если стабильно — записать; иначе пропустить
        Сохранить все записи в CSV.

    Параметры:
        num_curves          -- число кривых (не все могут пройти validate_curve)
        out_path            -- путь к выходному CSV
        seed                -- seed для воспроизводимости
        n_samples_per_curve -- число точек s на кривую
        s_start, s_end      -- диапазон параметра s
        drone               -- QuadModel; None -> умолчания QuadModel()
        oracle_cfg          -- OracleConfig; None -> умолчания OracleConfig()
        coarse_to_fine      -- передаётся в find_optimal_speed

    Возвращает:
        str -- путь к сохранённому CSV
    """
    if drone is None:
        drone = QuadModel()
    if oracle_cfg is None:
        oracle_cfg = OracleConfig()

    rng = np.random.default_rng(seed)
    curve_types = ["line", "circle", "spiral"]
    type_weights = np.array([0.2, 0.3, 0.5])

    records: list[dict] = []
    stats = {
        "attempted":          0,
        "invalid_curve":      0,
        "points_total":       0,
        "points_unstable":    0,
        "points_saved":       0,
    }

    t0_total = time.monotonic()
    log.info("Dataset generation started: %d curves requested", num_curves)
    log.info(
        "Drone limits: min_speed=%.2f  max_speed=%.2f  lateral_e_limit=%.2f",
        drone.min_speed, drone.max_speed, drone.lateral_error_limit,
    )
    log.info(
        "Oracle: horizon=%d  step=%.2f  coarse_to_fine=%s",
        oracle_cfg.rollout_horizon, oracle_cfg.speed_step, coarse_to_fine,
    )

    for curve_idx in range(num_curves):
        stats["attempted"] += 1
        t0_curve = time.monotonic()

        # --- 1. Генерация кривой ---
        curve_type = str(rng.choice(curve_types, p=type_weights))
        spec: CurveSpec = generate_curve(curve_type, rng=rng)

        # --- 2. Валидация (||t|| = const, ||t|| in [1, 5]) ---
        if not validate_curve(spec.curve.p, s_range=(s_start, s_end)):
            stats["invalid_curve"] += 1
            log.warning(
                "[%d/%d] Curve #%d (%s) failed validate_curve -- skipping",
                curve_idx + 1, num_curves, curve_idx, curve_type,
            )
            continue

        t_norm_val = float(spec.tangent_norm)

        # --- 3. Точки вдоль s ---
        s_values = np.linspace(s_start, s_end, n_samples_per_curve)
        curve_records: list[dict] = []

        for s in s_values:
            s = float(s)
            stats["points_total"] += 1

            # --- 4a. Начальное состояние ---
            state = _make_state_on_curve(spec, s)

            # --- 4b. Признаки ---
            feats = extract_features(state, spec.curve, drone=drone, s=s)

            # --- 4c. Оптимальная скорость ---
            # zeta0=s: NearestPointObserver стартует из той же точки, что и дрон,
            # иначе observer начинает с zeta=0 и долго сходится → ложные ошибки
            V_opt = find_optimal_speed(
                state, spec.curve,
                drone=drone,
                oracle_cfg=oracle_cfg,
                coarse_to_fine=coarse_to_fine,
                zeta0=s,
            )

            # Гарантируем, что V_opt в допустимом диапазоне
            V_opt = float(np.clip(V_opt, drone.min_speed, drone.max_speed))

            # --- 4d. Верификация стабильности при V_opt ---
            m = rollout_with_speed(
                state, spec.curve, V_opt,
                oracle_cfg.rollout_horizon,
                drone=drone,
                zeta0=s,
            )
            if not is_stable(m, drone):
                stats["points_unstable"] += 1
                log.debug(
                    "  s=%.2f  V_opt=%.2f  unstable (max_e2=%.4f) -- skip",
                    s, V_opt, m["max_e2"],
                )
                continue

            # --- 4e. Запись ---
            record = _build_record(feats, s, t_norm_val, V_opt)
            curve_records.append(record)
            stats["points_saved"] += 1

        records.extend(curve_records)

        elapsed = time.monotonic() - t0_curve
        log.info(
            "[%d/%d] %s  t_norm=%.3f  saved=%d/%d  V_opt_range=[%.2f, %.2f]  %.1fs",
            curve_idx + 1,
            num_curves,
            curve_type.ljust(6),
            t_norm_val,
            len(curve_records),
            n_samples_per_curve,
            min((r["V_opt"] for r in curve_records), default=float("nan")),
            max((r["V_opt"] for r in curve_records), default=float("nan")),
            elapsed,
        )

    # --- 5. Сохранение CSV ---
    _save_csv(records, out_path)

    total_elapsed = time.monotonic() - t0_total
    log.info("-" * 60)
    log.info("Done in %.1fs", total_elapsed)
    log.info("  Curves attempted : %d", stats["attempted"])
    log.info("  Invalid curves   : %d", stats["invalid_curve"])
    log.info("  Points total     : %d", stats["points_total"])
    log.info("  Points unstable  : %d", stats["points_unstable"])
    log.info("  Points saved     : %d", stats["points_saved"])
    log.info("  CSV saved to     : %s", out_path)

    return out_path


# ---------------------------------------------------------------------------
# CLI-запуск
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate V* dataset for MLP training"
    )
    parser.add_argument(
        "--num-curves", type=int, default=10,
        help="Number of curves to generate (default: 10)",
    )
    parser.add_argument(
        "--out", type=str, default=_DEFAULT_OUT_PATH,
        help=f"Output CSV path (default: {_DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Samples per curve (default: 5)",
    )
    parser.add_argument(
        "--coarse-fine", action="store_true",
        help="Use coarse-to-fine speed search (default: simple step)",
    )
    args = parser.parse_args()

    generate_dataset(
        num_curves=args.num_curves,
        out_path=args.out,
        seed=args.seed,
        n_samples_per_curve=args.samples,
        coarse_to_fine=args.coarse_fine,
    )
