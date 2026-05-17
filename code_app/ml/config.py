"""Конфигурация ML-пайплайна: константы и dataclass-параметры."""
from __future__ import annotations
from dataclasses import dataclass

# Ограничения скорости
MIN_SPEED: float = 0.1
MAX_SPEED: float = 10.0
SPEED_STEP: float = 0.1

# Параметры oracle-симуляции
# dt=0.005, kappa=100 → |1 − κ·a1·dt| = 0.5 < 1, наблюдатель сходится за ~5 шагов.
ORACLE_T: float = 20.0
ORACLE_DT: float = 0.005
ORACLE_KAPPA: float = 100.0
ORACLE_E_MAX: float = 1.5

# Параметры генерации датасета (дефолты для CLI)
N_CURVES: int = 200
N_SAMPLES_PER_CURVE: int = 5
DATASET_FILE: str = "ml/data/dataset.npz"

# Обучение MLP
MLP_HIDDEN: tuple = (64, 64)
MLP_LR: float = 1e-3
MLP_EPOCHS: int = 200
MLP_BATCH: int = 64
MODEL_FILE: str = "ml/data/vstar_model.pt"

DEFAULT_MODEL_PATH: str = "code_app/ml/data/saved_models/speed_model.pt"


def auto_rollout_horizon(
    s_start: float,
    s_end: float,
    n_samples: int,
    min_speed: float,
    dt: float = ORACLE_DT,
    safety: float = 1.5,
    min_steps: int = 100,
) -> int:
    """Минимальный горизонт ролаута: дрон должен пройти хотя бы одну секцию между sample-точками.

    Использует min_speed как консервативную оценку (медленная скорость → больше шагов).
    """
    import math
    section = (s_end - s_start) / max(n_samples - 1, 1)
    steps = section / max(min_speed, 1e-6) / max(dt, 1e-9)
    return max(min_steps, math.ceil(steps * safety))


@dataclass
class OracleConfig:
    """Параметры oracle-поиска V*: горизонт + шаги (linear / coarse-to-fine)."""

    rollout_horizon: int = 200   # переопределяется auto_rollout_horizon()
    speed_step: float = 0.3
    coarse_step: float = 0.5
    fine_step: float = 0.1
    min_stable_steps: int = 10


@dataclass
class MLConfig:
    """Полная конфигурация ML-пайплайна с дефолтами из модульных констант."""

    min_speed: float = MIN_SPEED
    max_speed: float = MAX_SPEED
    speed_step: float = SPEED_STEP

    oracle_T: float = ORACLE_T
    oracle_dt: float = ORACLE_DT
    oracle_kappa: float = ORACLE_KAPPA
    oracle_e_max: float = ORACLE_E_MAX

    n_curves: int = N_CURVES
    n_samples_per_curve: int = N_SAMPLES_PER_CURVE
    dataset_file: str = DATASET_FILE

    mlp_hidden: tuple = MLP_HIDDEN
    mlp_lr: float = MLP_LR
    mlp_epochs: int = MLP_EPOCHS
    mlp_batch: int = MLP_BATCH
    model_file: str = MODEL_FILE
