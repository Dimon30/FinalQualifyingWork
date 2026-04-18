"""Configuration constants for the ML pipeline."""
from __future__ import annotations
from dataclasses import dataclass

# Speed limits.
MIN_SPEED: float = 0.1   # Minimum target speed V*.
MAX_SPEED: float = 10.0  # Maximum target speed V*.
SPEED_STEP: float = 0.1  # Search step used by the oracle.

# Oracle simulation parameters.
# dt=0.005 + kappa=100 — рабочая пара: |1 - k×a1×dt| = |1 - 100×5×0.005| = 0.5 < 1.
# При dt=0.01 коэффициент = 0 (нейтральный), наблюдатель осциллирует при начальных возмущениях.
# dt=0.005 → наблюдатель сходится за ~5 шагов (0.025 с) от любого начального состояния.
# Совпадает с параметрами сценария run_ch4_line.py (kappa=100, dt=0.005).
ORACLE_T: float = 20.0       # Rollout duration in seconds (legacy field, not used by OracleConfig).
ORACLE_DT: float = 0.005     # RK4 step для oracle (kappa=100 требует dt≤0.005 для сходимости).
ORACLE_KAPPA: float = 100.0  # Observer gain для oracle.
ORACLE_E_MAX: float = 1.5    # Lateral-error threshold.

# Dataset generation parameters.
N_CURVES: int = 200          # Number of curves in the dataset.
N_SAMPLES_PER_CURVE: int = 5 # Samples per curve.
DATASET_FILE: str = "ml/data/dataset.npz"

# MLP training parameters.
MLP_HIDDEN: tuple = (64, 64)       # Hidden layer sizes.
MLP_LR: float = 1e-3               # Adam learning rate.
MLP_EPOCHS: int = 200              # Training epochs.
MLP_BATCH: int = 64                # Batch size.
MODEL_FILE: str = "ml/data/vstar_model.pt"

# Стандартный путь к обученной SpeedMLP (относительно корня проекта).
# Используется SpeedPredictor.default() для загрузки модели по умолчанию.
DEFAULT_MODEL_PATH: str = "code/ml/data/saved_models/speed_model.pt"


def auto_rollout_horizon(
    s_start: float,
    s_end: float,
    n_samples: int,
    min_speed: float,
    dt: float = ORACLE_DT,
    safety: float = 1.5,
    min_steps: int = 100,
) -> int:
    """Вычислить минимальный горизонт ролаута из геометрии датасета.

    Идея: дрон должен успеть пройти хотя бы одну секцию между соседними
    sample-точками, иначе оракл не поймает нестабильность, которая
    проявляется чуть дальше по траектории.

    ``min_speed`` используется как консервативная оценка: при медленной
    скорости на один шаг тратится больше времени → нужно больше шагов.

    Параметры:
        s_start    — начало параметра кривой
        s_end      — конец параметра кривой
        n_samples  — число стартовых точек на кривую
        min_speed  — минимальная параметрическая скорость V* [drone.min_speed]
        dt         — шаг RK4 ролаута [с]
        safety     — множитель запаса (по умолчанию 1.5×)
        min_steps  — нижняя граница горизонта (на случай коротких кривых)

    Возвращает:
        Целое число шагов горизонта.

    Пример (дефолтные параметры):
        s_end=15, n_samples=10 → section=15/9≈1.67 → steps=ceil(1.67/0.3/0.01*1.5)=835
        Но min_speed=0.3 → T=1.67/0.3≈5.6с → 560 шагов × 1.5 = 840
    """
    import math
    section = (s_end - s_start) / max(n_samples - 1, 1)
    steps = section / max(min_speed, 1e-6) / max(dt, 1e-9)
    return max(min_steps, math.ceil(steps * safety))


@dataclass
class OracleConfig:
    """Parameters for oracle speed search."""

    rollout_horizon: int = 200   # 200 × dt(0.01) = 2.0 с; переопределяется auto_rollout_horizon
    speed_step: float = 0.3
    coarse_step: float = 0.5
    fine_step: float = 0.1
    min_stable_steps: int = 10


@dataclass
class MLConfig:
    """ML pipeline configuration with defaults from module constants."""

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
