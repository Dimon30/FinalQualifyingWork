"""Inference wrapper for ``SpeedMLP``."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch

from drone_sim.models.quad_model import QuadModel
from ml.models.speed_model import (
    INPUT_SIZE, SpeedMLP,
    load_speed_model, load_drone_params_from_checkpoint, save_speed_model,
)

# Default model location (relative to project root).
_DEFAULT_MODELS_DIR = "code_app/ml/data/saved_models"
_DEFAULT_MODEL_NAME = "speed_model.pt"
DEFAULT_MODEL_PATH: str = "code_app/ml/data/saved_models/speed_model.pt"


class SpeedPredictor:
    """Inference wrapper with CPU execution and output clipping."""

    def __init__(
        self,
        model: SpeedMLP,
        drone: QuadModel | None = None,
    ) -> None:
        self._model = model.cpu().eval()
        self._drone = drone if drone is not None else QuadModel()

    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor, list],
    ) -> float:
        """Predict the target speed and clip it to drone limits."""
        x = self._to_tensor(features)  # shape (1, INPUT_SIZE)
        with torch.no_grad():
            raw = self._model(x).item()

        return float(np.clip(raw, self._drone.min_speed, self._drone.max_speed))

    def save(self, path: str | None = None) -> str:
        """Сохранить модель вместе с параметрами дрона.

        Параметры дрона берутся из self._drone — того, с которым создан предиктор.
        При последующем load() QuadModel восстановится автоматически.
        """
        if path is None:
            path = str(Path(_DEFAULT_MODELS_DIR) / _DEFAULT_MODEL_NAME)

        save_speed_model(self._model, path, drone=self._drone)
        return path

    @classmethod
    def default(cls) -> "SpeedPredictor":
        """Загрузить модель из стандартного пути проекта (``code_app/ml/data/saved_models/speed_model.pt``).

        Параметры дрона восстанавливаются из чекпоинта автоматически.
        Если файл не найден — бросает ``FileNotFoundError``.
        """
        if not Path(DEFAULT_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Модель по умолчанию не найдена: {DEFAULT_MODEL_PATH}\n"
                "Сначала запустите:\n"
                "  python code_app/scenarios/run_build_dataset.py --curves 10 --samples 20\n"
                "  python code_app/scenarios/train_speed_model.py"
            )
        return cls.load(DEFAULT_MODEL_PATH)

    @classmethod
    def load(
        cls,
        path: str,
        drone: QuadModel | None = None,
    ) -> "SpeedPredictor":
        """Загрузить предиктор из файла.

        Параметры дрона восстанавливаются из чекпоинта автоматически.
        Аргумент `drone` — опциональный override: если передан, его параметры
        используются вместо сохранённых, но при расхождении будет предупреждение.

        Параметры:
            path  — путь к .pt файлу
            drone — override QuadModel; None → использовать из чекпоинта
        """
        model = load_speed_model(path, device="cpu")

        # Читаем drone_params из чекпоинта (backward-compat: предупреждение если нет)
        saved_params = load_drone_params_from_checkpoint(path)

        if drone is None:
            # Восстанавливаем QuadModel из чекпоинта — основной путь
            drone = QuadModel(
                min_speed=saved_params["min_speed"],
                max_speed=saved_params["max_speed"],
                lateral_error_limit=saved_params["lateral_error_limit"],
                tangential_error_limit=saved_params["tangential_error_limit"],
                max_velocity_norm=saved_params["max_velocity_norm"],
            )
        else:
            # drone передан явно — проверить на расхождение с сохранёнными
            _warn_drone_mismatch(drone, saved_params, path)

        return cls(model=model, drone=drone)

    def _to_tensor(self, features: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        """Convert input features to a CPU tensor of shape ``(1, INPUT_SIZE)``."""
        if isinstance(features, torch.Tensor):
            x = features.float().cpu()
        elif isinstance(features, np.ndarray):
            x = torch.from_numpy(features.astype(np.float32))
        else:
            x = torch.tensor(features, dtype=torch.float32)

        x = x.reshape(1, -1)

        if x.shape[1] != INPUT_SIZE:
            raise ValueError(
                f"Expected {INPUT_SIZE} features, got {x.shape[1]}. "
                f"Use feature_vector() from ml.dataset.features to build the input."
            )
        return x

    @property
    def drone(self) -> QuadModel:
        """QuadModel, используемый предиктором (восстановлен из чекпоинта или передан явно)."""
        return self._drone

    def __repr__(self) -> str:
        return (
            f"SpeedPredictor("
            f"model={self._model}, "
            f"clip=[{self._drone.min_speed}, {self._drone.max_speed}], "
            f"lateral_e_lim={self._drone.lateral_error_limit})"
        )


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _warn_drone_mismatch(drone: QuadModel, saved_params: dict, path: str) -> None:
    """Предупредить если явно переданный drone расходится с сохранёнными параметрами."""
    mismatches = []
    checks = [
        ("min_speed",               drone.min_speed),
        ("max_speed",               drone.max_speed),
        ("lateral_error_limit",     drone.lateral_error_limit),
        ("tangential_error_limit",  drone.tangential_error_limit),
        ("max_velocity_norm",       drone.max_velocity_norm),
    ]
    for key, actual in checks:
        expected = saved_params.get(key)
        if expected is not None and abs(actual - expected) > 1e-9:
            mismatches.append(f"  {key}: чекпоинт={expected}, drone={actual}")

    if mismatches:
        lines = "\n".join(mismatches)
        warnings.warn(
            f"SpeedPredictor.load('{path}'): параметры дрона расходятся с чекпоинтом.\n"
            f"{lines}\n"
            "Нормировка признаков (feature_vector) должна использовать те же значения, "
            "что и при сборке датасета. Убедитесь, что drone совпадает с тем, "
            "который передавался в generate_dataset() и train().",
            UserWarning,
            stacklevel=3,
        )
