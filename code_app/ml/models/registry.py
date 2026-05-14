"""Реестр моделей предсказания V* и единый интерфейс сохранения/загрузки.

Кодовые имена:
    'mlp'  — SpeedMLP  (простой полносвязный, supervised MSE)
    'sac'  — SpeedSAC  (стохастический актор-критик, Gaussian NLL + entropy)
    'td3'  — SpeedTD3  (детерминированный актор, двойные критики, delayed update)
    'ppo'  — SpeedPPO  (Gaussian policy + value, PPO clip offline)

Использование::

    from ml.models.registry import get_speed_model, SpeedPredictorAny

    model = get_speed_model("sac", max_speed=10.0)   # необученный экземпляр
    predictor = SpeedPredictorAny.load("path/to/sac.pt")
    v = predictor.predict(feature_vector(...))
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from drone_sim.models.quad_model import QuadModel
from ml.models.speed_model import SpeedMLP, INPUT_SIZE
from ml.models.sac_model import SpeedSAC
from ml.models.td3_model import SpeedTD3
from ml.models.ppo_model import SpeedPPO

# ---------------------------------------------------------------------------
# Реестр
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": SpeedMLP,
    "sac": SpeedSAC,
    "td3": SpeedTD3,
    "ppo": SpeedPPO,
}

MODEL_NAMES: list[str] = list(_MODEL_REGISTRY)


def get_speed_model(
    name: str,
    max_speed: float = 10.0,
    input_size: int = INPUT_SIZE,
    **kwargs,
) -> nn.Module:
    """Создать новый необученный экземпляр модели по кодовому имени.

    Параметры:
        name       — кодовое имя: 'mlp', 'sac', 'td3', 'ppo'
        max_speed  — верхняя граница предсказания V*
        input_size — размер вектора признаков (обычно 7)

    Пример::

        model = get_speed_model("td3")   # SpeedTD3 с умолчаниями
        model = get_speed_model("sac", max_speed=5.0)
    """
    key = name.lower().strip()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Неизвестная модель: {name!r}. Доступные: {MODEL_NAMES}"
        )
    return _MODEL_REGISTRY[key](max_speed=max_speed, input_size=input_size, **kwargs)


# ---------------------------------------------------------------------------
# Единое сохранение / загрузка (с полем model_type)
# ---------------------------------------------------------------------------

def save_speed_model_any(
    model: nn.Module,
    path: str,
    drone: QuadModel | None = None,
) -> None:
    """Сохранить модель любого типа в .pt файл.

    Записывает поле ``model_type`` в чекпоинт, чтобы load_speed_model_any()
    мог восстановить правильный класс без дополнительных аргументов.
    """
    if drone is None:
        drone = QuadModel()

    drone_params = {
        "min_speed":               drone.min_speed,
        "max_speed":               drone.max_speed,
        "lateral_error_limit":     drone.lateral_error_limit,
        "tangential_error_limit":  drone.tangential_error_limit,
        "max_velocity_norm":       drone.max_velocity_norm,
    }

    model_type = getattr(model, "CODE_NAME", "mlp")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type":   model_type,
            "state_dict":   model.state_dict(),
            "max_speed":    model.max_speed,
            "input_size":   model.input_size,
            "drone_params": drone_params,
        },
        path,
    )


def load_speed_model_any(
    path: str,
    device: str = "cpu",
) -> nn.Module:
    """Загрузить модель любого типа из чекпоинта.

    Автоматически определяет тип по полю ``model_type``.
    Если поля нет (старый формат без model_type) — возвращает SpeedMLP.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model_type = checkpoint.get("model_type", "mlp")
    if model_type not in _MODEL_REGISTRY:
        warnings.warn(
            f"Неизвестный model_type={model_type!r} в {path!r}. "
            "Загружаю как SpeedMLP.",
            UserWarning,
            stacklevel=2,
        )
        model_type = "mlp"

    model = _MODEL_REGISTRY[model_type](
        max_speed=checkpoint["max_speed"],
        input_size=checkpoint["input_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def drone_from_checkpoint(path: str) -> QuadModel:
    """Восстановить QuadModel из сохранённых drone_params чекпоинта."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if "drone_params" not in checkpoint:
        warnings.warn(
            f"Чекпоинт {path!r} не содержит drone_params. "
            "Используются умолчания QuadModel().",
            UserWarning,
            stacklevel=2,
        )
        return QuadModel()

    p = checkpoint["drone_params"]
    return QuadModel(
        min_speed=p["min_speed"],
        max_speed=p["max_speed"],
        lateral_error_limit=p["lateral_error_limit"],
        tangential_error_limit=p["tangential_error_limit"],
        max_velocity_norm=p["max_velocity_norm"],
    )


# ---------------------------------------------------------------------------
# SpeedPredictorAny — обёртка инференса для любой модели
# ---------------------------------------------------------------------------

class SpeedPredictorAny:
    """Обёртка инференса, совместимая с любым типом модели из реестра.

    Интерфейс аналогичен ``SpeedPredictor`` из ml.inference.predict,
    но работает с 'mlp', 'sac', 'td3', 'ppo'.

    Пример::

        pred = SpeedPredictorAny.load("code/ml/data/saved_models/sac_model.pt")
        v = pred.predict(feature_vector(state, curve, drone=pred.drone))
    """

    def __init__(
        self,
        model: nn.Module,
        drone: QuadModel | None = None,
    ) -> None:
        self._model = model.cpu().eval()
        self._drone = drone if drone is not None else QuadModel()

    # ------------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------------

    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor, list],
    ) -> float:
        """Предсказать V* и обрезать до диапазона дрона."""
        x = self._to_tensor(features)
        with torch.no_grad():
            raw = self._model(x).item()
        return float(np.clip(raw, self._drone.min_speed, self._drone.max_speed))

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
        drone: QuadModel | None = None,
    ) -> "SpeedPredictorAny":
        """Загрузить предиктор из .pt файла любого типа."""
        model = load_speed_model_any(path, device="cpu")
        if drone is None:
            drone = drone_from_checkpoint(path)
        return cls(model=model, drone=drone)

    # ------------------------------------------------------------------
    # Внутренние
    # ------------------------------------------------------------------

    def _to_tensor(self, features) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            x = features.float().cpu()
        elif isinstance(features, np.ndarray):
            x = torch.from_numpy(features.astype(np.float32))
        else:
            x = torch.tensor(features, dtype=torch.float32)
        x = x.reshape(1, -1)
        if x.shape[1] != INPUT_SIZE:
            raise ValueError(
                f"Ожидается {INPUT_SIZE} признаков, получено {x.shape[1]}. "
                "Используйте feature_vector() из ml.dataset.features."
            )
        return x

    @property
    def drone(self) -> QuadModel:
        return self._drone

    @property
    def model_type(self) -> str:
        return getattr(self._model, "CODE_NAME", "mlp")

    def __repr__(self) -> str:
        return (
            f"SpeedPredictorAny("
            f"type={self.model_type}, "
            f"model={self._model}, "
            f"clip=[{self._drone.min_speed}, {self._drone.max_speed}])"
        )
