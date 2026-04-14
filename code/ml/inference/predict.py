"""
ml/inference/predict.py
=========================
Инференс SpeedMLP в контуре управления.

Публичный API:
    SpeedPredictor               — класс-предиктор с clip(V_pred, min_speed, max_speed)
    SpeedPredictor.save(path)    — сохранить модель с весами в файл
    SpeedPredictor.load(path)    — загрузить предиктор из файла
    SpeedPredictor.predict(features) -> float

Гарантии:
    - выход всегда в [drone.min_speed, drone.max_speed]
    - работает без GPU (принудительный CPU)
    - потокобезопасен для read-only инференса (одновременный predict)

Пример использования:

    from ml.inference.predict import SpeedPredictor
    from ml.dataset.features import feature_vector

    predictor = SpeedPredictor.load("code/ml/data/model.pt")
    feat = feature_vector(state, curve, drone=drone, s=s)
    V = predictor.predict(feat)
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from drone_sim.models.quad_model import QuadModel
from ml.models.speed_model import INPUT_SIZE, SpeedMLP, load_speed_model, save_speed_model

# Папка по умолчанию для сохранения моделей
_DEFAULT_MODELS_DIR = "code/ml/data/model"
_DEFAULT_MODEL_NAME = "speed_model.pt"


class SpeedPredictor:
    """Предиктор оптимальной скорости V* на основе обученного SpeedMLP.

    Оборачивает SpeedMLP и добавляет:
        - принудительный CPU (работает без GPU)
        - clip выхода в [drone.min_speed, drone.max_speed]
        - удобный API save / load

    Параметры:
        model      — обученный SpeedMLP
        drone      — QuadModel с ограничениями скорости; None → умолчания QuadModel()

    Пример:
        predictor = SpeedPredictor.load("code/ml/data/model/speed_model.pt")
        V = predictor.predict(feature_vector(state, curve, s=s))
    """

    def __init__(
        self,
        model: SpeedMLP,
        drone: QuadModel | None = None,
    ) -> None:
        self._model = model.cpu().eval()
        self._drone = drone if drone is not None else QuadModel()

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------

    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor, list],
    ) -> float:
        """Предсказать оптимальную скорость V* и обрезать по ограничениям дрона.

        Параметры:
            features — вектор признаков длиной INPUT_SIZE (7):
                       [e1, e2, de2_dt, v_norm, heading_error, kappa, kappa_max_lookahead]
                       Принимает ndarray, Tensor или list.

        Возвращает:
            float — V* ∈ [drone.min_speed, drone.max_speed]
        """
        x = self._to_tensor(features)          # shape (1, 7)
        with torch.no_grad():
            raw = self._model(x).item()        # сырое sigmoid * max_speed

        # Clip в допустимый диапазон из параметров дрона (не константы)
        return float(np.clip(raw, self._drone.min_speed, self._drone.max_speed))

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        """Сохранить модель с весами в файл.

        Параметры:
            path — путь к .pt файлу; если None — сохраняет в папку по умолчанию
                   (_DEFAULT_MODELS_DIR / _DEFAULT_MODEL_NAME)

        Возвращает:
            str — реальный путь к сохранённому файлу
        """
        if path is None:
            path = str(Path(_DEFAULT_MODELS_DIR) / _DEFAULT_MODEL_NAME)

        save_speed_model(self._model, path)
        return path

    @classmethod
    def load(
        cls,
        path: str,
        drone: QuadModel | None = None,
    ) -> "SpeedPredictor":
        """Загрузить предиктор из .pt файла.

        Параметры:
            path  — путь к файлу, сохранённому через save() или save_speed_model()
            drone — QuadModel для clip ограничений; None → QuadModel()

        Возвращает:
            SpeedPredictor — готовый к инференсу предиктор (CPU, eval)
        """
        model = load_speed_model(path, device="cpu")
        return cls(model=model, drone=drone)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _to_tensor(self, features: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        """Конвертировать входные признаки в shape (1, INPUT_SIZE) float32 CPU Tensor."""
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

    def __repr__(self) -> str:
        return (
            f"SpeedPredictor("
            f"model={self._model}, "
            f"clip=[{self._drone.min_speed}, {self._drone.max_speed}])"
        )
