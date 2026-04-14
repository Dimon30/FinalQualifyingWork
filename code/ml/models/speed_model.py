"""
ml/models/speed_model.py
=========================
MLP-модель для предсказания оптимальной параметрической скорости V*.

Архитектура:
    Linear(7 → 128) → ReLU
    Linear(128 → 128) → ReLU
    Linear(128 → 64)  → ReLU
    Linear(64 → 1)    → sigmoid × MAX_SPEED

Выход:
    V_pred = sigmoid(logit) * drone.max_speed
    Гарантия: V_pred ∈ (0, max_speed) — никогда не превышает физический предел.

Публичный API:
    SpeedMLP(max_speed, input_size)   — класс модели
    load_speed_model(path, max_speed) — загрузить веса из файла
    save_speed_model(model, path)     — сохранить веса в файл
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# Размерность входного вектора признаков (совпадает с len(extract_features(...)))
INPUT_SIZE: int = 7


class SpeedMLP(nn.Module):
    """MLP для предсказания оптимальной скорости V* вдоль траектории.

    Параметры:
        max_speed  — физический предел скорости V*; задаёт верхнюю границу выхода.
                     Рекомендуется передавать drone.max_speed из QuadModel.
        input_size — размерность входного вектора признаков (по умолчанию 7).

    Вход:
        x : Tensor[..., input_size] — нормированные признаки из extract_features:
            [e1, e2, de2_dt, v_norm, heading_error, kappa, kappa_max_lookahead]

    Выход:
        V_pred : Tensor[..., 1] — предсказанная скорость V* ∈ (0, max_speed)

    Формула выхода:
        V_pred = sigmoid(linear_output) * max_speed

    Замечание о диапазоне:
        Теоретически sigmoid ∈ (0, 1), поэтому V_pred ∈ (0, max_speed).
        На практике float32 насыщает sigmoid до 1.0 при |logit| > ~87,
        поэтому при экстремальных входах возможно V_pred = max_speed точно.
        Для принудительного ограничения снизу используй min_speed при постобработке.
    """

    def __init__(
        self,
        max_speed: float = 3.0,
        input_size: int = INPUT_SIZE,
    ) -> None:
        super().__init__()
        self.max_speed = float(max_speed)
        self.input_size = input_size

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход.

        Параметры:
            x — Tensor shape (..., input_size)

        Возвращает:
            V_pred — Tensor shape (..., 1), значения в (0, max_speed)
        """
        logit = self.net(x)
        return torch.sigmoid(logit) * self.max_speed

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Инференс без градиентов (удобный псевдоним forward для использования вне обучения).

        Параметры:
            x — Tensor shape (..., input_size) или ndarray (конвертируется автоматически)

        Возвращает:
            V_pred — Tensor shape (..., 1) на CPU, detached
        """
        import numpy as np

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu()

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return (
            f"SpeedMLP("
            f"input={self.input_size}, "
            f"hidden=[128, 128, 64], "
            f"out=sigmoid*{self.max_speed}, "
            f"params={total:,})"
        )


# ---------------------------------------------------------------------------
# Утилиты сохранения / загрузки
# ---------------------------------------------------------------------------

def save_speed_model(model: SpeedMLP, path: str) -> None:
    """Сохранить веса модели в файл.

    Сохраняет state_dict + метаданные (max_speed, input_size) в один .pt файл.

    Параметры:
        model — обученная SpeedMLP
        path  — путь к файлу (будет создан вместе с родительскими директориями)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict":  model.state_dict(),
            "max_speed":   model.max_speed,
            "input_size":  model.input_size,
        },
        path,
    )


def load_speed_model(
    path: str,
    max_speed: Optional[float] = None,
    device: str = "cpu",
) -> SpeedMLP:
    """Загрузить модель из файла.

    Параметры:
        path      — путь к .pt файлу, сохранённому через save_speed_model
        max_speed — переопределить max_speed из чекпоинта (опционально)
        device    — устройство для загрузки ('cpu' или 'cuda')

    Возвращает:
        SpeedMLP с загруженными весами в режиме eval()
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    ms = max_speed if max_speed is not None else checkpoint["max_speed"]
    model = SpeedMLP(max_speed=ms, input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
