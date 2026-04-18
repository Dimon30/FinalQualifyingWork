"""MLP model for target speed prediction."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# Input feature dimension.
INPUT_SIZE: int = 7


class SpeedMLP(nn.Module):
    """MLP that maps the feature vector to a target speed."""

    def __init__(
        self,
        max_speed: float = 10.0,
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
        """Run the network and scale output to ``(0, max_speed)``."""
        logit = self.net(x)
        return torch.sigmoid(logit) * self.max_speed

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference without gradients."""
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


def save_speed_model(model: SpeedMLP, path: str, drone=None) -> None:
    """Сохранить веса модели и метаданные в файл.

    Параметры:
        model — обученный SpeedMLP
        path  — путь к .pt файлу
        drone — QuadModel, параметры которого использовались при сборке датасета
                и нормировке признаков; None → используются умолчания QuadModel()

    Сохраняет `drone_params` — словарь из 5 полей, необходимых для воспроизводимого
    инференса (нормировка признаков + диапазон V*):
        min_speed, max_speed, lateral_error_limit,
        tangential_error_limit, max_velocity_norm
    """
    from drone_sim.models.quad_model import QuadModel  # локальный импорт во избежание цикличности

    if drone is None:
        drone = QuadModel()

    drone_params = {
        "min_speed":               drone.min_speed,
        "max_speed":               drone.max_speed,
        "lateral_error_limit":     drone.lateral_error_limit,
        "tangential_error_limit":  drone.tangential_error_limit,
        "max_velocity_norm":       drone.max_velocity_norm,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict":   model.state_dict(),
            "max_speed":    model.max_speed,
            "input_size":   model.input_size,
            "drone_params": drone_params,
        },
        path,
    )


def load_speed_model(
    path: str,
    max_speed: Optional[float] = None,
    device: str = "cpu",
) -> SpeedMLP:
    """Загрузить SpeedMLP из чекпоинта.

    Параметры drone_params (если есть в чекпоинте) не применяются здесь —
    они читаются отдельно через load_drone_params_from_checkpoint().
    Для полного восстановления контекста используй SpeedPredictor.load().
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    ms = max_speed if max_speed is not None else checkpoint["max_speed"]
    model = SpeedMLP(max_speed=ms, input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_drone_params_from_checkpoint(path: str) -> dict:
    """Загрузить параметры дрона, сохранённые в чекпоинте.

    Возвращает словарь с ключами:
        min_speed, max_speed, lateral_error_limit,
        tangential_error_limit, max_velocity_norm

    Если чекпоинт старого формата (без drone_params) — возвращает умолчания
    QuadModel() с предупреждением.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if "drone_params" not in checkpoint:
        warnings.warn(
            f"Чекпоинт '{path}' не содержит drone_params (старый формат). "
            "Используются умолчания QuadModel(). Пересохраните модель с drone=QuadModel(...) "
            "через save_speed_model, чтобы устранить это предупреждение.",
            UserWarning,
            stacklevel=2,
        )
        from drone_sim.models.quad_model import QuadModel
        d = QuadModel()
        return {
            "min_speed":               d.min_speed,
            "max_speed":               d.max_speed,
            "lateral_error_limit":     d.lateral_error_limit,
            "tangential_error_limit":  d.tangential_error_limit,
            "max_velocity_norm":       d.max_velocity_norm,
        }

    return dict(checkpoint["drone_params"])
