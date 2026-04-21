"""PPO (Proximal Policy Optimization) architecture for offline V* prediction.

Gaussian policy (actor) outputs N(mu, sigma) over V*.
Separate value network regresses on V*.
Training: clipped surrogate loss + value MSE + entropy bonus.
Inference: forward() returns mean mu.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE: int = 7
CODE_NAME: str = "ppo"


class SpeedPPO(nn.Module):
    """Гауссовская policy + value функция для offline V* регрессии.

    Кодовое имя: ``'ppo'``.

    Политика разделяет backbone с policy_head.
    Value network — отдельная сеть (для развязки policy и value градиентов).
    log_std — обучаемый scalar-параметр (одинаковый для всего батча).
    """

    CODE_NAME: str = "ppo"

    def __init__(
        self,
        max_speed: float = 10.0,
        input_size: int = INPUT_SIZE,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.max_speed = float(max_speed)
        self.input_size = input_size

        # Policy backbone + head.
        self.policy_backbone = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, 1)
        # Обучаемый глобальный log_std (инициализация: std ≈ 1.0).
        self.log_std = nn.Parameter(torch.zeros(1))

        # Value function (отдельная сеть).
        self.value_net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Детерминированный инференс: возвращает mean V*."""
        h = self.policy_backbone(x)
        return torch.sigmoid(self.policy_head(h)) * self.max_speed

    def forward_policy(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Вернуть (mean, std) — параметры Гауссовского распределения политики."""
        h = self.policy_backbone(x)
        mean = torch.sigmoid(self.policy_head(h)) * self.max_speed
        std = torch.exp(self.log_std).clamp(min=1e-4).expand_as(mean)
        return mean, std

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Оценка value функции."""
        return self.value_net(x)

    def log_prob(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-вероятность action под текущей политикой."""
        mean, std = self.forward_policy(x)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action)

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Энтропия текущей политики (батч)."""
        _, std = self.forward_policy(x)
        return torch.distributions.Normal(torch.zeros_like(std), std).entropy()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict(self, x) -> torch.Tensor:
        """Инференс без градиентов."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu()

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return (
            f"SpeedPPO(input={self.input_size}, hidden=128, "
            f"out=sigmoid*{self.max_speed}, params={total:,})"
        )
