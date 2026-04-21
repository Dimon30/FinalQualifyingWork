"""SAC (Soft Actor-Critic) architecture for offline V* prediction.

Stochastic actor outputs Gaussian(mu, sigma) over V*.
Twin Q-critics estimate Q(features, action) for actor guidance.
Training: Gaussian NLL + entropy bonus + critic regression.
Inference: forward() returns mean mu (deterministic).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE: int = 7
CODE_NAME: str = "sac"


class SpeedSAC(nn.Module):
    """Stochastic actor-critic for offline V* speed regression.

    Кодовое имя: ``'sac'``.
    """

    CODE_NAME: str = "sac"

    def __init__(
        self,
        max_speed: float = 10.0,
        input_size: int = INPUT_SIZE,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.max_speed = float(max_speed)
        self.input_size = input_size

        # Actor backbone (shared for mean and log_std heads).
        self.actor_backbone = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

        # Twin Q-critics: input = [features | action].
        q_in = input_size + 1
        self.q1 = nn.Sequential(
            nn.Linear(q_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(q_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Детерминированный инференс: возвращает mean V*."""
        h = self.actor_backbone(x)
        return torch.sigmoid(self.mean_head(h)) * self.max_speed

    def forward_actor(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Вернуть (mean, std) для стохастического шага обучения."""
        h = self.actor_backbone(x)
        mean = torch.sigmoid(self.mean_head(h)) * self.max_speed
        log_std = self.log_std_head(h).clamp(-4.0, 2.0)
        return mean, torch.exp(log_std)

    def q_values(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает Q1(x,a), Q2(x,a) для обучения критика."""
        xa = torch.cat([x, action], dim=-1)
        return self.q1(xa), self.q2(xa)

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
            f"SpeedSAC(input={self.input_size}, hidden=128, "
            f"out=sigmoid*{self.max_speed}, params={total:,})"
        )
