"""TD3 (Twin Delayed Deep Deterministic Policy Gradient) architecture for V* prediction.

Deterministic actor a(x) -> V*.
Twin Q-critics with target networks (Polyak averaging).
Training: critic regression + BC actor loss + Q-guided actor update (delayed).
Inference: forward() returns actor output.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE: int = 7
CODE_NAME: str = "td3"


def _make_critic(input_size: int, hidden: int) -> nn.Sequential:
    q_in = input_size + 1
    return nn.Sequential(
        nn.Linear(q_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class SpeedTD3(nn.Module):
    """Детерминированный актор с двойными критиками для offline V* регрессии.

    Кодовое имя: ``'td3'``.

    Целевые сети (q1_target, q2_target) используются только при обучении.
    Для инференса вызывается только forward() → self.actor.
    """

    CODE_NAME: str = "td3"

    def __init__(
        self,
        max_speed: float = 10.0,
        input_size: int = INPUT_SIZE,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.max_speed = float(max_speed)
        self.input_size = input_size

        # Deterministic actor.
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Twin Q-critics and their target copies.
        self.q1 = _make_critic(input_size, hidden)
        self.q2 = _make_critic(input_size, hidden)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Target networks не обновляются градиентами.
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Детерминированный инференс: возвращает V*."""
        return torch.sigmoid(self.actor(x)) * self.max_speed

    def q_values(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает Q1(x,a), Q2(x,a) для обучения критика."""
        xa = torch.cat([x, action], dim=-1)
        return self.q1(xa), self.q2(xa)

    def q_target_min(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """min(Q1_target, Q2_target) для Bellman-шага (TD3 clipping)."""
        xa = torch.cat([x, action], dim=-1)
        return torch.min(self.q1_target(xa), self.q2_target(xa))

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def update_targets(self, tau: float = 0.005) -> None:
        """Polyak-усреднение целевых сетей."""
        for src, tgt in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
            for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
                p_tgt.data.copy_(tau * p_src.data + (1.0 - tau) * p_tgt.data)

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
        # Count only trainable (non-frozen) params.
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"SpeedTD3(input={self.input_size}, hidden=128, "
            f"out=sigmoid*{self.max_speed}, trainable_params={total:,})"
        )
