"""
research/deep_learning/model.py
================================
Minimal MLP for FX sentiment signal research.

Architecture::

    input → Linear(in, hidden) → ReLU → Linear(hidden, 1) → output

Usage::

    from research.deep_learning.model import MLP

    model = MLP(input_dim=7)
    out = model(x)  # shape: (batch,)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer MLP: Linear → ReLU → Linear.

    Args:
        input_dim:  Number of input features.
        hidden_dim: Width of the hidden layer (default 32).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw scalar prediction for each sample.

        Args:
            x: Float tensor of shape ``(batch, input_dim)``.

        Returns:
            Tensor of shape ``(batch,)``.
        """
        return self.net(x).squeeze(-1)
