from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Minimal LSTM model for sequence regression.

    Architecture:
        LSTM → last hidden state → Linear → scalar output

    Design goals:
    - No unnecessary complexity
    - Stable training
    - Easy to reason about
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, seq_len, input_dim)

        Returns:
            shape (batch_size,)
        """
        out, _ = self.lstm(x)

        # Take last timestep output
        last = out[:, -1, :]  # (batch_size, hidden_dim)

        out = self.fc(last)   # (batch_size, 1)

        return out.squeeze(-1)
