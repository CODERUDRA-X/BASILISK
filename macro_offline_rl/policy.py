"""
policy.py
---------
Transformer-based policy network for offline RL over macro chaos-indicator
sequences. Upgraded to support 30D observations and 4D asset allocations.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for the temporal window."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]

class TransformerChaosPolicy(nn.Module):
    """
    Input:  window of observations, shape (B, window_len, state_dim) -> default 30
    Output: action distribution (mean, log_std) for 4 assets in [-1, 1] + value head.
    """
    def __init__(
        self,
        state_dim: int = 30,  # Synced with env.OBS_DIM
        action_dim: int = 4,  # Synced with MarketSimConfig.n_assets
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        window_len: int = 16,
    ):
        super().__init__()
        self.window_len = window_len
        self.input_proj = nn.Linear(state_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=window_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.actor_mean = nn.Linear(d_model, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = nn.Linear(d_model, 1)

    def forward(self, state_window: torch.Tensor):
        x = self.input_proj(state_window)
        x = self.pos_encoding(x)
        encoded = self.encoder(x)

        last_token = encoded[:, -1, :]

        # Tanh ensures action mean is bounded between -1 and 1 for the leverage projection
        action_mean = torch.tanh(self.actor_mean(last_token)) 
        value = self.critic_head(last_token)

        return action_mean, self.actor_log_std, value

    def act(self, state_window: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean, log_std, _ = self.forward(state_window)
        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            action = mean + std * torch.randn_like(mean)
        return torch.clamp(action, -1.0, 1.0)