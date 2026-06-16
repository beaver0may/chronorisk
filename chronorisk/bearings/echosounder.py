from __future__ import annotations

import torch
from torch import nn

from chronorisk.config import ModelConfig


class _DilatedBranch(nn.Module):
    def __init__(self, channels: int, dim: int, kernel: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * 2
        self.conv = nn.Conv1d(channels, dim, kernel_size=kernel, dilation=2, padding=self.pad)
        self.activation = nn.GELU()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        out = self.conv(signal)
        length = signal.shape[-1]
        out = out[..., :length]
        return self.activation(out)


class BiosensorEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            _DilatedBranch(config.bio_channels, config.dim, kernel)
            for kernel in config.conv_kernels
        )
        self.combine = nn.Linear(config.dim * len(config.conv_kernels), config.dim)
        self.missing = nn.Parameter(torch.zeros(config.dim))
        self.decay = nn.Parameter(torch.zeros(1))

    def forward(self, bio: torch.Tensor, bio_mask: torch.Tensor) -> torch.Tensor:
        batch, horizon, channels, length = bio.shape
        flat = bio.reshape(batch * horizon, channels, length)
        pooled = [branch(flat).mean(dim=-1) for branch in self.branches]
        merged = self.combine(torch.cat(pooled, dim=-1))
        observed = merged.reshape(batch, horizon, -1)
        gate = torch.sigmoid(self.decay)
        mask = bio_mask.unsqueeze(-1)
        imputed = gate * self.missing.view(1, 1, -1)
        return mask * observed + (1.0 - mask) * imputed
