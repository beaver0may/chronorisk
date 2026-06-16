from __future__ import annotations

import os
import tempfile
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": seed,
        "epoch": epoch,
        "extra": extra or {},
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(handle)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)
