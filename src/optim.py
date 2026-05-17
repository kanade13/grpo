from __future__ import annotations

from collections.abc import Iterable

import torch


def build_adam_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Adam:
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
