from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    name: Literal["adamw", "adam", "sgd"] = "adamw",
    *,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    n = (name or "adamw").lower()
    if n == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if n == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: Literal["cosine", "plateau", "onecycle", "linear_warmup"] = "plateau",
    *,
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
    factor: float = 0.5,
    patience: int = 1,
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
    n = (name or "plateau").lower()
    if n == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
    if n == "cosine":
        assert total_steps is not None and total_steps > 0
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if n == "onecycle":
        assert total_steps is not None and total_steps > 0
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max(g["lr"] for g in optimizer.param_groups), total_steps=total_steps)
    if n == "linear_warmup":
        assert total_steps is not None and total_steps > 0
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)


