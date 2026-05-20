from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss(
    task: str = "classification",
    *,
    label_smoothing: float = 0.0,
    focal_gamma: Optional[float] = None,
) -> nn.Module:
    t = (task or "classification").lower()
    if t in {"classification", "multiclass"}:
        if focal_gamma is not None:
            return FocalLoss(gamma=float(focal_gamma))
        # PyTorch CrossEntropyLoss supports label_smoothing >= 0
        return nn.CrossEntropyLoss(label_smoothing=float(max(0.0, label_smoothing)))
    if t in {"multilabel"}:
        return nn.BCEWithLogitsLoss()
    if t in {"regression"}:
        return nn.MSELoss()
    # default
    return nn.CrossEntropyLoss(label_smoothing=float(max(0.0, label_smoothing)))


