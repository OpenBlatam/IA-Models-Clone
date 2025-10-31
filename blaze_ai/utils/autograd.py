from __future__ import annotations

from typing import Iterable

import torch


def compute_gradient_norm(parameters: Iterable[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    device = params[0].grad.device  # type: ignore[union-attr]
    if norm_type == float("inf"):
        norms = [p.grad.detach().abs().max() for p in params]  # type: ignore[union-attr]
        total = torch.stack(norms).max()
    else:
        total = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]), norm_type)
    return float(total.item())


