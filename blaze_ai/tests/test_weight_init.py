from __future__ import annotations

import torch
from torch import nn

from blaze_ai.models.weight_init import init_transformer_like, freeze_modules, count_parameters


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e = nn.Embedding(10, 4)
        self.ln = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):  # type: ignore[override]
        return self.fc(self.ln(self.e(x)).mean(dim=1))


def test_weight_init_and_freeze() -> None:
    m = Tiny()
    init_transformer_like(m)
    # Check shapes intact
    assert m.e.weight.shape == (10, 4)
    freeze_modules([m.fc])
    assert all(not p.requires_grad for p in m.fc.parameters())
    stats = count_parameters(m)
    assert stats["total"] >= stats["trainable"]


