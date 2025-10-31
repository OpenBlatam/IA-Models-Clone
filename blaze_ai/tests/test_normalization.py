from __future__ import annotations

import torch
from torch import nn

from blaze_ai.utils.normalization import (
    apply_weight_norm,
    remove_weight_norm,
    apply_spectral_norm,
    remove_spectral_norm,
)


def test_weight_and_spectral_norm_roundtrip() -> None:
    m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    m = apply_weight_norm(m)
    m = apply_spectral_norm(m)
    x = torch.randn(2, 8)
    y = m(x)
    assert y.shape == (2, 4)
    m = remove_spectral_norm(m)
    m = remove_weight_norm(m)
    y2 = m(x)
    assert y2.shape == (2, 4)


