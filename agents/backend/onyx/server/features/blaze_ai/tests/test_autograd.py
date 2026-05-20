from __future__ import annotations

import torch
from torch import nn

from blaze_ai.utils.autograd import compute_gradient_norm


def test_compute_gradient_norm() -> None:
    model = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    y = torch.randint(0, 2, (3,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    gnorm = compute_gradient_norm(model.parameters())
    assert gnorm > 0.0


