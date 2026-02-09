from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 10, hidden: int = 32, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    model.train()
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


def _demo() -> None:  # pragma: no cover
    torch.manual_seed(0)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))
    print(train_step(model, opt, x, y))


if __name__ == "__main__":  # pragma: no cover
    _demo()


