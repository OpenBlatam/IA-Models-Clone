from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from blaze_ai.utils.experiment import ExperimentTracker
from blaze_ai.utils.checkpointing import save_checkpoint, load_checkpoint


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = nn.Linear(2, 2)

    def forward(self, x):  # type: ignore[override]
        return self.l(x)


def test_experiment_tracker_writes_csv_and_config(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "run1")
    tracker.log_config({"hello": "world"})
    tracker.log_metrics({"loss": 1.23, "acc": 0.9}, step=10)
    assert (tmp_path / "run1" / "config.json").exists()
    assert (tmp_path / "run1" / "metrics.csv").exists()


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    p = tmp_path / "ckpt.pt"
    path = save_checkpoint(p, model, opt, None, scaler, epoch=2, global_step=123, extra={"note": "ok"})
    assert Path(path).exists()
    # load
    state = load_checkpoint(path, model, opt, None, scaler)
    assert "extra" in state and state["extra"]["note"] == "ok"


