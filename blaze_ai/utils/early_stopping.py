from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingConfig:
    patience: int = 3
    min_delta: float = 0.0
    mode: str = "min"  # one of ["min", "max"]


class EarlyStopping:
    def __init__(self, config: EarlyStoppingConfig | None = None) -> None:
        cfg = config or EarlyStoppingConfig()
        self.patience = int(cfg.patience)
        self.min_delta = float(cfg.min_delta)
        self.mode = (cfg.mode or "min").lower()
        self.best_score: float | None = None
        self.num_bad_epochs = 0

    def step(self, metric_value: float) -> bool:
        if self.best_score is None:
            self.best_score = metric_value
            self.num_bad_epochs = 0
            return False

        improved = False
        if self.mode == "min":
            improved = metric_value < (self.best_score - self.min_delta)
        else:
            improved = metric_value > (self.best_score + self.min_delta)

        if improved:
            self.best_score = metric_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs > self.patience


