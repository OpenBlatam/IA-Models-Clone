from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List


def get_logger(
    name: str = "trainer",
    log_dir: str = "logs",
    level: int = logging.INFO,
    *,
    filename: str = "training.log",
) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, filename))
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


class TBWriter:
    def __init__(self, log_dir: str):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self.writer = SummaryWriter(log_dir=log_dir)
        except Exception:
            self.writer = None

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


@dataclass
class LogConfig:
    log_dir: str = "logs"
    log_name: str = "trainer"
    tb_subdir: str = "tb"
    level: int = logging.INFO


class TrainingLogger:
    def __init__(self, cfg: LogConfig):
        self.logger = get_logger(cfg.log_name, cfg.log_dir, cfg.level)
        self.tb = TBWriter(os.path.join(cfg.log_dir, cfg.tb_subdir))
        self.global_step = 0
        self.best_metric = float("-inf")

    def log_step(self, loss: float, lr: float, step_time_s: float) -> None:
        self.global_step += 1
        self.logger.info(f"step={self.global_step} loss={loss:.6f} lr={lr:.6e} dt={step_time_s:.3f}s")
        self.tb.add_scalar("train/loss", loss, self.global_step)
        self.tb.add_scalar("train/lr", lr, self.global_step)
        self.tb.add_scalar("train/step_time_s", step_time_s, self.global_step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
        epoch_time_s: Optional[float] = None,
    ) -> None:
        parts: List[str] = [f"epoch={epoch}"]
        for k, v in train_metrics.items():
            parts.append(f"train_{k}={v:.6f}")
            self.tb.add_scalar(f"train/{k}", v, epoch)
        if val_metrics:
            for k, v in val_metrics.items():
                parts.append(f"val_{k}={v:.6f}")
                self.tb.add_scalar(f"val/{k}", v, epoch)
            main_metric = val_metrics.get("acc") or val_metrics.get("f1") or val_metrics.get("loss", -float("inf"))
            if isinstance(main_metric, float) and main_metric > self.best_metric:
                self.best_metric = main_metric
                parts.append(f"best_metric={self.best_metric:.6f}")
        if lr is not None:
            parts.append(f"lr={lr:.6e}")
        if epoch_time_s is not None:
            parts.append(f"dt={epoch_time_s:.2f}s")
            self.tb.add_scalar("time/epoch_s", epoch_time_s, epoch)

        self.logger.info(" ".join(parts))

    def log_exception(self, where: str, exc: BaseException) -> None:
        self.logger.error(f"[{where}] {type(exc).__name__}: {exc}", exc_info=True)

    def close(self) -> None:
        self.tb.close()



