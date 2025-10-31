from __future__ import annotations

import math
import time
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .training_logging import TrainingLogger, LogConfig


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total, correct, vloss = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        vloss += loss_fn(logits, yb).item()
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    vloss /= max(len(loader), 1)
    vacc = correct / max(total, 1)
    return {"loss": vloss, "acc": vacc}


def train_with_logging(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    epochs: int = 5,
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 1.0,
    use_amp: bool = True,
    log_dir: str = "logs",
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stop_patience: Optional[int] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    tlog = TrainingLogger(LogConfig(log_dir=log_dir, log_name="trainer"))

    best_metric = -float("inf")
    bad_epochs = 0

    try:
        for epoch in range(epochs):
            model.train()
            epoch_t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            for it, (xb, yb) in enumerate(train_loader):
                step_t0 = time.perf_counter()
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    logits = model(xb)
                    loss = loss_fn(logits, yb) / max(1, grad_accum_steps)

                # skip non-finite batches safely
                if not torch.isfinite(loss.detach()):
                    optimizer.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.update()
                    continue

                scaler.scale(loss).backward()

                if ((it + 1) % grad_accum_steps) == 0:
                    scaler.unscale_(optimizer)
                    # clamp non-finite grads
                    for p in model.parameters():
                        if p.grad is not None and p.grad.is_floating_point() and not torch.isfinite(p.grad).all():
                            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    # clip
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    if not math.isfinite(float(grad_norm)):
                        optimizer.zero_grad(set_to_none=True)
                        if scaler.is_enabled():
                            scaler.update()
                        dt = time.perf_counter() - step_t0
                        lr = optimizer.param_groups[0]["lr"]
                        tlog.log_step(loss.item() * max(1, grad_accum_steps), lr, dt)
                        continue

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

                dt = time.perf_counter() - step_t0
                lr = optimizer.param_groups[0]["lr"]
                tlog.log_step(loss.item() * max(1, grad_accum_steps), lr, dt)

            val_metrics = _evaluate(model, val_loader, loss_fn, device) if val_loader is not None else None
            tlog.log_epoch(
                epoch=epoch,
                train_metrics={"loss": loss.item() * max(1, grad_accum_steps)},
                val_metrics=val_metrics,
                lr=optimizer.param_groups[0]["lr"],
                epoch_time_s=time.perf_counter() - epoch_t0,
            )

            # early stopping on val acc (or inverse val loss if no acc)
            if early_stop_patience is not None and val_metrics is not None:
                metric = val_metrics.get("acc")
                if metric is None:
                    metric = -val_metrics.get("loss", 0.0)
                if metric > best_metric:
                    best_metric = metric
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stop_patience:
                        break
    except Exception as e:
        tlog.log_exception("train_with_logging", e)
        raise
    finally:
        tlog.close()

    return {"best_val_metric": max(best_metric, tlog.best_metric)}


