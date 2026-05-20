from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def get_model_state(model: torch.nn.Module) -> Dict[str, Any]:
    if hasattr(model, "module"):
        return model.module.state_dict()  # type: ignore[attr-defined]
    return model.state_dict()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    global_step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    path = str(path)
    ckpt = {
        "model": get_model_state(model),
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()  # type: ignore[attr-defined]
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if extra:
        ckpt["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model"])
    if optimizer is not None and "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])  # type: ignore[arg-type]
    if scheduler is not None and "scheduler" in data:
        scheduler.load_state_dict(data["scheduler"])  # type: ignore[arg-type]
    if scaler is not None and "scaler" in data:
        scaler.load_state_dict(data["scaler"])  # type: ignore[arg-type]
    return {k: v for k, v in data.items() if k not in {"model", "optimizer", "scheduler", "scaler"}}


