"""
Checkpoint Callback Module

Implements checkpoint saving callback.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import TrainingCallback

logger = logging.getLogger(__name__)


class CheckpointCallback(TrainingCallback):
    """
    Checkpoint saving callback.
    
    Args:
        checkpoint_dir: Directory to save checkpoints.
        save_best: If True, save best model based on monitor.
        save_every: Save checkpoint every N epochs (None to disable).
        monitor: Metric to monitor for best model.
        mode: "min" or "max".
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_best: bool = True,
        save_every: Optional[int] = None,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_every = save_every
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        logger.debug(f"Initialized CheckpointCallback with checkpoint_dir='{checkpoint_dir}'")
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Not used."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Save checkpoint if needed."""
        if not TORCH_AVAILABLE:
            return
        
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("scheduler")
        
        if model is None:
            return
        
        # Save best model
        if self.save_best and self.monitor in metrics:
            current_score = metrics[self.monitor]
            is_better = (
                current_score < self.best_score if self.mode == "min"
                else current_score > self.best_score
            )
            
            if is_better:
                self.best_score = current_score
                self._save_checkpoint(
                    model, optimizer, scheduler, epoch, metrics, "best"
                )
        
        # Save periodic checkpoints
        if self.save_every and epoch % self.save_every == 0:
            self._save_checkpoint(
                model, optimizer, scheduler, epoch, metrics, f"epoch_{epoch}"
            )
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any], **kwargs):
        """Not used."""
        pass
    
    def _save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, Any],
        suffix: str
    ):
        """Save checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")



