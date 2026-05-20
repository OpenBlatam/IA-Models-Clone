"""
Checkpoint Manager
==================

Utility for managing model checkpoints.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manager for model checkpoints.
    
    Features:
    - Save/load checkpoints
    - Checkpoint versioning
    - Best model tracking
    - Automatic cleanup
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self._logger = logger
        self.best_checkpoint: Optional[Path] = None
        self.best_metric: Optional[float] = None
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        prefix: str = "checkpoint"
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Metrics dictionary
            is_best: Whether this is the best model
            prefix: Checkpoint prefix
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"{prefix}_epoch_{epoch}_{timestamp}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "timestamp": timestamp
        }
        
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        self._logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint = best_path
            if metrics and "val_loss" in metrics:
                self.best_metric = metrics["val_loss"]
            self._logger.info(f"Best model saved: {best_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(prefix)
        
        return checkpoint_path
    
    def load(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[Path] = None,
        load_best: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            model: Model to load into
            checkpoint_path: Path to checkpoint (None = load best)
            load_best: Whether to load best model
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
            else:
                # Load latest
                checkpoints = self.list_checkpoints()
                if not checkpoints:
                    raise ValueError("No checkpoints found")
                checkpoint_path = checkpoints[-1]
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self._logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint
    
    def list_checkpoints(self, prefix: Optional[str] = None) -> List[Path]:
        """
        List available checkpoints.
        
        Args:
            prefix: Filter by prefix
        
        Returns:
            List of checkpoint paths
        """
        pattern = f"{prefix}*.pt" if prefix else "*.pt"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        return checkpoints
    
    def _cleanup_old_checkpoints(self, prefix: str):
        """Cleanup old checkpoints, keeping only the most recent ones."""
        checkpoints = self.list_checkpoints(prefix)
        
        if len(checkpoints) > self.max_checkpoints:
            # Keep best model and most recent ones
            to_remove = checkpoints[:-self.max_checkpoints]
            
            for checkpoint in to_remove:
                if checkpoint != self.best_checkpoint:
                    checkpoint.unlink()
                    self._logger.info(f"Removed old checkpoint: {checkpoint}")




