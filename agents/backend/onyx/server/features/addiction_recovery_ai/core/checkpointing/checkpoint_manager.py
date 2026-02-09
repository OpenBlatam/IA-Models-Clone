"""
Checkpoint Manager
Model checkpointing and versioning
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import logging
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints with versioning
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_best: bool = True,
        save_latest: bool = True
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Save best checkpoint
            save_latest: Save latest checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.save_latest = save_latest
        
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.checkpoint_history: List[Dict[str, Any]] = []
    
    def save(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save checkpoint
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Training metrics
            optimizer: Optimizer state
            scheduler: Scheduler state
            is_best: Whether this is the best checkpoint
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / "checkpoint_best.pth"
            self.best_metric = metrics.get("loss", float('inf'))
            self.best_epoch = epoch
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest
        if self.save_latest:
            latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
            shutil.copy(checkpoint_path, latest_path)
        
        # Update history
        self.checkpoint_history.append({
            "epoch": epoch,
            "path": str(checkpoint_path),
            "metrics": metrics,
            "is_best": is_best
        })
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        load_latest: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            model: Model to load into
            checkpoint_path: Path to checkpoint
            load_best: Load best checkpoint
            load_latest: Load latest checkpoint
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            
        Returns:
            Checkpoint dictionary
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "checkpoint_best.pth"
        elif load_latest:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pth"
        elif checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
        else:
            raise ValueError("Must specify checkpoint_path, load_best, or load_latest")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints"""
        # Get all epoch checkpoints
        epoch_checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda x: int(x.stem.split("_")[-1])
        )
        
        # Keep only the most recent ones
        if len(epoch_checkpoints) > self.max_checkpoints:
            for old_checkpoint in epoch_checkpoints[:-self.max_checkpoints]:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint"""
        best_path = self.checkpoint_dir / "checkpoint_best.pth"
        return str(best_path) if best_path.exists() else None
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        return str(latest_path) if latest_path.exists() else None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pth"):
            try:
                checkpoint = torch.load(checkpoint_file, map_location="cpu")
                checkpoints.append({
                    "path": str(checkpoint_file),
                    "epoch": checkpoint.get("epoch", 0),
                    "metrics": checkpoint.get("metrics", {}),
                    "timestamp": checkpoint.get("timestamp", "")
                })
            except Exception as e:
                logger.warning(f"Failed to load checkpoint info from {checkpoint_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x["epoch"], reverse=True)


def create_checkpoint_manager(
    checkpoint_dir: str = "checkpoints",
    **kwargs
) -> CheckpointManager:
    """Factory for checkpoint manager"""
    return CheckpointManager(checkpoint_dir, **kwargs)













