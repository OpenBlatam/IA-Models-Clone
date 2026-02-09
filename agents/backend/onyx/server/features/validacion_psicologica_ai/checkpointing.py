"""
Model Checkpointing
===================
Save and load model checkpoints with metadata
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from pathlib import Path
import json
import structlog
from datetime import datetime

logger = structlog.get_logger()


class CheckpointManager:
    """
    Manager for saving and loading model checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_best: bool = True,
        max_checkpoints: int = 5
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory for checkpoints
            save_best: Save best model based on validation loss
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.max_checkpoints = max_checkpoints
        self.best_val_loss = float('inf')
        self.checkpoint_history = []
        
        logger.info("CheckpointManager initialized", checkpoint_dir=str(self.checkpoint_dir))
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss (optional)
            metrics: Additional metrics (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": metrics or {},
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save checkpoint
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pt"
        if val_loss is not None:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}_val_{val_loss:.4f}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if self.save_best and val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logger.info("Best model saved", val_loss=val_loss, epoch=epoch)
        
        # Save metadata JSON
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "metrics": metrics,
                "metadata": metadata,
                "timestamp": checkpoint["timestamp"]
            }, f, indent=2)
        
        # Track checkpoint
        self.checkpoint_history.append({
            "path": str(checkpoint_path),
            "epoch": epoch,
            "val_loss": val_loss
        })
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info("Checkpoint saved", path=str(checkpoint_path), epoch=epoch)
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            device: Device to load on
            
        Returns:
            Checkpoint metadata
        """
        if device is None:
            from .deep_learning_models import get_device
            device = get_device()
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            
            # Load optimizer state
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            logger.info(
                "Checkpoint loaded",
                path=checkpoint_path,
                epoch=checkpoint.get("epoch", 0)
            )
            
            return {
                "epoch": checkpoint.get("epoch", 0),
                "train_loss": checkpoint.get("train_loss"),
                "val_loss": checkpoint.get("val_loss"),
                "metrics": checkpoint.get("metrics", {}),
                "metadata": checkpoint.get("metadata", {})
            }
            
        except Exception as e:
            logger.error("Error loading checkpoint", path=checkpoint_path, error=str(e))
            raise
    
    def load_best_model(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load best model checkpoint
        
        Args:
            model: Model to load
            optimizer: Optimizer to load (optional)
            
        Returns:
            Checkpoint metadata
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            raise FileNotFoundError(f"Best model not found at {best_path}")
        
        return self.load_checkpoint(str(best_path), model, optimizer)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to latest checkpoint
        
        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Keep best model and latest checkpoints
        best_path = self.checkpoint_dir / "best_model.pt"
        checkpoints = [c for c in checkpoints if c != best_path]
        
        if len(checkpoints) > self.max_checkpoints:
            for old_checkpoint in checkpoints[self.max_checkpoints:]:
                old_checkpoint.unlink()
                # Also remove metadata JSON
                old_checkpoint.with_suffix('.json').unlink(missing_ok=True)
                logger.debug("Removed old checkpoint", path=str(old_checkpoint))


# Global checkpoint manager
checkpoint_manager = CheckpointManager()




