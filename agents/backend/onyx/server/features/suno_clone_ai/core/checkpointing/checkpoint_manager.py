"""
Checkpoint Management

Handles saving and loading model checkpoints.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        filename: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Epoch number
            loss: Loss value
            metrics: Additional metrics
            filename: Checkpoint filename
            **kwargs: Additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics or {},
            **kwargs
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Generate filename
        if filename is None:
            if epoch is not None:
                filename = f"checkpoint_epoch_{epoch}.pt"
            else:
                filename = "checkpoint.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load state into
            checkpoint_path: Path to checkpoint
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load on
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from: {checkpoint_path}")
        
        # Load optimizer
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
        
        # Load scheduler
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Loaded scheduler state")
        
        return checkpoint
    
    def list_checkpoints(self) -> list:
        """List all checkpoints."""
        return list(self.checkpoint_dir.glob("*.pt"))


def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: str = "./checkpoints",
    **kwargs
) -> str:
    """
    Convenience function to save checkpoint.
    
    Args:
        model: Model to save
        checkpoint_dir: Checkpoint directory
        **kwargs: Additional checkpoint data
        
    Returns:
        Path to saved checkpoint
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.save_checkpoint(model, **kwargs)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to load checkpoint.
    
    Args:
        model: Model to load
        checkpoint_path: Path to checkpoint
        **kwargs: Additional arguments
        
    Returns:
        Checkpoint dictionary
    """
    manager = CheckpointManager()
    return manager.load_checkpoint(model, checkpoint_path, **kwargs)



