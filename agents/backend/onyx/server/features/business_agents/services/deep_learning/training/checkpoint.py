"""
Checkpoint Manager - Model Checkpointing
========================================

Utilities for saving and loading model checkpoints.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manager for saving and loading model checkpoints.
    
    Handles:
    - Model state dict
    - Optimizer state dict
    - Training metrics
    - Epoch information
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save (optional)
            epoch: Current epoch
            metrics: Training metrics (optional)
            is_best: Whether this is the best model
            filename: Custom filename (optional)
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        checkpoint['timestamp'] = datetime.utcnow().isoformat()
        
        # Determine filename
        if filename is None:
            if is_best:
                filename = "best_model.pt"
            else:
                filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # Also save as latest
        if not is_best:
            latest_path = self.checkpoint_dir / "latest.pt"
            torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: Optional[str] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            filename: Checkpoint filename (defaults to 'latest.pt')
            map_location: Device to map checkpoint to
        
        Returns:
            Checkpoint dictionary
        """
        if filename is None:
            filename = "latest.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Model loaded from {checkpoint_path}")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✅ Optimizer state loaded")
        
        return checkpoint
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint filenames
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return best_path if best_path.exists() else None
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        latest_path = self.checkpoint_dir / "latest.pt"
        return latest_path if latest_path.exists() else None



