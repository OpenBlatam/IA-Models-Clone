"""
Modular Checkpoint Saver
Handles saving of checkpoints with metadata
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class CheckpointSaver:
    """Handles saving of checkpoints"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        checkpoint_name: str,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save checkpoint
        
        Args:
            checkpoint_name: Name for checkpoint
            model: Model to save
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            epoch: Current epoch
            metrics: Training metrics
            metadata: Additional metadata
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        if metadata is not None:
            checkpoint["metadata"] = metadata
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata as JSON for easy inspection
        metadata_path = self.checkpoint_dir / f"{checkpoint_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "epoch": epoch,
                "metrics": metrics or {},
                "metadata": metadata or {},
                "timestamp": checkpoint["timestamp"]
            }, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path



