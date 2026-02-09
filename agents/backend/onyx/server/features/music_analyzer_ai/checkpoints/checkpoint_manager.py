"""
Modular Checkpoint Manager
Manages model and training checkpoints
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from .checkpoint_loader import CheckpointLoader
from .checkpoint_saver import CheckpointSaver
from .checkpoint_validator import CheckpointValidator


class CheckpointManager:
    """
    Centralized checkpoint management
    Handles saving, loading, and validation of checkpoints
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = CheckpointLoader()
        self.saver = CheckpointSaver(self.checkpoint_dir)
        self.validator = CheckpointValidator()
    
    def save_checkpoint(
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
        return self.saver.save(
            checkpoint_name=checkpoint_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            metadata=metadata
        )
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Optional model to load into
            optimizer: Optional optimizer to load into
            scheduler: Optional scheduler to load into
            map_location: Device to load on
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = self.loader.load(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location
        )
        
        # Validate checkpoint
        if self.validator.validate(checkpoint):
            logger.info(f"Checkpoint validated: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint validation failed: {checkpoint_path}")
        
        return checkpoint
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints"""
        checkpoints = []
        for path in self.checkpoint_dir.glob("*.pt"):
            checkpoints.append(path.stem)
        return sorted(checkpoints)
    
    def get_best_checkpoint(self, metric: str = "val_loss", mode: str = "min") -> Optional[str]:
        """
        Get best checkpoint based on metric
        
        Args:
            metric: Metric name to compare
            mode: "min" or "max"
        
        Returns:
            Path to best checkpoint or None
        """
        best_checkpoint = None
        best_value = float('inf') if mode == "min" else float('-inf')
        
        for checkpoint_path in self.checkpoint_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if "metrics" in checkpoint and metric in checkpoint["metrics"]:
                    value = checkpoint["metrics"][metric]
                    is_better = (
                        value < best_value if mode == "min"
                        else value > best_value
                    )
                    if is_better:
                        best_value = value
                        best_checkpoint = str(checkpoint_path)
            except Exception as e:
                logger.warning(f"Error reading checkpoint {checkpoint_path}: {str(e)}")
                continue
        
        return best_checkpoint



