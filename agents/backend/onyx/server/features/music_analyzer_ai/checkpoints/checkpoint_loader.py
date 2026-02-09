"""
Modular Checkpoint Loader
Handles loading of checkpoints with validation
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class CheckpointLoader:
    """Handles loading of checkpoints"""
    
    def load(
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
            checkpoint_path: Path to checkpoint file
            model: Optional model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            map_location: Device to load on
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
        
        # Load model state
        if model is not None and "model_state_dict" in checkpoint:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Model state loaded")
            except Exception as e:
                logger.error(f"Error loading model state: {str(e)}")
                raise
        
        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state loaded")
            except Exception as e:
                logger.warning(f"Error loading optimizer state: {str(e)}")
        
        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Scheduler state loaded")
            except Exception as e:
                logger.warning(f"Error loading scheduler state: {str(e)}")
        
        return checkpoint



