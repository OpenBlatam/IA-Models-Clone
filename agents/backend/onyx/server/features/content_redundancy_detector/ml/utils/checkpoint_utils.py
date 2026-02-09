"""
Checkpoint Utilities
Advanced checkpoint management utilities
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class CheckpointUtils:
    """
    Advanced checkpoint utilities
    """
    
    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        filepath: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save checkpoint with metadata
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            filepath: Path to save checkpoint
            metadata: Additional metadata (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metadata': metadata or {},
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
    
    @staticmethod
    def load_checkpoint(
        filepath: Path,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            filepath: Path to checkpoint
            model: Model to load into (optional)
            optimizer: Optimizer to load into (optional)
            device: Device to load on (optional)
            
        Returns:
            Checkpoint dictionary
        """
        filepath = Path(filepath)
        if device is None:
            device = torch.device('cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model state")
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
        
        return checkpoint
    
    @staticmethod
    def find_best_checkpoint(
        checkpoint_dir: Path,
        metric: str = "loss",
        mode: str = "min",
    ) -> Optional[Path]:
        """
        Find best checkpoint based on metric
        
        Args:
            checkpoint_dir: Directory with checkpoints
            metric: Metric name
            mode: "min" or "max"
            
        Returns:
            Path to best checkpoint or None
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return None
        
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            return None
        
        best_checkpoint = None
        best_value = float('inf') if mode == "min" else float('-inf')
        
        for checkpoint_path in checkpoints:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'metadata' in checkpoint and metric in checkpoint['metadata']:
                    value = checkpoint['metadata'][metric]
                    if (mode == "min" and value < best_value) or (mode == "max" and value > best_value):
                        best_value = value
                        best_checkpoint = checkpoint_path
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        
        return best_checkpoint
    
    @staticmethod
    def list_checkpoints(checkpoint_dir: Path) -> List[Dict[str, Any]]:
        """
        List all checkpoints with metadata
        
        Args:
            checkpoint_dir: Directory with checkpoints
            
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for checkpoint_path in checkpoint_dir.glob("*.pth"):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                info = {
                    'path': checkpoint_path,
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'loss': checkpoint.get('loss', 'unknown'),
                    'metadata': checkpoint.get('metadata', {}),
                }
                checkpoints.append(info)
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_path}: {e}")
        
        return sorted(checkpoints, key=lambda x: x.get('epoch', 0), reverse=True)



