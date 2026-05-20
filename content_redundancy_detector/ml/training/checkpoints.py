"""
Checkpoint Management
Utilities for saving and loading model checkpoints
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with metadata
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path]):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            epoch: Current epoch
            metrics: Training metrics
            metadata: Additional metadata
            filename: Custom filename (optional)
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save metadata as JSON
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'timestamp': checkpoint['timestamp'],
                'metrics': metrics or {},
                'metadata': metadata or {},
            }, f, indent=2)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load on
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'metadata': checkpoint.get('metadata', {}),
            'timestamp': checkpoint.get('timestamp', ''),
        }
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get the latest checkpoint file
        
        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def get_best_checkpoint(self, metric: str = "val_loss", mode: str = "min") -> Optional[Path]:
        """
        Get the best checkpoint based on a metric
        
        Args:
            metric: Metric name to compare
            mode: 'min' or 'max'
            
        Returns:
            Path to best checkpoint or None
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        if not checkpoints:
            return None
        
        best_checkpoint = None
        best_value = float('inf') if mode == 'min' else float('-inf')
        
        for checkpoint_path in checkpoints:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                metrics = checkpoint.get('metrics', {})
                
                if metric in metrics:
                    value = metrics[metric]
                    if mode == 'min':
                        if value < best_value:
                            best_value = value
                            best_checkpoint = checkpoint_path
                    else:
                        if value > best_value:
                            best_value = value
                            best_checkpoint = checkpoint_path
            except Exception as e:
                logger.warning(f"Error reading checkpoint {checkpoint_path}: {e}")
                continue
        
        return best_checkpoint
    
    def list_checkpoints(self) -> list[Dict[str, Any]]:
        """
        List all checkpoints with metadata
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        for checkpoint_path in self.checkpoint_dir.glob("checkpoint_*.pth"):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                checkpoints.append({
                    'path': str(checkpoint_path),
                    'epoch': checkpoint.get('epoch', 0),
                    'timestamp': checkpoint.get('timestamp', ''),
                    'metrics': checkpoint.get('metrics', {}),
                })
            except Exception as e:
                logger.warning(f"Error reading checkpoint {checkpoint_path}: {e}")
        
        return sorted(checkpoints, key=lambda x: x['epoch'])



