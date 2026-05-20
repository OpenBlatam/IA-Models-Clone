"""
Checkpoint Utilities
====================

Advanced checkpoint management utilities:
- Smart checkpointing
- Checkpoint comparison
- Checkpoint merging
- Model versioning
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import hashlib
import json

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Advanced checkpoint manager with versioning and comparison.
    """
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return {}
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        checkpoint_name: Optional[str] = None
    ) -> Path:
        """
        Save checkpoint with metadata.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            scheduler: Scheduler (optional)
            epoch: Epoch number
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
            checkpoint_name: Custom checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
            if is_best:
                checkpoint_name = "best_model.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {},
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Calculate hash
        with open(checkpoint_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Update metadata
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics or {},
            'is_best': is_best,
            'hash': file_hash,
            'size_mb': checkpoint_path.stat().st_size / 1024**2
        }
        
        self.metadata[checkpoint_name] = checkpoint_info
        self._save_metadata()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_name: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint
            model: Model to load into
            optimizer: Optimizer to load into (optional)
            scheduler: Scheduler to load into (optional)
            strict: Strict loading
            
        Returns:
            Checkpoint information
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'path': checkpoint_path
        }
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        return list(self.metadata.values())
    
    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get best checkpoint."""
        for info in self.metadata.values():
            if info.get('is_best', False):
                return info
        return None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """
        Cleanup old checkpoints, keeping only the last N.
        
        Args:
            keep_last_n: Number of checkpoints to keep
        """
        checkpoints = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('epoch', 0),
            reverse=True
        )
        
        for name, info in checkpoints[keep_last_n:]:
            checkpoint_path = Path(info['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Deleted old checkpoint: {checkpoint_path}")
            del self.metadata[name]
        
        self._save_metadata()



