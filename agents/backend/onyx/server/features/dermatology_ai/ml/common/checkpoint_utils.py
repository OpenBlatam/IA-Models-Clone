"""
Checkpoint Utilities
Enhanced checkpoint management
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Enhanced checkpoint manager"""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        keep_best: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.checkpoints: List[Dict[str, Any]] = []
        self.best_checkpoint: Optional[Path] = None
        self.best_metric: Optional[float] = None
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            scheduler: Scheduler (optional)
            epoch: Current epoch
            metrics: Metrics dictionary
            is_best: Whether this is the best checkpoint
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        # Get underlying model if wrapped
        model_to_save = model
        if hasattr(model, 'module'):
            model_to_save = model.module
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Determine checkpoint name
        if is_best:
            checkpoint_name = "best.pt"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Track checkpoint
        checkpoint_info = {
            'path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics or {},
            'is_best': is_best,
            'timestamp': checkpoint['timestamp']
        }
        self.checkpoints.append(checkpoint_info)
        
        # Update best checkpoint
        if is_best or (metrics and self._is_better(metrics)):
            if self.best_checkpoint and self.best_checkpoint.exists():
                # Keep old best as backup
                backup_path = self.best_checkpoint.parent / f"best_backup_{self.best_checkpoint.stem}.pt"
                shutil.copy(self.best_checkpoint, backup_path)
            
            self.best_checkpoint = checkpoint_path
            if metrics:
                self.best_metric = self._extract_metric(metrics)
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return checkpoint_path
    
    def load(
        self,
        path: Optional[Path] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            path: Path to checkpoint (if None, loads best or latest)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            load_best: Whether to load best checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        if path is None:
            if load_best and self.best_checkpoint:
                path = self.best_checkpoint
            else:
                # Load latest checkpoint
                checkpoints = sorted(
                    self.checkpoint_dir.glob("checkpoint_*.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if not checkpoints:
                    raise FileNotFoundError("No checkpoints found")
                path = checkpoints[0]
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location='cpu')
        logger.info(f"Checkpoint loaded: {path}")
        
        # Load model state
        if model:
            model_to_load = model
            if hasattr(model, 'module'):
                model_to_load = model.module
            
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state loaded")
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded")
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state loaded")
        
        return checkpoint
    
    def _is_better(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics are better than current best"""
        if self.best_metric is None:
            return True
        
        # Use val_loss as primary metric
        val_loss = metrics.get('val_loss')
        if val_loss is None:
            return False
        
        return val_loss < self.best_metric
    
    def _extract_metric(self, metrics: Dict[str, float]) -> float:
        """Extract primary metric for comparison"""
        return metrics.get('val_loss', float('inf'))
    
    def _cleanup(self):
        """Cleanup old checkpoints"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by epoch (oldest first)
        sorted_checkpoints = sorted(
            [c for c in self.checkpoints if not c['is_best']],
            key=lambda x: x['epoch']
        )
        
        # Remove oldest checkpoints
        num_to_remove = len(sorted_checkpoints) - (self.max_checkpoints - (1 if self.keep_best else 0))
        for checkpoint_info in sorted_checkpoints[:num_to_remove]:
            checkpoint_path = checkpoint_info['path']
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
        
        # Update checkpoints list
        self.checkpoints = [
            c for c in self.checkpoints
            if c['is_best'] or c['epoch'] > sorted_checkpoints[num_to_remove - 1]['epoch']
        ]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints"""
        return self.checkpoints.copy()
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        return self.best_checkpoint


def save_checkpoint(
    checkpoint_dir: Union[str, Path],
    model: torch.nn.Module,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    metrics: Optional[Dict[str, float]] = None,
    is_best: bool = False
) -> Path:
    """Save checkpoint (convenience function)"""
    manager = CheckpointManager(checkpoint_dir)
    return manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics=metrics,
        is_best=is_best
    )


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, Any]:
    """Load checkpoint (convenience function)"""
    manager = CheckpointManager(Path(checkpoint_path).parent)
    return manager.load(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler
    )

