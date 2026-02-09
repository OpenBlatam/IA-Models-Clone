"""
Training Callbacks

Implements:
- Early stopping
- Model checkpointing
- Learning rate monitoring
- Training progress tracking
"""

import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for loss/metric
            restore_best_weights: Restore best weights on stop
            verbose: Whether to log messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score (loss or metric)
            model: Model to save weights from
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self._save_weights(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_weights(model)
            if self.verbose:
                logger.info(f"Early stopping: New best score: {score:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self._restore_weights(model)
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered after {self.patience} "
                        f"epochs without improvement"
                    )
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def _save_weights(self, model: nn.Module) -> None:
        """Save model weights."""
        self.best_weights = model.state_dict().copy()
    
    def _restore_weights(self, model: nn.Module) -> None:
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                logger.info("Restored best model weights")


class ModelCheckpoint:
    """
    Model checkpointing callback.
    """
    
    def __init__(
        self,
        save_dir: str,
        save_best: bool = True,
        save_every_n_epochs: Optional[int] = None,
        monitor: str = 'val_loss',
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize model checkpoint.
        
        Args:
            save_dir: Directory to save checkpoints
            save_best: Save best model based on monitor
            save_every_n_epochs: Save checkpoint every N epochs
            monitor: Metric to monitor
            mode: 'min' or 'max' for monitor metric
            verbose: Whether to log messages
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
    
    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """
        Save checkpoint.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optional optimizer to save
            metrics: Optional metrics dictionary
            **kwargs: Additional items to save
        """
        metrics = metrics or {}
        
        # Save best model
        if self.save_best and self.monitor in metrics:
            score = metrics[self.monitor]
            if self.best_score is None or self._is_better(score, self.best_score):
                self.best_score = score
                self._save_checkpoint(
                    epoch, model, optimizer, metrics,
                    filepath=self.save_dir / 'best_model.pt',
                    **kwargs
                )
                if self.verbose:
                    logger.info(
                        f"Saved best model ({self.monitor}={score:.4f}) "
                        f"at epoch {epoch}"
                    )
        
        # Save periodic checkpoint
        if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
            self._save_checkpoint(
                epoch, model, optimizer, metrics,
                filepath=self.save_dir / f'checkpoint_epoch_{epoch+1}.pt',
                **kwargs
            )
            if self.verbose:
                logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        metrics: Dict[str, float],
        filepath: Path,
        **kwargs
    ) -> None:
        """Save checkpoint to file."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            **kwargs
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)


class LearningRateMonitor:
    """Monitor learning rate during training."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.lr_history = []
    
    def __call__(self, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        """Record learning rate."""
        current_lr = optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        if self.verbose:
            logger.info(f"Epoch {epoch}: Learning rate = {current_lr:.2e}")



