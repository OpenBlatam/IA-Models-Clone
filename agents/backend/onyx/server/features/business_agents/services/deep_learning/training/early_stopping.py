"""
Early Stopping - Training Callback
==================================

Early stopping callback to prevent overfitting.
"""

from typing import Optional
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    
    Monitors a metric and stops training if no improvement is observed.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights
            verbose: Whether to log early stopping events
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            model: Model to save weights from
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                import torch
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                import torch
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience}. "
                    f"Best score: {self.best_score:.4f}"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                if self.verbose:
                    logger.info(f"Early stopping triggered. Best score: {self.best_score:.4f}")
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """
        Check if current score is better than best.
        
        Args:
            current: Current score
            best: Best score so far
        
        Returns:
            True if current is better
        """
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        logger.debug("Early stopping reset")



