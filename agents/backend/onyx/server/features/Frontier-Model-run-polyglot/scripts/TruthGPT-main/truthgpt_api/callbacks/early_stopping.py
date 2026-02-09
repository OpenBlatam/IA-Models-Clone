"""
Early Stopping Callback for TruthGPT API
=======================================

TensorFlow-like early stopping callback implementation.

Refactored to:
- Eliminate code duplication in min/max comparison logic
- Consolidate best_weights saving logic
- Improve parameter validation
- Better error handling
"""

from typing import Optional, Dict, Any, Callable
from .base import Callback


class EarlyStopping(Callback):
    """
    Early stopping callback.
    
    Similar to tf.keras.callbacks.EarlyStopping, this callback
    stops training when a monitored metric has stopped improving.
    
    Responsibilities:
    - Monitor a metric during training
    - Track best value and patience counter
    - Save/restore best model weights if requested
    - Signal when to stop training
    """
    
    # Valid modes for comparison
    VALID_MODES = {'auto', 'min', 'max'}
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 0,
        mode: str = 'auto',
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize EarlyStopping callback.
        
        Args:
            monitor: Metric to monitor (e.g., 'val_loss', 'val_accuracy')
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs to wait before stopping
            mode: One of 'auto', 'min', 'max'
            baseline: Baseline value for the monitored metric
            restore_best_weights: Whether to restore best weights on stop
            name: Optional name for the callback
        
        Raises:
            ValueError: If mode is invalid or patience is negative
        """
        super().__init__(name)
        
        # Validate parameters
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got '{mode}'"
            )
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        # Determine comparison mode
        self.mode = self._determine_mode(mode, monitor)
        
        # Internal state (initialized in on_train_begin)
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best = None
    
    @staticmethod
    def _determine_mode(mode: str, monitor: str) -> str:
        """
        Determine the comparison mode.
        
        Args:
            mode: User-specified mode ('auto', 'min', 'max')
            monitor: Metric name to monitor
        
        Returns:
            'min' or 'max' based on mode and monitor name
        """
        if mode != 'auto':
            return mode
        
        # Auto-detect mode based on metric name
        # Accuracy metrics should be maximized, loss metrics minimized
        if 'acc' in monitor.lower():
            return 'max'
        return 'min'
    
    def _is_improvement(self, current: float, best: float) -> bool:
        """
        Check if current value is an improvement over best value.
        
        Args:
            current: Current metric value
            best: Best metric value so far
        
        Returns:
            True if current is an improvement, False otherwise
        """
        if self.mode == 'min':
            return current < best - self.min_delta
        else:  # mode == 'max'
            return current > best + self.min_delta
    
    def _save_best_weights(self) -> None:
        """
        Save current model weights as best weights.
        
        Only saves if restore_best_weights is enabled and model
        has state_dict method.
        """
        if not self.restore_best_weights:
            return
        
        if self.model is None:
            return
        
        if not hasattr(self.model, 'state_dict'):
            return
        
        try:
            self.best_weights = self.model.state_dict().copy()
        except Exception:
            # Silently fail if state_dict() is not available or fails
            pass
    
    def _restore_best_weights(self) -> None:
        """
        Restore best model weights.
        
        Only restores if restore_best_weights is enabled and
        best_weights are available.
        """
        if not self.restore_best_weights:
            return
        
        if self.best_weights is None:
            return
        
        if self.model is None:
            return
        
        if not hasattr(self.model, 'load_state_dict'):
            return
        
        try:
            self.model.load_state_dict(self.best_weights)
        except Exception:
            # Silently fail if load_state_dict fails
            pass
    
    def _reset_state(self) -> None:
        """Reset internal state for new training session."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best = self.baseline if self.baseline is not None else None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        self._reset_state()
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> Optional[bool]:
        """
        Called at the end of an epoch.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics for current epoch
        
        Returns:
            True if training should stop, None otherwise
        """
        if logs is None:
            logs = {}
        
        current = logs.get(self.monitor)
        if current is None:
            # Metric not available, skip this epoch
            return None
        
        # Initialize best value if this is the first epoch
        if self.best is None:
            self.best = current
            self._save_best_weights()
            return None
        
        # Check if current value is an improvement
        if self._is_improvement(current, self.best):
            self.best = current
            self.wait = 0
            self._save_best_weights()
        else:
            self.wait += 1
        
        # Check if we should stop training
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self._restore_best_weights()
            return True  # Signal to stop training
        
        return None
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        if self.stopped_epoch > 0:
            print(f"Early stopping at epoch {self.stopped_epoch}")
    
    def get_best_value(self) -> Optional[float]:
        """
        Get the best metric value observed.
        
        Returns:
            Best metric value or None if no value recorded yet
        """
        return self.best
    
    def get_stopped_epoch(self) -> int:
        """
        Get the epoch at which training was stopped.
        
        Returns:
            Epoch number (0 if training wasn't stopped)
        """
        return self.stopped_epoch
    
    def __repr__(self) -> str:
        """String representation of the callback."""
        return (
            f"EarlyStopping(monitor='{self.monitor}', "
            f"patience={self.patience}, mode='{self.mode}', "
            f"min_delta={self.min_delta})"
        )
