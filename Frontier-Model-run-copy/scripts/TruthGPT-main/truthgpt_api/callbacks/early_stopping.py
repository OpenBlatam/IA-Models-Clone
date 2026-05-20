"""
Early Stopping Callback for TruthGPT API
=======================================

TensorFlow-like early stopping callback implementation.
"""

from typing import Optional, Dict, Any
from .base import Callback


class EarlyStopping(Callback):
    """
    Early stopping callback.
    
    Similar to tf.keras.callbacks.EarlyStopping, this callback
    stops training when a monitored metric has stopped improving.
    """
    
    def __init__(self, 
                 monitor: str = 'val_loss',
                 min_delta: float = 0.0,
                 patience: int = 0,
                 mode: str = 'auto',
                 baseline: Optional[float] = None,
                 restore_best_weights: bool = False,
                 name: Optional[str] = None):
        """
        Initialize EarlyStopping callback.
        
        Args:
            monitor: Metric to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs to wait before stopping
            mode: One of 'auto', 'min', 'max'
            baseline: Baseline value for the monitored metric
            restore_best_weights: Whether to restore best weights
            name: Optional name for the callback
        """
        super().__init__(name)
        
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        # Internal state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best = None
        
        # Determine mode
        if mode == 'auto':
            if 'acc' in monitor:
                self.mode = 'max'
            else:
                self.mode = 'min'
        else:
            self.mode = mode
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best = None
        
        if self.baseline is not None:
            self.best = self.baseline
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of an epoch."""
        if logs is None:
            logs = {}
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.best is None:
            self.best = current
            if self.restore_best_weights and hasattr(self.model, 'state_dict'):
                self.best_weights = self.model.state_dict().copy()
        else:
            if self.mode == 'min':
                if current < self.best - self.min_delta:
                    self.best = current
                    self.wait = 0
                    if self.restore_best_weights and hasattr(self.model, 'state_dict'):
                        self.best_weights = self.model.state_dict().copy()
                else:
                    self.wait += 1
            else:  # mode == 'max'
                if current > self.best + self.min_delta:
                    self.best = current
                    self.wait = 0
                    if self.restore_best_weights and hasattr(self.model, 'state_dict'):
                        self.best_weights = self.model.state_dict().copy()
                else:
                    self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                self.model.load_state_dict(self.best_weights)
            return True  # Stop training
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        if self.stopped_epoch > 0:
            print(f"Early stopping at epoch {self.stopped_epoch}")
    
    def __repr__(self):
        return f"EarlyStopping(monitor={self.monitor}, patience={self.patience}, mode={self.mode})"


