"""
Base Callback for TruthGPT API
=============================

TensorFlow-like callback base class implementation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Callback(ABC):
    """
    Abstract base class for callbacks.
    
    Similar to tf.keras.callbacks.Callback, this class provides
    the interface for training callbacks.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize callback.
        
        Args:
            name: Optional name for the callback
        """
        self.name = name or self.__class__.__name__
        self.model = None
        self.params = {}
        self.validation_data = None
    
    def set_params(self, params: Dict[str, Any]):
        """Set parameters for the callback."""
        self.params = params
    
    def set_model(self, model: Any):
        """Set the model for the callback."""
        self.model = model
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of a batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of a batch."""
        pass
    
    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of a training batch."""
        pass
    
    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of a training batch."""
        pass
    
    def on_test_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of a test batch."""
        pass
    
    def on_test_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of a test batch."""
        pass
    
    def on_predict_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of a prediction batch."""
        pass
    
    def on_predict_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of a prediction batch."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"









