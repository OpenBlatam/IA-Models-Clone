"""
Base Training Callback Module

Base class for training callbacks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """
    Base class for training callbacks.
    
    Provides hooks for training lifecycle events.
    """
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, **kwargs):
        """Called at the start of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any], **kwargs):
        """Called at the end of each batch."""
        pass
    
    def on_training_start(self, **kwargs):
        """Called at the start of training."""
        pass
    
    def on_training_end(self, **kwargs):
        """Called at the end of training."""
        pass



