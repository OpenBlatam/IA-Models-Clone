"""
Model Utils Submodule
Aggregates model utility components.
"""

from .parameters import count_parameters, get_model_size_mb
from .checkpoint import save_model_checkpoint, load_model_checkpoint
from .summary import initialize_weights, get_model_summary


class ModelUtils:
    """
    Utility functions for model management.
    Combines all model utility functions.
    """
    
    @staticmethod
    def count_parameters(model, trainable_only: bool = False):
        """Count model parameters."""
        return count_parameters(model, trainable_only)
    
    @staticmethod
    def get_model_size_mb(model):
        """Get model size in MB."""
        return get_model_size_mb(model)
    
    @staticmethod
    def save_model_checkpoint(model, optimizer, epoch, loss, filepath, metadata=None):
        """Save model checkpoint with metadata."""
        return save_model_checkpoint(model, optimizer, epoch, loss, filepath, metadata)
    
    @staticmethod
    def load_model_checkpoint(model, filepath, optimizer=None, device="cpu"):
        """Load model checkpoint."""
        return load_model_checkpoint(model, filepath, optimizer, device)
    
    @staticmethod
    def initialize_weights(model, method: str = "xavier_uniform"):
        """Initialize model weights."""
        return initialize_weights(model, method)
    
    @staticmethod
    def get_model_summary(model):
        """Get model summary."""
        return get_model_summary(model)


__all__ = [
    "ModelUtils",
    "count_parameters",
    "get_model_size_mb",
    "save_model_checkpoint",
    "load_model_checkpoint",
    "initialize_weights",
    "get_model_summary",
]



