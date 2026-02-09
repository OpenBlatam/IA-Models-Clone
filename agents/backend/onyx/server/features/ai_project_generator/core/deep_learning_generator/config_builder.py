"""
Configuration Builder Module

Fluent API for building generator configurations.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ConfigBuilder:
    """
    Fluent builder for generator configurations.
    
    Example:
        config = (ConfigBuilder()
                 .with_framework("pytorch")
                 .with_model_type("transformer")
                 .with_gpu(True)
                 .with_mixed_precision(True)
                 .build())
    """
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
    
    def with_framework(self, framework: str) -> "ConfigBuilder":
        """Set the framework."""
        self._config["framework"] = framework
        return self
    
    def with_model_type(self, model_type: str) -> "ConfigBuilder":
        """Set the model type."""
        self._config["model_type"] = model_type
        return self
    
    def with_gpu(self, use_gpu: bool = True) -> "ConfigBuilder":
        """Enable/disable GPU usage."""
        self._config["use_gpu"] = use_gpu
        return self
    
    def with_mixed_precision(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable mixed precision training."""
        self._config["mixed_precision"] = enabled
        return self
    
    def with_batch_size(self, batch_size: int) -> "ConfigBuilder":
        """Set batch size."""
        self._config["batch_size"] = batch_size
        return self
    
    def with_learning_rate(self, learning_rate: float) -> "ConfigBuilder":
        """Set learning rate."""
        self._config["learning_rate"] = learning_rate
        return self
    
    def with_epochs(self, num_epochs: int) -> "ConfigBuilder":
        """Set number of epochs."""
        self._config["num_epochs"] = num_epochs
        return self
    
    def with_early_stopping(self, enabled: bool = True, patience: int = 5) -> "ConfigBuilder":
        """Enable/disable early stopping."""
        self._config["early_stopping"] = enabled
        if enabled:
            self._config["early_stopping_patience"] = patience
        return self
    
    def with_gradient_clipping(self, enabled: bool = True, max_norm: float = 1.0) -> "ConfigBuilder":
        """Enable/disable gradient clipping."""
        self._config["gradient_clipping"] = enabled
        if enabled:
            self._config["gradient_clipping_max_norm"] = max_norm
        return self
    
    def with_checkpointing(self, enabled: bool = True, save_best: bool = True) -> "ConfigBuilder":
        """Enable/disable model checkpointing."""
        self._config["checkpointing"] = enabled
        if enabled:
            self._config["checkpoint_save_best"] = save_best
        return self
    
    def with_experiment_tracking(self, enabled: bool = True, backend: str = "wandb") -> "ConfigBuilder":
        """Enable/disable experiment tracking."""
        self._config["experiment_tracking"] = enabled
        if enabled:
            self._config["experiment_tracking_backend"] = backend
        return self
    
    def with_custom(self, key: str, value: Any) -> "ConfigBuilder":
        """Add custom configuration."""
        self._config[key] = value
        return self
    
    def with_dict(self, config_dict: Dict[str, Any]) -> "ConfigBuilder":
        """Merge a dictionary into the configuration."""
        self._config.update(config_dict)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return the configuration."""
        return self._config.copy()
    
    def reset(self) -> "ConfigBuilder":
        """Reset the builder to start fresh."""
        self._config = {}
        return self


def create_config_builder() -> ConfigBuilder:
    """Create a new configuration builder."""
    return ConfigBuilder()















