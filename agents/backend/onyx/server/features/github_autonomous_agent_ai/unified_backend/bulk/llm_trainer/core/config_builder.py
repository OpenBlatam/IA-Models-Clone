"""
Configuration Builder Module
============================

Builder pattern for creating training configurations.
Provides fluent API for building complex configurations.

Author: BUL System
Date: 2024
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigBuilder:
    """
    Builder for creating training configurations.
    
    Provides a fluent API to build complex training configurations step by step.
    
    Example:
        >>> config = (ConfigBuilder()
        ...     .with_model("gpt2")
        ...     .with_dataset("data.json")
        ...     .with_learning_rate(3e-5)
        ...     .with_batch_size(8)
        ...     .build())
    """
    
    def __init__(self):
        """Initialize ConfigBuilder."""
        self.config: Dict[str, Any] = {}
    
    def with_model(self, model_name: str, model_type: str = "causal") -> 'ConfigBuilder':
        """
        Set model configuration.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (causal/seq2seq)
            
        Returns:
            self for method chaining
        """
        self.config["model_name"] = model_name
        self.config["model_type"] = model_type
        return self
    
    def with_dataset(self, dataset_path: Union[str, Path]) -> 'ConfigBuilder':
        """
        Set dataset path.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            self for method chaining
        """
        self.config["dataset_path"] = dataset_path
        return self
    
    def with_output_dir(self, output_dir: Union[str, Path]) -> 'ConfigBuilder':
        """
        Set output directory.
        
        Args:
            output_dir: Output directory
            
        Returns:
            self for method chaining
        """
        self.config["output_dir"] = output_dir
        return self
    
    def with_learning_rate(self, learning_rate: float) -> 'ConfigBuilder':
        """
        Set learning rate.
        
        Args:
            learning_rate: Learning rate
            
        Returns:
            self for method chaining
        """
        self.config["learning_rate"] = learning_rate
        return self
    
    def with_epochs(self, num_epochs: int) -> 'ConfigBuilder':
        """
        Set number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            self for method chaining
        """
        self.config["num_train_epochs"] = num_epochs
        return self
    
    def with_batch_size(self, batch_size: int) -> 'ConfigBuilder':
        """
        Set batch size.
        
        Args:
            batch_size: Batch size
            
        Returns:
            self for method chaining
        """
        self.config["batch_size"] = batch_size
        return self
    
    def with_early_stopping(self, patience: int = 3) -> 'ConfigBuilder':
        """
        Enable early stopping.
        
        Args:
            patience: Patience for early stopping
            
        Returns:
            self for method chaining
        """
        self.config["early_stopping_patience"] = patience
        self.config["evaluation_strategy"] = "steps"
        self.config["eval_steps"] = 100
        self.config["load_best_model_at_end"] = True
        return self
    
    def with_gradient_checkpointing(self, enabled: bool = True) -> 'ConfigBuilder':
        """
        Enable gradient checkpointing.
        
        Args:
            enabled: Whether to enable gradient checkpointing
            
        Returns:
            self for method chaining
        """
        self.config["gradient_checkpointing"] = enabled
        return self
    
    def with_mixed_precision(self, fp16: bool = False, bf16: bool = False) -> 'ConfigBuilder':
        """
        Configure mixed precision training.
        
        Args:
            fp16: Enable FP16
            bf16: Enable BF16
            
        Returns:
            self for method chaining
        """
        self.config["fp16"] = fp16
        self.config["bf16"] = bf16
        return self
    
    def with_optimizer(self, optimizer: str = "adamw_torch") -> 'ConfigBuilder':
        """
        Set optimizer.
        
        Args:
            optimizer: Optimizer type
            
        Returns:
            self for method chaining
        """
        self.config["optimizer"] = optimizer
        return self
    
    def with_custom(self, key: str, value: Any) -> 'ConfigBuilder':
        """
        Add custom configuration.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            self for method chaining
        """
        self.config[key] = value
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build the configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        # Set defaults if not specified
        defaults = {
            "output_dir": "./checkpoints",
            "learning_rate": 3e-5,
            "num_train_epochs": 3,
            "batch_size": 8,
            "model_type": "causal",
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
        
        return self.config.copy()

