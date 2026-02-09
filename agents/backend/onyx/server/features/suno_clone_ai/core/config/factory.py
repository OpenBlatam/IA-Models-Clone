"""
Configuration Factory

Provides:
- Factory functions for creating configured objects
- Configuration-based initialization
- Default configurations
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from ...config_loader import get_config_loader

logger = logging.getLogger(__name__)


class ConfigFactory:
    """Factory for creating objects from configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config factory.
        
        Args:
            config_path: Path to config file
        """
        self.config = get_config_loader(config_path)
    
    def create_model(self, **overrides) -> Any:
        """
        Create model from configuration.
        
        Args:
            **overrides: Override config values
            
        Returns:
            Model instance
        """
        from ...models import create_enhanced_music_model
        
        model_config = self.config.get_model_config()
        model_config.update(overrides)
        
        return create_enhanced_music_model(**model_config)
    
    def create_optimizer(self, model: Any, **overrides) -> Any:
        """
        Create optimizer from configuration.
        
        Args:
            model: Model to create optimizer for
            **overrides: Override config values
            
        Returns:
            Optimizer instance
        """
        from ...training import create_optimizer
        
        training_config = self.config.get_training_config()
        optimizer_config = {
            'model': model,
            'optimizer_type': training_config.get('optimizer', 'adamw'),
            'learning_rate': training_config.get('learning_rate', 1e-4),
            'weight_decay': training_config.get('weight_decay', 0.01),
            **training_config.get('optimizer_params', {}),
            **overrides
        }
        
        return create_optimizer(**optimizer_config)
    
    def create_scheduler(self, optimizer: Any, **overrides) -> Any:
        """
        Create scheduler from configuration.
        
        Args:
            optimizer: Optimizer to schedule
            **overrides: Override config values
            
        Returns:
            Scheduler instance
        """
        from ...training import create_scheduler
        
        training_config = self.config.get_training_config()
        scheduler_config = {
            'optimizer': optimizer,
            'scheduler_type': training_config.get('scheduler', 'cosine'),
            **training_config.get('scheduler_params', {}),
            **overrides
        }
        
        return create_scheduler(**scheduler_config)
    
    def create_loss(self, **overrides) -> Any:
        """
        Create loss function from configuration.
        
        Args:
            **overrides: Override config values
            
        Returns:
            Loss function
        """
        from ...training import create_loss_function
        
        training_config = self.config.get_training_config()
        loss_type = training_config.get('loss', 'mse')
        
        return create_loss_function(loss_type, **overrides)
    
    def create_generator(self, **overrides) -> Any:
        """
        Create generator from configuration.
        
        Args:
            **overrides: Override config values
            
        Returns:
            Generator instance
        """
        from ...generators import TransformersMusicGenerator
        
        inference_config = self.config.get_inference_config()
        model_selection = self.config.get('model_selection', {})
        
        generator_config = {
            'model_name': model_selection.get('base_model', 'facebook/musicgen-medium'),
            'use_mixed_precision': inference_config.get('mixed_precision', True),
            'use_compile': inference_config.get('compile', {}).get('enabled', True),
            'compile_mode': inference_config.get('compile', {}).get('mode', 'reduce-overhead'),
            **overrides
        }
        
        return TransformersMusicGenerator(**generator_config)


def create_from_config(
    object_type: str,
    config_path: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create object from configuration.
    
    Args:
        object_type: Type of object ('model', 'optimizer', 'scheduler', etc.)
        config_path: Path to config file
        **kwargs: Additional arguments
        
    Returns:
        Created object
    """
    factory = ConfigFactory(config_path)
    
    create_map = {
        'model': factory.create_model,
        'optimizer': factory.create_optimizer,
        'scheduler': factory.create_scheduler,
        'loss': factory.create_loss,
        'generator': factory.create_generator
    }
    
    if object_type not in create_map:
        raise ValueError(
            f"Unknown object type: {object_type}. "
            f"Available: {list(create_map.keys())}"
        )
    
    return create_map[object_type](**kwargs)

