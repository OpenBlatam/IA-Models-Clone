"""
Weight Initializer
==================

Applies initialization strategies to model layers.
"""

import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Callable

from .initialization_strategies import InitializationStrategies

logger = logging.getLogger(__name__)


class WeightInitializer:
    """Applies weight initialization strategies to models."""
    
    STRATEGY_MAP = {
        "xavier": InitializationStrategies.xavier_uniform,
        "xavier_uniform": InitializationStrategies.xavier_uniform,
        "xavier_normal": InitializationStrategies.xavier_normal,
        "he": InitializationStrategies.kaiming_uniform,
        "kaiming": InitializationStrategies.kaiming_normal,
        "kaiming_uniform": InitializationStrategies.kaiming_uniform,
        "kaiming_normal": InitializationStrategies.kaiming_normal,
        "orthogonal": InitializationStrategies.orthogonal,
        "zeros": InitializationStrategies.zeros,
        "ones": InitializationStrategies.ones,
        "normal": InitializationStrategies.normal,
        "uniform": InitializationStrategies.uniform,
    }
    
    @staticmethod
    def initialize_weights(
        module: nn.Module,
        strategy: str = "xavier",
        **kwargs
    ) -> None:
        """
        Initialize model weights using specified strategy.
        
        Args:
            module: PyTorch module to initialize
            strategy: Initialization strategy name
            **kwargs: Strategy-specific parameters
        """
        if strategy not in WeightInitializer.STRATEGY_MAP:
            logger.warning(f"Unknown strategy '{strategy}', using 'xavier'")
            strategy = "xavier"
        
        init_func = WeightInitializer.STRATEGY_MAP[strategy]
        
        for module_item in module.modules():
            if isinstance(module_item, nn.Linear):
                init_func(module_item, **kwargs)
            elif isinstance(module_item, nn.Conv2d):
                init_func(module_item, **kwargs)
            elif isinstance(module_item, nn.LayerNorm):
                nn.init.ones_(module_item.weight)
                nn.init.zeros_(module_item.bias)
            elif isinstance(module_item, nn.BatchNorm2d):
                nn.init.ones_(module_item.weight)
                nn.init.zeros_(module_item.bias)
                if module_item.running_mean is not None:
                    nn.init.zeros_(module_item.running_mean)
                if module_item.running_var is not None:
                    nn.init.ones_(module_item.running_var)
    
    @staticmethod
    def initialize_with_custom(
        module: nn.Module,
        init_func: Callable[[nn.Module], None]
    ) -> None:
        """
        Initialize with custom function.
        
        Args:
            module: PyTorch module
            init_func: Custom initialization function
        """
        for module_item in module.modules():
            if isinstance(module_item, (nn.Linear, nn.Conv2d)):
                init_func(module_item)


