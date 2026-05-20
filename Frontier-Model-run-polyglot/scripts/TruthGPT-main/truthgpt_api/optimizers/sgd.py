"""
SGD Optimizer for TruthGPT API
==============================

TensorFlow-like SGD optimizer implementation.
Now integrated with optimization_core for enhanced performance.
"""

import torch.optim as optim
from typing import Optional, Dict, Any

from .base_optimizer import BaseOptimizer
from .optimizer_constants import (
    DEFAULT_SGD_LEARNING_RATE,
    DEFAULT_MOMENTUM,
    DEFAULT_NESTEROV
)


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Similar to tf.keras.optimizers.SGD, this optimizer implements
    the SGD algorithm for stochastic optimization.
    
    Now uses optimization_core when available for better performance.
    """
    
    def __init__(self, 
                 learning_rate: float = DEFAULT_SGD_LEARNING_RATE,
                 momentum: float = DEFAULT_MOMENTUM,
                 nesterov: bool = DEFAULT_NESTEROV,
                 name: Optional[str] = None,
                 use_optimization_core: bool = True):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            nesterov: Whether to use Nesterov momentum
            name: Optional name for the optimizer
            use_optimization_core: Whether to use optimization_core if available
        """
        super().__init__(
            optimizer_type='sgd',
            learning_rate=learning_rate,
            name=name,
            use_optimization_core=use_optimization_core,
            momentum=momentum,
            nesterov=nesterov
        )
        
        # Store SGD-specific parameters for direct access
        self.momentum = momentum
        self.nesterov = nesterov
    
    def _create_pytorch_optimizer(self, parameters):
        """Create PyTorch SGD optimizer as fallback."""
        return optim.SGD(
            parameters,
            lr=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov
        )
    
    def _get_optimizer_specific_config(self) -> Dict[str, Any]:
        """Get SGD-specific configuration parameters."""
        return {
            'momentum': self.momentum,
            'nesterov': self.nesterov
        }
    
    def _format_parameters(self) -> str:
        """Format SGD parameters for string representation."""
        return f"learning_rate={self.learning_rate}, momentum={self.momentum}, nesterov={self.nesterov}"
