"""
Adagrad Optimizer for TruthGPT API
==================================

TensorFlow-like Adagrad optimizer implementation.
Now integrated with optimization_core for enhanced performance.
"""

import torch.optim as optim
from typing import Optional, Dict, Any

from .base_optimizer import BaseOptimizer
from .optimizer_constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPSILON
)


class Adagrad(BaseOptimizer):
    """
    Adagrad optimizer.
    
    Similar to tf.keras.optimizers.Adagrad, this optimizer implements
    the Adagrad algorithm for stochastic optimization.
    
    Now uses optimization_core when available for better performance.
    """
    
    def __init__(self, 
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 initial_accumulator_value: float = 0.1,
                 epsilon: float = DEFAULT_EPSILON,
                 name: Optional[str] = None,
                 use_optimization_core: bool = True):
        """
        Initialize Adagrad optimizer.
        
        Args:
            learning_rate: Learning rate
            initial_accumulator_value: Initial accumulator value
            epsilon: Small constant for numerical stability
            name: Optional name for the optimizer
            use_optimization_core: Whether to use optimization_core if available
        """
        super().__init__(
            optimizer_type='adagrad',
            learning_rate=learning_rate,
            name=name,
            use_optimization_core=use_optimization_core,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon
        )
    
    def _create_pytorch_optimizer(self, parameters):
        """Create PyTorch Adagrad optimizer as fallback."""
        return optim.Adagrad(
            parameters,
            lr=self.learning_rate,
            eps=self.epsilon,
            initial_accumulator_value=self.initial_accumulator_value
        )
    
    def _get_optimizer_specific_config(self) -> Dict[str, Any]:
        """Get Adagrad-specific configuration parameters."""
        return {
            'initial_accumulator_value': self.initial_accumulator_value,
            'epsilon': self.epsilon
        }
    
    def _format_parameters(self) -> str:
        """Format Adagrad parameters for string representation."""
        return f"learning_rate={self.learning_rate}, initial_accumulator_value={self.initial_accumulator_value}"
