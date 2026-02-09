"""
RMSprop Optimizer for TruthGPT API
==================================

TensorFlow-like RMSprop optimizer implementation.
Now integrated with optimization_core for enhanced performance.
"""

import torch.optim as optim
from typing import Optional, Dict, Any

from .base_optimizer import BaseOptimizer
from .optimizer_constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_RHO,
    DEFAULT_MOMENTUM,
    DEFAULT_EPSILON,
    DEFAULT_CENTERED
)


class RMSprop(BaseOptimizer):
    """
    RMSprop optimizer.
    
    Similar to tf.keras.optimizers.RMSprop, this optimizer implements
    the RMSprop algorithm for stochastic optimization.
    
    Now uses optimization_core when available for better performance.
    """
    
    def __init__(self, 
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 rho: float = DEFAULT_RHO,
                 momentum: float = DEFAULT_MOMENTUM,
                 epsilon: float = DEFAULT_EPSILON,
                 centered: bool = DEFAULT_CENTERED,
                 name: Optional[str] = None,
                 use_optimization_core: bool = True):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            rho: Decay rate for moving average
            momentum: Momentum factor
            epsilon: Small constant for numerical stability
            centered: Whether to center the gradients
            name: Optional name for the optimizer
            use_optimization_core: Whether to use optimization_core if available
        """
        super().__init__(
            optimizer_type='rmsprop',
            learning_rate=learning_rate,
            name=name,
            use_optimization_core=use_optimization_core,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered
        )
    
    def _create_pytorch_optimizer(self, parameters):
        """Create PyTorch RMSprop optimizer as fallback."""
        return optim.RMSprop(
            parameters,
            lr=self.learning_rate,
            alpha=self.rho,
            eps=self.epsilon,
            momentum=self.momentum,
            centered=self.centered
        )
    
    def _get_optimizer_specific_config(self) -> Dict[str, Any]:
        """Get RMSprop-specific configuration parameters."""
        return {
            'rho': self.rho,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'centered': self.centered
        }
    
    def _format_parameters(self) -> str:
        """Format RMSprop parameters for string representation."""
        return f"learning_rate={self.learning_rate}, rho={self.rho}, momentum={self.momentum}"
