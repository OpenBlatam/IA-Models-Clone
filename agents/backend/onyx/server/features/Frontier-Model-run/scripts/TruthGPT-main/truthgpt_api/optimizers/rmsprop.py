"""
RMSprop Optimizer for TruthGPT API
==================================

TensorFlow-like RMSprop optimizer implementation.
"""

import torch
import torch.optim as optim
from typing import Optional, Dict, Any


class RMSprop:
    """
    RMSprop optimizer.
    
    Similar to tf.keras.optimizers.RMSprop, this optimizer implements
    the RMSprop algorithm for stochastic optimization.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 rho: float = 0.9,
                 momentum: float = 0.0,
                 epsilon: float = 1e-7,
                 centered: bool = False,
                 name: Optional[str] = None):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            rho: Decay rate for moving average
            momentum: Momentum factor
            epsilon: Small constant for numerical stability
            centered: Whether to center the gradients
            name: Optional name for the optimizer
        """
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered
        self.name = name or "RMSprop"
        
        self._optimizer = None
        self._parameters = None
    
    def _create_optimizer(self, parameters):
        """Create PyTorch optimizer."""
        self._parameters = parameters
        self._optimizer = optim.RMSprop(
            parameters,
            lr=self.learning_rate,
            alpha=self.rho,
            eps=self.epsilon,
            momentum=self.momentum,
            centered=self.centered
        )
        return self._optimizer
    
    def __call__(self, parameters):
        """Create optimizer for given parameters."""
        return self._create_optimizer(parameters)
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'rho': self.rho,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'centered': self.centered
        }
    
    def __repr__(self):
        return f"RMSprop(learning_rate={self.learning_rate}, rho={self.rho}, momentum={self.momentum})"









