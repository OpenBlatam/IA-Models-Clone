"""
SGD Optimizer for TruthGPT API
==============================

TensorFlow-like SGD optimizer implementation.
"""

import torch
import torch.optim as optim
from typing import Optional, Dict, Any


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Similar to tf.keras.optimizers.SGD, this optimizer implements
    the SGD algorithm for stochastic optimization.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 name: Optional[str] = None):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            nesterov: Whether to use Nesterov momentum
            name: Optional name for the optimizer
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.name = name or "SGD"
        
        self._optimizer = None
        self._parameters = None
    
    def _create_optimizer(self, parameters):
        """Create PyTorch optimizer."""
        self._parameters = parameters
        self._optimizer = optim.SGD(
            parameters,
            lr=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov
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
            'momentum': self.momentum,
            'nesterov': self.nesterov
        }
    
    def __repr__(self):
        return f"SGD(learning_rate={self.learning_rate}, momentum={self.momentum}, nesterov={self.nesterov})"









