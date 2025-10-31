"""
Adagrad Optimizer for TruthGPT API
==================================

TensorFlow-like Adagrad optimizer implementation.
"""

import torch
import torch.optim as optim
from typing import Optional, Dict, Any


class Adagrad:
    """
    Adagrad optimizer.
    
    Similar to tf.keras.optimizers.Adagrad, this optimizer implements
    the Adagrad algorithm for stochastic optimization.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 initial_accumulator_value: float = 0.1,
                 epsilon: float = 1e-7,
                 name: Optional[str] = None):
        """
        Initialize Adagrad optimizer.
        
        Args:
            learning_rate: Learning rate
            initial_accumulator_value: Initial accumulator value
            epsilon: Small constant for numerical stability
            name: Optional name for the optimizer
        """
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon
        self.name = name or "Adagrad"
        
        self._optimizer = None
        self._parameters = None
    
    def _create_optimizer(self, parameters):
        """Create PyTorch optimizer."""
        self._parameters = parameters
        self._optimizer = optim.Adagrad(
            parameters,
            lr=self.learning_rate,
            eps=self.epsilon,
            initial_accumulator_value=self.initial_accumulator_value
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
            'initial_accumulator_value': self.initial_accumulator_value,
            'epsilon': self.epsilon
        }
    
    def __repr__(self):
        return f"Adagrad(learning_rate={self.learning_rate}, initial_accumulator_value={self.initial_accumulator_value})"









