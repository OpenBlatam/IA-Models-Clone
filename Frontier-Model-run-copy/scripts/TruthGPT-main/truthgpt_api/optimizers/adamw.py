"""
AdamW Optimizer for TruthGPT API
================================

TensorFlow-like AdamW optimizer implementation.
"""

import torch
import torch.optim as optim
from typing import Optional, Dict, Any


class AdamW:
    """
    AdamW optimizer.
    
    Similar to tf.keras.optimizers.AdamW, this optimizer implements
    the AdamW algorithm for stochastic optimization with decoupled weight decay.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-7,
                 amsgrad: bool = False,
                 weight_decay: float = 0.01,
                 name: Optional[str] = None):
        """
        Initialize AdamW optimizer.
        
        Args:
            learning_rate: Learning rate
            beta_1: Exponential decay rate for first moment estimates
            beta_2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            amsgrad: Whether to use AMSGrad variant
            weight_decay: Weight decay factor
            name: Optional name for the optimizer
        """
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.weight_decay = weight_decay
        self.name = name or "AdamW"
        
        self._optimizer = None
        self._parameters = None
    
    def _create_optimizer(self, parameters):
        """Create PyTorch optimizer."""
        self._parameters = parameters
        self._optimizer = optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            eps=self.epsilon,
            amsgrad=self.amsgrad,
            weight_decay=self.weight_decay
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
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'weight_decay': self.weight_decay
        }
    
    def __repr__(self):
        return f"AdamW(learning_rate={self.learning_rate}, beta_1={self.beta_1}, beta_2={self.beta_2}, weight_decay={self.weight_decay})"


