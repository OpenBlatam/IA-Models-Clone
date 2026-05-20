"""
AdamW Optimizer for TruthGPT API
================================

TensorFlow-like AdamW optimizer implementation.
Now integrated with optimization_core for enhanced performance.
"""

import torch.optim as optim
from typing import Optional

from .adam_base import AdamBaseOptimizer
from .optimizer_constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_BETA_1,
    DEFAULT_BETA_2,
    DEFAULT_EPSILON,
    DEFAULT_AMSGRAD,
    DEFAULT_WEIGHT_DECAY
)


class AdamW(AdamBaseOptimizer):
    """
    AdamW optimizer.
    
    Similar to tf.keras.optimizers.AdamW, this optimizer implements
    the AdamW algorithm for stochastic optimization with decoupled weight decay.
    
    Now uses optimization_core when available for better performance.
    """
    
    def __init__(self, 
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 beta_1: float = DEFAULT_BETA_1,
                 beta_2: float = DEFAULT_BETA_2,
                 epsilon: float = DEFAULT_EPSILON,
                 amsgrad: bool = DEFAULT_AMSGRAD,
                 weight_decay: float = DEFAULT_WEIGHT_DECAY,
                 name: Optional[str] = None,
                 use_optimization_core: bool = True):
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
            use_optimization_core: Whether to use optimization_core if available
        """
        super().__init__(
            optimizer_type='adamw',
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            use_optimization_core=use_optimization_core,
            weight_decay=weight_decay
        )
        
        # Store AdamW-specific parameter
        self.weight_decay = weight_decay
    
    def _create_pytorch_optimizer(self, parameters):
        """Create PyTorch AdamW optimizer as fallback."""
        return optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            eps=self.epsilon,
            amsgrad=self.amsgrad,
            weight_decay=self.weight_decay
        )
