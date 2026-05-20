"""
Adam Base Optimizer - Base class for Adam and AdamW optimizers.
==============================================================

Shared functionality for Adam and AdamW optimizers to eliminate duplication.
"""

import torch.optim as optim
from typing import Optional, Dict, Any

from .base_optimizer import BaseOptimizer
from .optimizer_constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_BETA_1,
    DEFAULT_BETA_2,
    DEFAULT_EPSILON,
    DEFAULT_AMSGRAD
)


class AdamBaseOptimizer(BaseOptimizer):
    """
    Base class for Adam and AdamW optimizers.
    
    Provides shared functionality for:
    - Parameter storage (beta_1, beta_2, epsilon, amsgrad)
    - PyTorch optimizer creation
    - Configuration generation
    - Parameter formatting
    
    Single Responsibility: Eliminate duplication between Adam and AdamW.
    """
    
    def __init__(
        self,
        optimizer_type: str,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        beta_1: float = DEFAULT_BETA_1,
        beta_2: float = DEFAULT_BETA_2,
        epsilon: float = DEFAULT_EPSILON,
        amsgrad: bool = DEFAULT_AMSGRAD,
        name: Optional[str] = None,
        use_optimization_core: bool = True,
        **kwargs
    ):
        """
        Initialize Adam-based optimizer.
        
        Args:
            optimizer_type: Type of optimizer ('adam' or 'adamw')
            learning_rate: Learning rate
            beta_1: Exponential decay rate for first moment estimates
            beta_2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            amsgrad: Whether to use AMSGrad variant
            name: Optional name for the optimizer
            use_optimization_core: Whether to use optimization_core if available
            **kwargs: Additional optimizer-specific parameters (e.g., weight_decay for AdamW)
        """
        super().__init__(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            name=name,
            use_optimization_core=use_optimization_core,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            **kwargs
        )
        # Parameters are now accessed via self.kwargs
    
    def _get_optimizer_specific_config(self) -> Dict[str, Any]:
        """
        Get Adam-based optimizer-specific configuration parameters.
        
        Returns:
            Dictionary with optimizer-specific parameters
        """
        return self.kwargs
    
    def _format_parameters(self) -> str:
        """
        Format Adam-based optimizer parameters for string representation.
        
        Returns:
            Formatted parameter string
        """
        params = [f"{k}={v}" for k, v in self.kwargs.items()]
        return f"learning_rate={self.learning_rate}, {', '.join(params)}"

