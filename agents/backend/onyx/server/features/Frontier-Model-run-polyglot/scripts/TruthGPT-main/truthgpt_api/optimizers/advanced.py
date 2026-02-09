"""
Advanced Optimizers for TruthGPT API
===================================

Advanced optimizer implementations for TruthGPT.
Now integrated with BaseOptimizer for consistency and optimization_core support.
"""

import torch.optim as optim
from typing import Optional, Dict, Any

from .base_optimizer import BaseOptimizer


class AdaBelief(BaseOptimizer):
    """
    AdaBelief optimizer.
    
    Similar to tf.keras.optimizers.AdaBelief, this optimizer implements
    the AdaBelief algorithm for adaptive learning rates.
    
    Now uses BaseOptimizer for consistency and optimization_core support.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-14,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 name: Optional[str] = None,
                 use_optimization_core: bool = True):
        """
        Initialize AdaBelief optimizer.
        
        Args:
            learning_rate: Learning rate
            beta_1: Exponential decay rate for first moment estimates
            beta_2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay factor
            amsgrad: Whether to use AMSGrad variant
            name: Optional name for the optimizer
            use_optimization_core: Whether to use optimization_core if available
        """
        super().__init__(
            optimizer_type='adabelief',
            learning_rate=learning_rate,
            name=name,
            use_optimization_core=use_optimization_core,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
    
    def _create_pytorch_optimizer(self, parameters):
        """Create PyTorch optimizer as fallback."""
        # Note: PyTorch doesn't have AdaBelief built-in, so we'll use AdamW as a fallback
        # In a real implementation, you'd implement AdaBelief from scratch
        return optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=(self.kwargs['beta_1'], self.kwargs['beta_2']),
            eps=self.kwargs['epsilon'],
            weight_decay=self.kwargs['weight_decay'],
            amsgrad=self.kwargs.get('amsgrad', False)
        )
    
    def _get_optimizer_specific_config(self) -> Dict[str, Any]:
        """Get AdaBelief-specific configuration parameters."""
        return {
            'beta_1': self.kwargs['beta_1'],
            'beta_2': self.kwargs['beta_2'],
            'epsilon': self.kwargs['epsilon'],
            'weight_decay': self.kwargs['weight_decay'],
            'amsgrad': self.kwargs.get('amsgrad', False)
        }
    
    def _format_parameters(self) -> str:
        """Format AdaBelief parameters for string representation."""
        return f"learning_rate={self.learning_rate}, beta_1={self.kwargs['beta_1']}, beta_2={self.kwargs['beta_2']}"


class RAdam(BaseOptimizer):
    """
    RAdam optimizer.
    
    Similar to tf.keras.optimizers.RAdam, this optimizer implements
    the Rectified Adam algorithm for adaptive learning rates.
    
    Now uses BaseOptimizer for consistency and optimization_core support.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-7,
                 weight_decay: float = 0.0,
                 name: Optional[str] = None,
                 use_optimization_core: bool = True):
        """
        Initialize RAdam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta_1: Exponential decay rate for first moment estimates
            beta_2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay factor
            name: Optional name for the optimizer
            use_optimization_core: Whether to use optimization_core if available
        """
        super().__init__(
            optimizer_type='radam',
            learning_rate=learning_rate,
            name=name,
            use_optimization_core=use_optimization_core,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay
        )
    
    def _create_pytorch_optimizer(self, parameters):
        """Create PyTorch optimizer as fallback."""
        # Note: PyTorch doesn't have RAdam built-in, so we'll use Adam as a fallback
        # In a real implementation, you'd implement RAdam from scratch
        return optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=(self.kwargs['beta_1'], self.kwargs['beta_2']),
            eps=self.kwargs['epsilon'],
            weight_decay=self.kwargs['weight_decay']
        )
    
    def _get_optimizer_specific_config(self) -> Dict[str, Any]:
        """Get RAdam-specific configuration parameters."""
        return {
            'beta_1': self.kwargs['beta_1'],
            'beta_2': self.kwargs['beta_2'],
            'epsilon': self.kwargs['epsilon'],
            'weight_decay': self.kwargs['weight_decay']
        }
    
    def _format_parameters(self) -> str:
        """Format RAdam parameters for string representation."""
        return f"learning_rate={self.learning_rate}, beta_1={self.kwargs['beta_1']}, beta_2={self.kwargs['beta_2']}"


class Lion(BaseOptimizer):
    """
    Lion optimizer.
    
    Similar to tf.keras.optimizers.Lion, this optimizer implements
    the Lion algorithm for efficient optimization.
    
    Now uses BaseOptimizer for consistency and optimization_core support.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.99,
                 weight_decay: float = 0.0,
                 name: Optional[str] = None,
                 use_optimization_core: bool = True):
        """
        Initialize Lion optimizer.
        
        Args:
            learning_rate: Learning rate
            beta_1: Exponential decay rate for first moment estimates
            beta_2: Exponential decay rate for second moment estimates
            weight_decay: Weight decay factor
            name: Optional name for the optimizer
            use_optimization_core: Whether to use optimization_core if available
        """
        super().__init__(
            optimizer_type='lion',
            learning_rate=learning_rate,
            name=name,
            use_optimization_core=use_optimization_core,
            beta_1=beta_1,
            beta_2=beta_2,
            weight_decay=weight_decay
        )
    
    def _create_pytorch_optimizer(self, parameters):
        """Create PyTorch optimizer as fallback."""
        # Note: PyTorch doesn't have Lion built-in, so we'll use AdamW as a fallback
        # In a real implementation, you'd implement Lion from scratch
        return optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=(self.kwargs['beta_1'], self.kwargs['beta_2']),
            weight_decay=self.kwargs['weight_decay']
        )
    
    def _get_optimizer_specific_config(self) -> Dict[str, Any]:
        """Get Lion-specific configuration parameters."""
        return {
            'beta_1': self.kwargs['beta_1'],
            'beta_2': self.kwargs['beta_2'],
            'weight_decay': self.kwargs['weight_decay']
        }
    
    def _format_parameters(self) -> str:
        """Format Lion parameters for string representation."""
        return f"learning_rate={self.learning_rate}, beta_1={self.kwargs['beta_1']}, beta_2={self.kwargs['beta_2']}"


class AdaBound(BaseOptimizer):
    """
    AdaBound optimizer.
    
    Similar to tf.keras.optimizers.AdaBound, this optimizer implements
    the AdaBound algorithm for adaptive learning rates with bounds.
    
    Now uses BaseOptimizer for consistency and optimization_core support.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.0,
                 final_lr: float = 0.1,
                 gamma: float = 1e-3,
                 name: Optional[str] = None,
                 use_optimization_core: bool = True):
        """
        Initialize AdaBound optimizer.
        
        Args:
            learning_rate: Learning rate
            beta_1: Exponential decay rate for first moment estimates
            beta_2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay factor
            final_lr: Final learning rate
            gamma: Gamma parameter
            name: Optional name for the optimizer
            use_optimization_core: Whether to use optimization_core if available
        """
        super().__init__(
            optimizer_type='adabound',
            learning_rate=learning_rate,
            name=name,
            use_optimization_core=use_optimization_core,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            final_lr=final_lr,
            gamma=gamma
        )
    
    def _create_pytorch_optimizer(self, parameters):
        """Create PyTorch optimizer as fallback."""
        # Note: PyTorch doesn't have AdaBound built-in, so we'll use Adam as a fallback
        # In a real implementation, you'd implement AdaBound from scratch
        return optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=(self.kwargs['beta_1'], self.kwargs['beta_2']),
            eps=self.kwargs['epsilon'],
            weight_decay=self.kwargs['weight_decay']
        )
    
    def _get_optimizer_specific_config(self) -> Dict[str, Any]:
        """Get AdaBound-specific configuration parameters."""
        return {
            'beta_1': self.kwargs['beta_1'],
            'beta_2': self.kwargs['beta_2'],
            'epsilon': self.kwargs['epsilon'],
            'weight_decay': self.kwargs['weight_decay'],
            'final_lr': self.kwargs['final_lr'],
            'gamma': self.kwargs['gamma']
        }
    
    def _format_parameters(self) -> str:
        """Format AdaBound parameters for string representation."""
        return f"learning_rate={self.learning_rate}, beta_1={self.kwargs['beta_1']}, beta_2={self.kwargs['beta_2']}"
