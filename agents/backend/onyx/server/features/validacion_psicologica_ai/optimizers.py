"""
Custom Optimizers
=================
Optimizers with advanced features
"""

from typing import Dict, Any, Optional, List
import torch
import torch.optim as optim
import structlog

logger = structlog.get_logger()


class OptimizerFactory:
    """Factory for creating optimizers"""
    
    @staticmethod
    def create_optimizer(
        optimizer_type: str,
        model_parameters,
        learning_rate: float = 1e-3,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create optimizer
        
        Args:
            optimizer_type: Type of optimizer (adam, adamw, sgd, rmsprop)
            model_parameters: Model parameters
            learning_rate: Learning rate
            **kwargs: Additional optimizer arguments
            
        Returns:
            Optimizer instance
        """
        if optimizer_type.lower() == "adam":
            return optim.Adam(
                model_parameters,
                lr=learning_rate,
                **kwargs
            )
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(
                model_parameters,
                lr=learning_rate,
                **kwargs
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                model_parameters,
                lr=learning_rate,
                **kwargs
            )
        elif optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(
                model_parameters,
                lr=learning_rate,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class LookaheadOptimizer:
    """
    Lookahead optimizer wrapper
    Improves training stability
    """
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5
    ):
        """
        Initialize lookahead optimizer
        
        Args:
            base_optimizer: Base optimizer
            k: Update frequency
            alpha: Interpolation factor
        """
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Store slow weights
        self.slow_weights = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.slow_weights[p] = p.data.clone()
    
    def step(self, closure=None):
        """Perform optimization step"""
        loss = self.base_optimizer.step(closure)
        self.step_count += 1
        
        # Update slow weights every k steps
        if self.step_count % self.k == 0:
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    if p in self.slow_weights:
                        # Interpolate
                        p.data.mul_(self.alpha).add_(
                            self.slow_weights[p],
                            alpha=1 - self.alpha
                        )
                        self.slow_weights[p].copy_(p.data)
        
        return loss
    
    def zero_grad(self):
        """Zero gradients"""
        self.base_optimizer.zero_grad()
    
    def state_dict(self):
        """Get state dict"""
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "slow_weights": self.slow_weights,
            "step_count": self.step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.slow_weights = state_dict["slow_weights"]
        self.step_count = state_dict["step_count"]


class GradientCentralizationOptimizer:
    """
    Gradient Centralization optimizer wrapper
    Improves generalization
    """
    
    def __init__(self, base_optimizer: torch.optim.Optimizer):
        """
        Initialize gradient centralization optimizer
        
        Args:
            base_optimizer: Base optimizer
        """
        self.base_optimizer = base_optimizer
    
    def _centralize_gradients(self, param_group):
        """Centralize gradients"""
        for p in param_group['params']:
            if p.grad is not None:
                # Centralize gradient
                grad = p.grad.data
                grad.add_(-grad.mean(dim=tuple(range(1, grad.ndim)), keepdim=True))
    
    def step(self, closure=None):
        """Perform optimization step with gradient centralization"""
        # Centralize gradients
        for group in self.base_optimizer.param_groups:
            self._centralize_gradients(group)
        
        return self.base_optimizer.step(closure)
    
    def zero_grad(self):
        """Zero gradients"""
        self.base_optimizer.zero_grad()
    
    def state_dict(self):
        """Get state dict"""
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.base_optimizer.load_state_dict(state_dict)


# Factory function
def create_optimizer(
    optimizer_type: str,
    model_parameters,
    learning_rate: float = 1e-3,
    use_lookahead: bool = False,
    use_gradient_centralization: bool = False,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with optional enhancements
    
    Args:
        optimizer_type: Type of optimizer
        model_parameters: Model parameters
        learning_rate: Learning rate
        use_lookahead: Use lookahead wrapper
        use_gradient_centralization: Use gradient centralization
        **kwargs: Additional arguments
        
    Returns:
        Optimizer instance
    """
    base_optimizer = OptimizerFactory.create_optimizer(
        optimizer_type,
        model_parameters,
        learning_rate,
        **kwargs
    )
    
    if use_gradient_centralization:
        base_optimizer = GradientCentralizationOptimizer(base_optimizer)
    
    if use_lookahead:
        base_optimizer = LookaheadOptimizer(base_optimizer)
    
    return base_optimizer




