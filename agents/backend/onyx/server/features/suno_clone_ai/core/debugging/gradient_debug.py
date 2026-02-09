"""
Gradient Debugging

Utilities for debugging gradients during training.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class GradientDebugger:
    """Debug gradients during training."""
    
    def __init__(self, log_interval: int = 100):
        """
        Initialize gradient debugger.
        
        Args:
            log_interval: Log every N steps
        """
        self.log_interval = log_interval
        self.step_count = 0
        self.gradient_history = defaultdict(list)
    
    def check_gradients(
        self,
        model: nn.Module,
        step: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Check gradients for all parameters.
        
        Args:
            model: Model to check
            step: Current step number
            
        Returns:
            Dictionary with gradient statistics
        """
        self.step_count = step or self.step_count
        
        stats = {
            'total_norm': 0.0,
            'max_norm': 0.0,
            'min_norm': float('inf'),
            'num_zero_grads': 0,
            'num_nan_grads': 0,
            'num_inf_grads': 0
        }
        
        param_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                param_norms.append(grad_norm)
                
                stats['total_norm'] += grad_norm ** 2
                stats['max_norm'] = max(stats['max_norm'], grad_norm)
                stats['min_norm'] = min(stats['min_norm'], grad_norm)
                
                # Check for NaN/Inf
                if torch.isnan(param.grad.data).any():
                    stats['num_nan_grads'] += 1
                    logger.warning(f"NaN gradient in {name}")
                
                if torch.isinf(param.grad.data).any():
                    stats['num_inf_grads'] += 1
                    logger.warning(f"Inf gradient in {name}")
                
                # Check for zero gradients
                if grad_norm == 0.0:
                    stats['num_zero_grads'] += 1
                
                # Store history
                self.gradient_history[name].append(grad_norm)
            else:
                stats['num_zero_grads'] += 1
        
        stats['total_norm'] = stats['total_norm'] ** 0.5
        
        if self.step_count % self.log_interval == 0:
            logger.info(f"Gradient stats: {stats}")
        
        return stats
    
    def log_gradient_norms(
        self,
        model: nn.Module,
        step: int
    ) -> None:
        """
        Log gradient norms for all parameters.
        
        Args:
            model: Model to check
            step: Current step number
        """
        if step % self.log_interval != 0:
            return
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                logger.info(f"Step {step} - {name}: grad_norm={grad_norm:.6f}")


def check_gradients(
    model: nn.Module,
    step: Optional[int] = None
) -> Dict[str, float]:
    """
    Convenience function to check gradients.
    
    Args:
        model: Model to check
        step: Current step number
        
    Returns:
        Gradient statistics
    """
    debugger = GradientDebugger()
    return debugger.check_gradients(model, step)


def log_gradient_norms(
    model: nn.Module,
    step: int
) -> None:
    """
    Convenience function to log gradient norms.
    
    Args:
        model: Model to check
        step: Current step number
    """
    debugger = GradientDebugger()
    debugger.log_gradient_norms(model, step)



