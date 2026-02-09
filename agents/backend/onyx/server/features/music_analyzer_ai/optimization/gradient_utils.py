"""
Modular Gradient Utilities
Gradient clipping, accumulation, and monitoring
"""

from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class GradientClipper:
    """Gradient clipping utility"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip(self, parameters) -> float:
        """
        Clip gradients
        
        Args:
            parameters: Model parameters
        
        Returns:
            Gradient norm before clipping
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )


class GradientAccumulator:
    """Gradient accumulation utility"""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self):
        """Reset accumulator"""
        self.current_step = 0


class GradientMonitor:
    """Monitor gradient statistics"""
    
    def __init__(self):
        self.gradient_stats: List[Dict[str, float]] = []
    
    def monitor(self, parameters, step: int = 0) -> Dict[str, float]:
        """
        Monitor gradient statistics
        
        Args:
            parameters: Model parameters
            step: Current step
        
        Returns:
            Dictionary of gradient statistics
        """
        if not TORCH_AVAILABLE:
            return {}
        
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        num_params = 0
        
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, param.grad.data.abs().max().item())
                min_grad = min(min_grad, param.grad.data.abs().min().item())
                num_params += 1
        
        total_norm = total_norm ** (1. / 2)
        
        stats = {
            "grad_norm": total_norm,
            "max_grad": max_grad,
            "min_grad": min_grad if min_grad != float('inf') else 0.0,
            "num_params_with_grad": num_params,
            "step": step
        }
        
        self.gradient_stats.append(stats)
        return stats
    
    def get_stats(self) -> List[Dict[str, float]]:
        """Get all gradient statistics"""
        return self.gradient_stats.copy()
    
    def clear(self):
        """Clear statistics"""
        self.gradient_stats.clear()



