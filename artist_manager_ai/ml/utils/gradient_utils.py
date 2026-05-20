"""
Gradient Utilities
=================

Utilities for gradient analysis and optimization.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GradientAnalyzer:
    """Analyze gradients during training."""
    
    @staticmethod
    def analyze_gradients(
        model: nn.Module,
        threshold: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Analyze gradients.
        
        Args:
            model: PyTorch model
            threshold: Threshold for vanishing gradients
        
        Returns:
            Gradient analysis
        """
        grad_stats = {}
        vanishing_grads = []
        exploding_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = {
                    "norm": grad_norm,
                    "mean": param.grad.mean().item(),
                    "std": param.grad.std().item(),
                    "max": param.grad.max().item(),
                    "min": param.grad.min().item()
                }
                
                if grad_norm < threshold:
                    vanishing_grads.append(name)
                elif grad_norm > 100:
                    exploding_grads.append(name)
        
        return {
            "gradient_stats": grad_stats,
            "vanishing_gradients": vanishing_grads,
            "exploding_gradients": exploding_grads,
            "total_params_with_grad": len(grad_stats)
        }
    
    @staticmethod
    def clip_gradients(
        model: nn.Module,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ) -> float:
        """
        Clip gradients.
        
        Args:
            model: PyTorch model
            max_norm: Maximum norm
            norm_type: Norm type
        
        Returns:
            Total gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type
        )
        return total_norm.item()
    
    @staticmethod
    def zero_gradients(model: nn.Module):
        """Zero all gradients."""
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()


class GradientAccumulator:
    """Gradient accumulator for large batch simulation."""
    
    def __init__(self, accumulation_steps: int = 4):
        """
        Initialize accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        """Check if should perform optimizer step."""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False
    
    def reset(self):
        """Reset accumulator."""
        self.current_step = 0




