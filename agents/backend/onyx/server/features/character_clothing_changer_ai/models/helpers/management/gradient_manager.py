"""
Gradient Manager
================

Manages gradient operations and monitoring.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class GradientManager:
    """Manages gradient operations."""
    
    @staticmethod
    def get_gradient_norm(model: nn.Module) -> float:
        """
        Calculate gradient norm.
        
        Args:
            model: PyTorch model
            
        Returns:
            Gradient norm
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    @staticmethod
    def clip_gradients(
        model: nn.Module,
        max_norm: float = 1.0
    ) -> float:
        """
        Clip gradients to prevent explosion.
        
        Args:
            model: PyTorch model
            max_norm: Maximum gradient norm
            
        Returns:
            Gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def zero_gradients(model: nn.Module) -> None:
        """Zero all gradients."""
        model.zero_grad()
    
    @staticmethod
    def get_gradient_stats(model: nn.Module) -> dict:
        """
        Get gradient statistics.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with gradient statistics
        """
        gradients = []
        for p in model.parameters():
            if p.grad is not None:
                gradients.append(p.grad.data)
        
        if not gradients:
            return {
                "has_gradients": False,
                "num_gradients": 0,
            }
        
        # Calculate statistics
        all_grads = torch.cat([g.flatten() for g in gradients])
        
        return {
            "has_gradients": True,
            "num_gradients": len(gradients),
            "mean": float(all_grads.mean()),
            "std": float(all_grads.std()),
            "min": float(all_grads.min()),
            "max": float(all_grads.max()),
            "norm": float(all_grads.norm()),
        }


