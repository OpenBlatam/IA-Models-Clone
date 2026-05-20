"""
Debugging Utilities
===================

Utilities for debugging PyTorch models and training.
"""

import torch
import logging
from typing import Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Debugger:
    """Debugging utilities for PyTorch."""
    
    def __init__(self, enable_anomaly_detection: bool = False):
        """
        Initialize debugger.
        
        Args:
            enable_anomaly_detection: Enable anomaly detection
        """
        self.enable_anomaly_detection = enable_anomaly_detection
        self._logger = logger
    
    @contextmanager
    def detect_anomaly(self):
        """
        Context manager for anomaly detection.
        
        Usage:
            with debugger.detect_anomaly():
                # Training code
        """
        if self.enable_anomaly_detection:
            with torch.autograd.detect_anomaly():
                yield
        else:
            yield
    
    @staticmethod
    def check_gradients(model: torch.nn.Module, threshold: float = 1e-6) -> Dict[str, float]:
        """
        Check for vanishing/exploding gradients.
        
        Args:
            model: Model to check
            threshold: Threshold for gradient magnitude
        
        Returns:
            Dictionary of gradient statistics
        """
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = grad_norm
                
                if grad_norm < threshold:
                    logger.warning(f"Vanishing gradient detected in {name}: {grad_norm}")
                elif grad_norm > 100:
                    logger.warning(f"Exploding gradient detected in {name}: {grad_norm}")
        
        return grad_stats
    
    @staticmethod
    def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """
        Check for NaN or Inf values.
        
        Args:
            tensor: Tensor to check
            name: Name for logging
        
        Returns:
            True if NaN/Inf found
        """
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan:
            logger.error(f"NaN detected in {name}")
        if has_inf:
            logger.error(f"Inf detected in {name}")
        
        return has_nan or has_inf
    
    @staticmethod
    def print_model_summary(model: torch.nn.Module):
        """
        Print model summary.
        
        Args:
            model: Model to summarize
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model Summary:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Print layer information
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    logger.info(f"    {name}: {params:,} parameters")




