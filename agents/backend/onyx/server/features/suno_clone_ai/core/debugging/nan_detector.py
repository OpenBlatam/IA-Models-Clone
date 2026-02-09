"""
NaN/Inf Detection

Utilities for detecting NaN and Inf values in models and tensors.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NaNDetector:
    """Detect NaN and Inf values in models and tensors."""
    
    @staticmethod
    def check_tensor(
        tensor: torch.Tensor,
        name: str = "tensor"
    ) -> Dict[str, bool]:
        """
        Check tensor for NaN/Inf.
        
        Args:
            tensor: Tensor to check
            name: Name for logging
            
        Returns:
            Dictionary with detection results
        """
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan:
            logger.warning(f"NaN detected in {name}")
        
        if has_inf:
            logger.warning(f"Inf detected in {name}")
        
        return {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'num_nan': torch.isnan(tensor).sum().item() if has_nan else 0,
            'num_inf': torch.isinf(tensor).sum().item() if has_inf else 0
        }
    
    @staticmethod
    def check_model(
        model: nn.Module,
        check_gradients: bool = True
    ) -> Dict[str, List[str]]:
        """
        Check model parameters and gradients for NaN/Inf.
        
        Args:
            model: Model to check
            check_gradients: Also check gradients
            
        Returns:
            Dictionary with problematic parameter names
        """
        issues = {
            'nan_params': [],
            'inf_params': [],
            'nan_grads': [],
            'inf_grads': []
        }
        
        for name, param in model.named_parameters():
            # Check parameters
            if torch.isnan(param.data).any():
                issues['nan_params'].append(name)
                logger.warning(f"NaN in parameter: {name}")
            
            if torch.isinf(param.data).any():
                issues['inf_params'].append(name)
                logger.warning(f"Inf in parameter: {name}")
            
            # Check gradients
            if check_gradients and param.grad is not None:
                if torch.isnan(param.grad.data).any():
                    issues['nan_grads'].append(name)
                    logger.warning(f"NaN in gradient: {name}")
                
                if torch.isinf(param.grad.data).any():
                    issues['inf_grads'].append(name)
                    logger.warning(f"Inf in gradient: {name}")
        
        return issues


def check_for_nan_inf(
    tensor: torch.Tensor,
    name: str = "tensor"
) -> Dict[str, bool]:
    """
    Convenience function to check tensor for NaN/Inf.
    
    Args:
        tensor: Tensor to check
        name: Name for logging
        
    Returns:
        Detection results
    """
    return NaNDetector.check_tensor(tensor, name)


def detect_nan_in_model(
    model: nn.Module,
    check_gradients: bool = True
) -> Dict[str, List[str]]:
    """
    Convenience function to detect NaN/Inf in model.
    
    Args:
        model: Model to check
        check_gradients: Also check gradients
        
    Returns:
        Dictionary with issues
    """
    return NaNDetector.check_model(model, check_gradients)



