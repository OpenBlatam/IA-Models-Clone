"""
Training Debugger
Debugging utilities for training
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class TrainingDebugger:
    """Debugging utilities for training"""
    
    def __init__(self, enable_anomaly_detection: bool = False):
        self.enable_anomaly_detection = enable_anomaly_detection
        if enable_anomaly_detection and TORCH_AVAILABLE:
            torch.autograd.set_detect_anomaly(True)
            logger.info("Anomaly detection enabled")
    
    def check_gradients(
        self,
        model: nn.Module,
        step: int = 0
    ) -> Dict[str, Any]:
        """
        Check gradient statistics
        
        Args:
            model: Model to check
            step: Current step
        
        Returns:
            Dictionary of gradient statistics
        """
        if not TORCH_AVAILABLE:
            return {}
        
        stats = {
            "step": step,
            "parameters_with_grad": 0,
            "parameters_without_grad": 0,
            "zero_gradients": 0,
            "nan_gradients": 0,
            "inf_gradients": 0,
            "max_grad": 0.0,
            "min_grad": float('inf')
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    stats["parameters_with_grad"] += 1
                    
                    # Check for NaN/Inf
                    if torch.isnan(param.grad).any():
                        stats["nan_gradients"] += 1
                        logger.warning(f"NaN gradient in {name}")
                    
                    if torch.isinf(param.grad).any():
                        stats["inf_gradients"] += 1
                        logger.warning(f"Inf gradient in {name}")
                    
                    # Check for zero gradients
                    if (param.grad == 0).all():
                        stats["zero_gradients"] += 1
                    
                    # Gradient magnitude
                    grad_norm = param.grad.norm().item()
                    stats["max_grad"] = max(stats["max_grad"], grad_norm)
                    stats["min_grad"] = min(stats["min_grad"], grad_norm)
                else:
                    stats["parameters_without_grad"] += 1
            else:
                stats["parameters_without_grad"] += 1
        
        if stats["min_grad"] == float('inf'):
            stats["min_grad"] = 0.0
        
        return stats
    
    def check_weights(
        self,
        model: nn.Module,
        step: int = 0
    ) -> Dict[str, Any]:
        """
        Check weight statistics
        
        Args:
            model: Model to check
            step: Current step
        
        Returns:
            Dictionary of weight statistics
        """
        if not TORCH_AVAILABLE:
            return {}
        
        stats = {
            "step": step,
            "total_parameters": 0,
            "nan_weights": 0,
            "inf_weights": 0,
            "max_weight": 0.0,
            "min_weight": float('inf')
        }
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            stats["total_parameters"] += num_params
            
            # Check for NaN/Inf
            if torch.isnan(param).any():
                stats["nan_weights"] += num_params
                logger.warning(f"NaN weights in {name}")
            
            if torch.isinf(param).any():
                stats["inf_weights"] += num_params
                logger.warning(f"Inf weights in {name}")
            
            # Weight magnitude
            weight_max = param.abs().max().item()
            weight_min = param.abs().min().item()
            stats["max_weight"] = max(stats["max_weight"], weight_max)
            stats["min_weight"] = min(stats["min_weight"], weight_min)
        
        if stats["min_weight"] == float('inf'):
            stats["min_weight"] = 0.0
        
        return stats



