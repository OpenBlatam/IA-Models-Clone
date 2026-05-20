"""
Training Debugging Module

Debugging utilities for training.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainingDebugger:
    """
    Debugging tools for training.
    """
    
    @staticmethod
    def enable_anomaly_detection():
        """Enable anomaly detection for debugging"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for anomaly detection")
        
        torch.autograd.set_detect_anomaly(True)
        logger.warning("Anomaly detection enabled - this will slow down training")
    
    @staticmethod
    def disable_anomaly_detection():
        """Disable anomaly detection"""
        if TORCH_AVAILABLE:
            torch.autograd.set_detect_anomaly(False)
    
    @staticmethod
    def check_gradients(model: torch.nn.Module, verbose: bool = False) -> List[str]:
        """
        Check for gradient issues.
        
        Args:
            model: Model to check
            verbose: Print detailed information
        
        Returns:
            List of gradient issues found
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        issues = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if torch.isnan(param.grad).any():
                    issues.append(f"{name}: NaN gradients")
                elif torch.isinf(param.grad).any():
                    issues.append(f"{name}: Inf gradients")
                elif grad_norm == 0:
                    issues.append(f"{name}: Zero gradients")
                elif grad_norm > 100:
                    issues.append(f"{name}: Very large gradients ({grad_norm:.2f})")
                
                if verbose:
                    logger.info(f"{name}: grad_norm={grad_norm:.4f}")
        
        if issues:
            logger.warning(f"Gradient issues found: {issues}")
        else:
            logger.info("No gradient issues detected")
        
        return issues
    
    @staticmethod
    def check_weights(model: torch.nn.Module) -> List[str]:
        """
        Check for weight issues.
        
        Args:
            model: Model to check
        
        Returns:
            List of weight issues found
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        issues = []
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                issues.append(f"{name}: NaN weights")
            elif torch.isinf(param).any():
                issues.append(f"{name}: Inf weights")
            elif param.abs().max().item() > 1e6:
                issues.append(f"{name}: Very large weights")
        
        if issues:
            logger.warning(f"Weight issues found: {issues}")
        else:
            logger.info("No weight issues detected")
        
        return issues
    
    @staticmethod
    def check_loss(loss: torch.Tensor) -> bool:
        """
        Check loss value.
        
        Args:
            loss: Loss tensor to check
        
        Returns:
            True if loss is valid, False otherwise
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        loss_value = loss.item()
        
        if torch.isnan(loss):
            logger.error("Loss is NaN!")
            return False
        elif torch.isinf(loss):
            logger.error("Loss is Inf!")
            return False
        elif loss_value > 1e6:
            logger.warning(f"Very large loss: {loss_value:.2f}")
            return False
        
        return True



