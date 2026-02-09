"""
Advanced Debugging Tools
Debugging utilities for training and inference
"""

from typing import Optional, Callable
import logging
import warnings

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainingDebugger:
    """
    Debugging tools for training
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
    def check_gradients(model: torch.nn.Module, verbose: bool = False):
        """Check for gradient issues"""
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
    def check_weights(model: torch.nn.Module):
        """Check for weight issues"""
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
    def check_loss(loss: torch.Tensor):
        """Check loss value"""
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


class InferenceDebugger:
    """
    Debugging tools for inference
    """
    
    @staticmethod
    def validate_input(input_data: torch.Tensor, expected_shape: tuple):
        """Validate input shape and values"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if input_data.shape != expected_shape:
            raise ValueError(
                f"Input shape mismatch: expected {expected_shape}, got {input_data.shape}"
            )
        
        if torch.isnan(input_data).any():
            raise ValueError("Input contains NaN values")
        
        if torch.isinf(input_data).any():
            raise ValueError("Input contains Inf values")
    
    @staticmethod
    def validate_output(output: torch.Tensor, expected_shape: Optional[tuple] = None):
        """Validate output shape and values"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if expected_shape and output.shape != expected_shape:
            logger.warning(
                f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
            )
        
        if torch.isnan(output).any():
            logger.error("Output contains NaN values")
            return False
        
        if torch.isinf(output).any():
            logger.error("Output contains Inf values")
            return False
        
        return True


def debug_training_step(
    model: torch.nn.Module,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    check_gradients: bool = True,
    check_weights: bool = False
):
    """Debug a training step"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    # Check loss
    if not TrainingDebugger.check_loss(loss):
        return False
    
    # Check gradients
    if check_gradients:
        issues = TrainingDebugger.check_gradients(model)
        if issues:
            return False
    
    # Check weights
    if check_weights:
        issues = TrainingDebugger.check_weights(model)
        if issues:
            return False
    
    return True

