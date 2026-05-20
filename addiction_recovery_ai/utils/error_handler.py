"""
Advanced Error Handling for Recovery AI
"""

import torch
import logging
from typing import Optional, Dict, Any
import traceback
import sys

logger = logging.getLogger(__name__)


class RecoveryAIError(Exception):
    """Base exception for Recovery AI"""
    pass


class ModelLoadError(RecoveryAIError):
    """Model loading error"""
    pass


class InferenceError(RecoveryAIError):
    """Inference error"""
    pass


class CUDAOutOfMemoryError(RecoveryAIError):
    """CUDA out of memory error"""
    pass


class ErrorHandler:
    """Advanced error handler"""
    
    @staticmethod
    def handle_cuda_oom(error: RuntimeError) -> Dict[str, Any]:
        """
        Handle CUDA out of memory error
        
        Args:
            error: CUDA OOM error
        
        Returns:
            Dictionary with error info and suggestions
        """
        if "out of memory" in str(error).lower():
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "error": "CUDA_OUT_OF_MEMORY",
                "message": "GPU out of memory",
                "suggestions": [
                    "Reduce batch size",
                    "Use CPU instead of GPU",
                    "Use quantized models",
                    "Clear GPU cache",
                    "Reduce model size"
                ],
                "action": "switched_to_cpu"
            }
        return None
    
    @staticmethod
    def handle_nan_inf(tensor: torch.Tensor, operation: str = "unknown") -> bool:
        """
        Check for NaN/Inf in tensor
        
        Args:
            tensor: Tensor to check
            operation: Operation name
        
        Returns:
            True if NaN/Inf detected
        """
        if torch.isnan(tensor).any():
            logger.error(f"NaN detected in {operation}")
            return True
        
        if torch.isinf(tensor).any():
            logger.error(f"Inf detected in {operation}")
            return True
        
        return False
    
    @staticmethod
    def safe_inference(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        fallback_value: Any = None,
        device: Optional[torch.device] = None
    ) -> Any:
        """
        Safe inference with error handling
        
        Args:
            model: Model to run inference
            inputs: Input tensor
            fallback_value: Fallback value on error
            device: Device to use
        
        Returns:
            Model output or fallback value
        """
        try:
            model.eval()
            with torch.no_grad():
                if device:
                    inputs = inputs.to(device)
                outputs = model(inputs)
                
                # Check for NaN/Inf
                if isinstance(outputs, torch.Tensor):
                    if ErrorHandler.handle_nan_inf(outputs, "inference"):
                        logger.warning("NaN/Inf in output, using fallback")
                        return fallback_value
                
                return outputs
        except RuntimeError as e:
            oom_info = ErrorHandler.handle_cuda_oom(e)
            if oom_info:
                logger.error(f"CUDA OOM: {oom_info['message']}")
                return fallback_value
            raise InferenceError(f"Inference failed: {e}")
        except Exception as e:
            logger.error(f"Inference error: {e}\n{traceback.format_exc()}")
            return fallback_value
    
    @staticmethod
    def validate_input(
        data: Any,
        expected_type: type,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> bool:
        """
        Validate input data
        
        Args:
            data: Data to validate
            expected_type: Expected type
            min_value: Minimum value (if numeric)
            max_value: Maximum value (if numeric)
        
        Returns:
            True if valid
        """
        if not isinstance(data, expected_type):
            logger.error(f"Invalid type: expected {expected_type}, got {type(data)}")
            return False
        
        if isinstance(data, (int, float)):
            if min_value is not None and data < min_value:
                logger.error(f"Value below minimum: {data} < {min_value}")
                return False
            if max_value is not None and data > max_value:
                logger.error(f"Value above maximum: {data} > {max_value}")
                return False
        
        return True
    
    @staticmethod
    def log_error(error: Exception, context: Dict[str, Any] = None):
        """
        Log error with context
        
        Args:
            error: Exception to log
            context: Additional context
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        
        if context:
            error_info["context"] = context
        
        logger.error(f"Error occurred: {error_info}")


def error_decorator(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ErrorHandler.log_error(e, {"function": func.__name__})
            raise
    return wrapper

