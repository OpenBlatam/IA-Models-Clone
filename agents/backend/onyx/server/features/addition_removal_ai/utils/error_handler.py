"""
Advanced Error Handling and Debugging
"""

import torch
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
import traceback
import sys

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Advanced error handler with debugging"""
    
    def __init__(self, enable_debugging: bool = False):
        """
        Initialize error handler
        
        Args:
            enable_debugging: Enable PyTorch debugging
        """
        self.enable_debugging = enable_debugging
        if enable_debugging:
            torch.autograd.set_detect_anomaly(True)
            logger.info("PyTorch anomaly detection enabled")
    
    @staticmethod
    def handle_errors(func: Callable) -> Callable:
        """Decorator for error handling"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM: {e}")
                torch.cuda.empty_cache()
                return {"success": False, "error": "Out of memory. Try reducing batch size."}
            except RuntimeError as e:
                logger.error(f"Runtime error: {e}")
                return {"success": False, "error": f"Runtime error: {str(e)}"}
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}\n{traceback.format_exc()}")
                return {"success": False, "error": str(e)}
        return wrapper
    
    @staticmethod
    def validate_input(input_data: Any, expected_type: type, name: str = "input") -> bool:
        """Validate input data"""
        if not isinstance(input_data, expected_type):
            raise TypeError(f"{name} must be {expected_type.__name__}, got {type(input_data).__name__}")
        return True
    
    @staticmethod
    def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check for NaN/Inf in tensor"""
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {name}")
            return False
        return True
    
    @staticmethod
    def safe_forward(model: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Safe forward pass with error handling"""
        try:
            with torch.no_grad():
                output = model(*args, **kwargs)
            
            # Check for NaN/Inf
            if isinstance(output, torch.Tensor):
                ErrorHandler.check_nan_inf(output, "model_output")
            
            return output
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise


def create_error_handler(enable_debugging: bool = False) -> ErrorHandler:
    """Factory function for error handler"""
    return ErrorHandler(enable_debugging=enable_debugging)

