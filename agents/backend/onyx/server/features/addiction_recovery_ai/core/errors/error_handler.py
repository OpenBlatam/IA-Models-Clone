"""
Error Handler
Centralized error handling
"""

import torch
import logging
from typing import Optional, Callable, Any
from functools import wraps

from .custom_exceptions import (
    RecoveryAIError,
    ModelError,
    ModelInferenceError,
    CUDAOutOfMemoryError,
    DataValidationError
)

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Centralized error handler
    """
    
    @staticmethod
    def handle_cuda_error(error: Exception) -> RecoveryAIError:
        """
        Handle CUDA errors
        
        Args:
            error: Original error
            
        Returns:
            Appropriate exception
        """
        error_str = str(error).lower()
        
        if "out of memory" in error_str:
            return CUDAOutOfMemoryError(f"CUDA out of memory: {error}")
        elif "cuda" in error_str:
            return ModelInferenceError(f"CUDA error: {error}")
        else:
            return ModelError(f"Model error: {error}")
    
    @staticmethod
    def handle_model_error(error: Exception) -> ModelError:
        """
        Handle model errors
        
        Args:
            error: Original error
            
        Returns:
            Model error
        """
        if isinstance(error, ModelError):
            return error
        
        return ModelInferenceError(f"Model inference failed: {error}")
    
    @staticmethod
    def handle_data_error(error: Exception) -> DataValidationError:
        """
        Handle data errors
        
        Args:
            error: Original error
            
        Returns:
            Data validation error
        """
        return DataValidationError(f"Data validation failed: {error}")


def handle_errors(
    error_type: type = RecoveryAIError,
    default_return: Any = None,
    log_error: bool = True
):
    """
    Decorator for error handling
    
    Args:
        error_type: Exception type to catch
        default_return: Default return value on error
        log_error: Log errors
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                if default_return is not None:
                    return default_return
                raise
            except Exception as e:
                if log_error:
                    logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                
                # Handle specific errors
                if isinstance(e, RuntimeError) and "cuda" in str(e).lower():
                    raise ErrorHandler.handle_cuda_error(e)
                elif isinstance(e, (ValueError, TypeError)):
                    raise ErrorHandler.handle_data_error(e)
                else:
                    raise error_type(f"Error in {func.__name__}: {e}")
        
        return wrapper
    return decorator


def safe_inference(func: Callable) -> Callable:
    """
    Decorator for safe model inference
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Clear cache and retry once
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise CUDAOutOfMemoryError(f"CUDA OOM: {e}")
            raise ModelInferenceError(f"Inference failed: {e}")
        except Exception as e:
            raise ModelInferenceError(f"Inference error: {e}")
    
    return wrapper













