"""
Error Handling Utilities
=========================

Advanced error handling and recovery utilities:
- Context managers for error handling
- Automatic retry mechanisms
- Graceful degradation
- Error recovery strategies
"""

import logging
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps
import time
import torch

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Context manager for error handling with recovery."""
    
    def __init__(
        self,
        error_types: Tuple[Type[Exception], ...] = (Exception,),
        max_retries: int = 3,
        retry_delay: float = 1.0,
        on_error: Optional[Callable] = None,
        default_return: Any = None
    ):
        """
        Initialize error handler.
        
        Args:
            error_types: Types of errors to catch
            max_retries: Maximum number of retries
            retry_delay: Delay between retries (seconds)
            on_error: Callback function on error
            default_return: Default return value on failure
        """
        self.error_types = error_types
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.on_error = on_error
        self.default_return = default_return
        self.retry_count = 0
        self.last_error = None
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type and issubclass(exc_type, self.error_types):
            self.last_error = exc_val
            
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                logger.warning(
                    f"Error occurred (attempt {self.retry_count}/{self.max_retries}): {exc_val}"
                )
                time.sleep(self.retry_delay)
                return True  # Suppress exception and retry
            
            if self.on_error:
                self.on_error(exc_val)
            
            logger.error(f"Max retries exceeded. Last error: {exc_val}")
            return False  # Re-raise exception
        
        return False
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with error handling.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result or default_return
        """
        while self.retry_count < self.max_retries:
            try:
                return func(*args, **kwargs)
            except self.error_types as e:
                self.last_error = e
                self.retry_count += 1
                
                if self.retry_count < self.max_retries:
                    logger.warning(f"Retry {self.retry_count}/{self.max_retries}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    if self.on_error:
                        self.on_error(e)
                    logger.error(f"All retries failed: {e}")
                    return self.default_return
        
        return self.default_return


def retry_on_error(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    error_types: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for automatic retry on error.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Delay between retries
        error_types: Types of errors to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler(
                error_types=error_types,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            return handler.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def handle_cuda_errors(func: Callable) -> Callable:
    """
    Decorator for handling CUDA-specific errors.
    
    Args:
        func: Function to wrap
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM error: {e}")
            torch.cuda.empty_cache()
            raise
        except torch.cuda.CudaError as e:
            logger.error(f"CUDA error: {e}")
            raise
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA runtime error: {e}")
                torch.cuda.empty_cache()
            raise
    return wrapper


class GracefulDegradation:
    """Context manager for graceful degradation on errors."""
    
    def __init__(self, fallback: Optional[Callable] = None, log_errors: bool = True):
        """
        Initialize graceful degradation.
        
        Args:
            fallback: Fallback function to call on error
            log_errors: Whether to log errors
        """
        self.fallback = fallback
        self.log_errors = log_errors
        self.error_occurred = False
        self.last_error = None
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is not None:
            self.error_occurred = True
            self.last_error = exc_val
            
            if self.log_errors:
                logger.warning(f"Error occurred, using graceful degradation: {exc_val}")
            
            if self.fallback:
                try:
                    return self.fallback(exc_val)
                except Exception as e:
                    logger.error(f"Fallback also failed: {e}")
            
            return True  # Suppress exception
        
        return False



