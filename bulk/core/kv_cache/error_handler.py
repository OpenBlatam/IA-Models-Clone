"""
Error handling and recovery for KV Cache.

Provides robust error handling with automatic recovery.
"""
import logging
from typing import Optional, Callable, Any
import torch

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for cache errors."""
    pass


class CacheMemoryError(CacheError):
    """Memory-related cache error."""
    pass


class CacheValidationError(CacheError):
    """Validation-related cache error."""
    pass


class CacheDeviceError(CacheError):
    """Device-related cache error."""
    pass


class ErrorHandler:
    """
    Handles errors with automatic recovery strategies.
    
    Provides retry logic, graceful degradation, and error recovery.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retries
            retry_delay: Delay between retries (seconds)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._error_counts: dict = {}
    
    def handle_oom(self, func: Callable, *args, **kwargs) -> Any:
        """
        Handle out-of-memory errors with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CacheMemoryError: If OOM persists after retries
        """
        import time
        import gc
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        # Try to free memory
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        logger.warning(
                            f"OOM error on attempt {attempt + 1}/{self.max_retries}, "
                            f"retrying after {self.retry_delay}s"
                        )
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise CacheMemoryError(f"Out of memory after {self.max_retries} attempts: {e}") from e
                raise
        
        raise CacheMemoryError("Failed to execute function")
    
    def handle_device_error(self, func: Callable, *args, **kwargs) -> Any:
        """
        Handle device-related errors.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CacheDeviceError: If device error persists
        """
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "cuda" in str(e).lower() or "device" in str(e).lower():
                raise CacheDeviceError(f"Device error: {e}") from e
            raise
        except Exception as e:
            raise CacheDeviceError(f"Unexpected device error: {e}") from e
    
    def record_error(self, error_type: str) -> None:
        """Record error for monitoring."""
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
    
    def get_error_stats(self) -> dict:
        """Get error statistics."""
        return dict(self._error_counts)
    
    def reset_stats(self) -> None:
        """Reset error statistics."""
        self._error_counts.clear()


def safe_execute(func: Callable, error_handler: Optional[ErrorHandler] = None, *args, **kwargs) -> Any:
    """
    Safely execute function with error handling.
    
    Args:
        func: Function to execute
        error_handler: Optional error handler
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result or None on error
    """
    if error_handler is None:
        error_handler = ErrorHandler()
    
    try:
        return func(*args, **kwargs)
    except CacheMemoryError as e:
        error_handler.record_error("memory")
        logger.error(f"Memory error: {e}")
        return None
    except CacheDeviceError as e:
        error_handler.record_error("device")
        logger.error(f"Device error: {e}")
        return None
    except CacheValidationError as e:
        error_handler.record_error("validation")
        logger.error(f"Validation error: {e}")
        return None
    except Exception as e:
        error_handler.record_error("unknown")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return None

