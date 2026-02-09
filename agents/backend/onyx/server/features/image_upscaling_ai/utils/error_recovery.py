"""
Error Recovery
=============

Advanced error recovery mechanisms.
"""

import logging
import time
from typing import Callable, Any, Optional, Dict
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential base for backoff
        exceptions: Exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def fallback_on_error(fallback_func: Callable, exceptions: tuple = (Exception,)):
    """
    Fallback decorator that calls fallback function on error.
    
    Args:
        fallback_func: Fallback function to call
        exceptions: Exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                logger.warning(f"Error in {func.__name__}: {e}. Using fallback.")
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.warning(f"Error in {func.__name__}: {e}. Using fallback.")
                return fallback_func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ErrorRecovery:
    """
    Error recovery manager.
    
    Features:
    - Automatic retry
    - Fallback strategies
    - Error classification
    - Recovery strategies
    """
    
    def __init__(self):
        """Initialize error recovery."""
        self.recovery_strategies = {}
        logger.info("ErrorRecovery initialized")
    
    def register_strategy(
        self,
        error_type: type,
        strategy: Callable
    ) -> None:
        """
        Register recovery strategy for error type.
        
        Args:
            error_type: Error type to handle
            strategy: Recovery strategy function
        """
        self.recovery_strategies[error_type] = strategy
        logger.debug(f"Registered recovery strategy for {error_type}")
    
    async def recover(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Attempt to recover from error.
        
        Args:
            error: The error that occurred
            context: Additional context
            
        Returns:
            Recovery result or None
        """
        error_type = type(error)
        
        # Find matching strategy
        strategy = self.recovery_strategies.get(error_type)
        if not strategy:
            # Try base exception
            strategy = self.recovery_strategies.get(Exception)
        
        if strategy:
            try:
                if asyncio.iscoroutinefunction(strategy):
                    return await strategy(error, context or {})
                else:
                    return strategy(error, context or {})
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
        
        return None


