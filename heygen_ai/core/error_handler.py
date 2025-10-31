"""
Error Handling Utilities for HeyGen AI
=====================================

Provides comprehensive error handling, retry logic, and error recovery mechanisms.
"""

import asyncio
import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling for the system."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_error_history = 1000
        
    def handle_error(self, error: Exception, context: str = "", **kwargs):
        """Handle an error with context."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": time.time(),
            "traceback": traceback.format_exc(),
            **kwargs
        }
        
        # Update error counts
        error_key = f"{context}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        # Log the error
        logger.error(f"Error in {context}: {error}", extra=error_info)
        
        return error_info
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of errors."""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts,
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Error history cleared")


def with_error_handling(
    error_handler: Optional[ErrorHandler] = None,
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False
):
    """Decorator to add error handling to functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    error_handler.handle_error(e, context=func.__name__)
                elif log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                
                return default_return
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    error_handler.handle_error(e, context=func.__name__)
                elif log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception,
    error_handler: Optional[ErrorHandler] = None
):
    """Decorator to add retry logic to functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}, "
                            f"retrying in {wait_time:.2f}s: {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        if error_handler:
                            error_handler.handle_error(e, context=f"{func.__name__}_retry_failed")
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}, "
                            f"retrying in {wait_time:.2f}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        if error_handler:
                            error_handler.handle_error(e, context=f"{func.__name__}_retry_failed")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }


class ErrorRecovery:
    """Error recovery strategies."""
    
    @staticmethod
    def exponential_backoff(
        func: Callable,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ) -> Callable:
        """Apply exponential backoff to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
            
            return None
        
        return wrapper
    
    @staticmethod
    async def async_exponential_backoff(
        func: Callable,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ) -> Callable:
        """Apply exponential backoff to an async function."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
            
            return None
        
        return wrapper


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_global_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler


def set_global_error_handler(error_handler: ErrorHandler):
    """Set the global error handler instance."""
    global _global_error_handler
    _global_error_handler = error_handler
