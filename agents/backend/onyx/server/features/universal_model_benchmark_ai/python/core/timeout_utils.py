"""
Timeout Utilities - Comprehensive timeout handling.

Provides:
- Cross-platform timeout support
- Timeout context managers
- Timeout decorators
- Timeout configuration
"""

import time
import logging
import threading
from typing import Callable, Any, Optional, TypeVar
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TimeoutException(Exception):
    """Raised when an operation times out."""
    def __init__(self, message: str, timeout: float):
        super().__init__(message)
        self.timeout = timeout


def _timeout_handler(timeout: float):
    """Internal timeout handler."""
    def handler():
        time.sleep(timeout)
        raise TimeoutException(f"Operation timed out after {timeout}s", timeout)
    return handler


@contextmanager
def timeout_context(timeout: float, operation: str = "Operation"):
    """
    Context manager for timeout support (cross-platform).
    
    Args:
        timeout: Timeout in seconds
        operation: Description of operation
    
    Raises:
        TimeoutException: If operation times out
    
    Example:
        >>> with timeout_context(5.0, "Loading model"):
        >>>     model = load_model()
    """
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    
    # Use threading for cross-platform support
    timer = None
    exception_container = []
    
    def timeout_handler():
        exception_container.append(TimeoutException(
            f"{operation} timed out after {timeout}s",
            timeout
        ))
    
    timer = threading.Timer(timeout, timeout_handler)
    timer.start()
    
    try:
        yield
    except TimeoutException:
        raise
    finally:
        if timer:
            timer.cancel()
    
    if exception_container:
        raise exception_container[0]


def with_timeout(timeout: float, default_return: Any = None):
    """
    Decorator to add timeout to a function.
    
    Args:
        timeout: Timeout in seconds
        default_return: Return value on timeout
    
    Example:
        >>> @with_timeout(5.0, default_return=None)
        >>> def slow_function():
        >>>     time.sleep(10)
        >>>     return "done"
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                with timeout_context(timeout, f"{func.__name__}()"):
                    return func(*args, **kwargs)
            except TimeoutException as e:
                logger.warning(f"{func.__name__} timed out: {e}")
                return default_return
        return wrapper
    return decorator


def execute_with_timeout(
    func: Callable[..., T],
    timeout: float,
    *args,
    default_return: Any = None,
    **kwargs,
) -> T:
    """
    Execute a function with timeout.
    
    Args:
        func: Function to execute
        timeout: Timeout in seconds
        *args: Positional arguments
        default_return: Return value on timeout
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_return on timeout
    
    Raises:
        TimeoutException: If operation times out (unless default_return is provided)
    
    Example:
        >>> result = execute_with_timeout(slow_function, 5.0, arg1, arg2)
    """
    try:
        with timeout_context(timeout, f"{func.__name__}()"):
            return func(*args, **kwargs)
    except TimeoutException as e:
        logger.warning(f"{func.__name__} timed out: {e}")
        if default_return is not None:
            return default_return
        raise


class TimeoutManager:
    """
    Manager for timeout operations.
    
    Provides centralized timeout configuration and management.
    """
    
    def __init__(self, default_timeout: Optional[float] = None):
        """
        Initialize timeout manager.
        
        Args:
            default_timeout: Default timeout in seconds
        """
        self.default_timeout = default_timeout
        self._active_timers: Dict[str, threading.Timer] = {}
    
    def set_default_timeout(self, timeout: float) -> None:
        """Set default timeout."""
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.default_timeout = timeout
    
    def execute(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        default_return: Any = None,
        operation_id: Optional[str] = None,
        **kwargs,
    ) -> T:
        """
        Execute function with timeout.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            timeout: Timeout in seconds (uses default if None)
            default_return: Return value on timeout
            operation_id: Optional operation ID for tracking
            **kwargs: Keyword arguments
        
        Returns:
            Function result or default_return on timeout
        """
        timeout = timeout or self.default_timeout
        if timeout is None:
            return func(*args, **kwargs)
        
        return execute_with_timeout(
            func,
            timeout,
            *args,
            default_return=default_return,
            **kwargs,
        )
    
    @contextmanager
    def context(
        self,
        timeout: Optional[float] = None,
        operation: str = "Operation",
    ):
        """
        Get timeout context.
        
        Args:
            timeout: Timeout in seconds (uses default if None)
            operation: Description of operation
        """
        timeout = timeout or self.default_timeout
        if timeout is None:
            yield
        else:
            with timeout_context(timeout, operation):
                yield


# Global timeout manager instance
_default_timeout_manager = TimeoutManager()


def get_timeout_manager() -> TimeoutManager:
    """Get default timeout manager."""
    return _default_timeout_manager


__all__ = [
    "TimeoutException",
    "timeout_context",
    "with_timeout",
    "execute_with_timeout",
    "TimeoutManager",
    "get_timeout_manager",
]












