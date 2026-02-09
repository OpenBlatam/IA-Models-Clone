"""Context utilities."""

from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager, asynccontextmanager
import threading
import time
from contextlib import AsyncGenerator

from utils.logger import get_logger
from utils.metrics import metrics_collector

logger = get_logger(__name__)


class ContextManager:
    """Thread-local context manager."""
    
    def __init__(self):
        self._local = threading.local()
    
    def set(self, key: str, value: Any) -> None:
        """
        Set context value.
        
        Args:
            key: Context key
            value: Value to set
        """
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get context value.
        
        Args:
            key: Context key
            default: Default value
            
        Returns:
            Context value or default
        """
        if not hasattr(self._local, 'context'):
            return default
        return self._local.context.get(key, default)
    
    def clear(self) -> None:
        """Clear all context."""
        if hasattr(self._local, 'context'):
            self._local.context.clear()
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all context values.
        
        Returns:
            Dictionary of all context values
        """
        if not hasattr(self._local, 'context'):
            return {}
        return self._local.context.copy()


# Global context manager
_context = ContextManager()


def set_context(key: str, value: Any) -> None:
    """Set context value."""
    _context.set(key, value)


def get_context(key: str, default: Any = None) -> Any:
    """Get context value."""
    return _context.get(key, default)


def clear_context() -> None:
    """Clear all context."""
    _context.clear()


@contextmanager
def context(**kwargs):
    """
    Context manager for setting temporary context.
    
    Args:
        **kwargs: Context key-value pairs
        
    Yields:
        None
    """
    old_values = {}
    
    # Save old values
    for key in kwargs:
        old_values[key] = get_context(key)
    
    # Set new values
    for key, value in kwargs.items():
        set_context(key, value)
    
    try:
        yield
    finally:
        # Restore old values
        for key, value in old_values.items():
            if value is not None:
                set_context(key, value)
            else:
                _context._local.context.pop(key, None)


# Context managers from context_managers.py
@asynccontextmanager
async def timing_context(operation_name: str) -> AsyncGenerator[None, None]:
    """Context manager to track operation timing."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        metrics_collector.record_timing(f"{operation_name}.duration", duration)
        logger.debug(f"{operation_name} took {duration:.2f}s")


@asynccontextmanager
async def error_tracking_context(
    operation_name: str,
    on_error: Optional[Callable[[Exception], None]] = None
) -> AsyncGenerator[None, None]:
    """Context manager to track errors."""
    try:
        yield
    except Exception as e:
        metrics_collector.increment(f"{operation_name}.errors")
        logger.error(f"Error in {operation_name}: {e}")
        if on_error:
            on_error(e)
        raise

