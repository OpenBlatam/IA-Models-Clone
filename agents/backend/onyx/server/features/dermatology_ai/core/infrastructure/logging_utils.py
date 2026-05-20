"""
Structured logging utilities
Provides context-aware logging with performance metrics
"""

import logging
import time
from typing import Dict, Any, Optional
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Structured logger with context support"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set logging context"""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self._context.clear()
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with context"""
        context_str = " ".join([f"{k}={v}" for k, v in {**self._context, **kwargs}.items()])
        return f"{message} | {context_str}" if context_str else message
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message with context"""
        self.logger.error(self._format_message(message, **kwargs), exc_info=exc_info)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    @contextmanager
    def operation(self, operation_name: str, **context):
        """Context manager for logging operations with timing"""
        start_time = time.time()
        operation_context = {**self._context, **context, "operation": operation_name}
        
        try:
            self.logger.info(f"Starting {operation_name} | {' '.join([f'{k}={v}' for k, v in operation_context.items()])}")
            yield
            duration = time.time() - start_time
            self.logger.info(
                f"Completed {operation_name} | duration={duration:.3f}s | "
                f"{' '.join([f'{k}={v}' for k, v in operation_context.items()])}"
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"Failed {operation_name} | duration={duration:.3f}s | error={str(e)} | "
                f"{' '.join([f'{k}={v}' for k, v in operation_context.items()])}",
                exc_info=True
            )
            raise


def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Function {func_name} completed | duration={duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Function {func_name} failed | duration={duration:.3f}s | error={str(e)}",
                exc_info=True
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Function {func_name} completed | duration={duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Function {func_name} failed | duration={duration:.3f}s | error={str(e)}",
                exc_info=True
            )
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper















