"""
Advanced Error Handler
======================

Advanced error handling system.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Error information."""
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    handled: bool = False
    retryable: bool = False


class ErrorHandler:
    """Advanced error handler."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.error_history: List[ErrorInfo] = []
        self.max_history = 1000
    
    def register_handler(self, exception_type: Type[Exception], handler: Callable):
        """
        Register error handler for exception type.
        
        Args:
            exception_type: Exception type
            handler: Handler function
        """
        self.error_handlers[exception_type] = handler
        logger.debug(f"Registered handler for {exception_type.__name__}")
    
    def handle_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        retryable: bool = False
    ) -> ErrorInfo:
        """
        Handle error.
        
        Args:
            error: Exception instance
            severity: Error severity
            context: Optional context
            retryable: Whether error is retryable
            
        Returns:
            Error information
        """
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            context=context or {},
            stack_trace=traceback.format_exc(),
            retryable=retryable
        )
        
        # Check for registered handler
        error_type = type(error)
        if error_type in self.error_handlers:
            try:
                self.error_handlers[error_type](error, error_info)
                error_info.handled = True
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
        
        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.ERROR)
        
        logger.log(log_level, f"Error: {error_info.message}", exc_info=error)
        
        return error_info
    
    def get_error_history(
        self,
        error_type: Optional[str] = None,
        severity: Optional[ErrorSeverity] = None,
        limit: int = 100
    ) -> List[ErrorInfo]:
        """
        Get error history.
        
        Args:
            error_type: Optional error type filter
            severity: Optional severity filter
            limit: Maximum number of results
            
        Returns:
            List of error information
        """
        errors = self.error_history
        
        if error_type:
            errors = [e for e in errors if e.error_type == error_type]
        
        if severity:
            errors = [e for e in errors if e.severity == severity]
        
        return errors[-limit:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        total = len(self.error_history)
        by_type = {}
        by_severity = {}
        
        for error in self.error_history:
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
            by_severity[error.severity.value] = by_severity.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "retryable_count": sum(1 for e in self.error_history if e.retryable),
            "handled_count": sum(1 for e in self.error_history if e.handled)
        }
    
    def clear_history(self):
        """Clear error history."""


class ErrorHandlerDecorator:
    """Decorator for error handling."""
    
    def __init__(self, error_handler: ErrorHandler):
        """
        Initialize decorator.
        
        Args:
            error_handler: Error handler instance
        """
        self.error_handler = error_handler
    
    def handle(
        self,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        retryable: bool = False,
        reraise: bool = True
    ):
        """
        Decorator for error handling.
        
        Args:
            severity: Error severity
            retryable: Whether error is retryable
            reraise: Whether to reraise exception
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            import functools
            import asyncio
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = {
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                    self.error_handler.handle_error(e, severity, context, retryable)
                    if reraise:
                        raise
                    return None
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                    self.error_handler.handle_error(e, severity, context, retryable)
                    if reraise:
                        raise
                    return None
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator

