"""
Error Handling Module - Centralized error handling utilities.

Provides:
- Custom exception classes
- Error context managers
- Error aggregation
- Error formatting
- Error recovery strategies
"""

import logging
import traceback
from typing import Dict, Any, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Information about an error."""
    message: str
    exception_type: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "exception_type": self.exception_type,
            "severity": self.severity.value,
            "context": self.context,
            "traceback": self.traceback,
            "timestamp": self.timestamp,
        }


class BenchmarkError(Exception):
    """Base exception for benchmark-related errors."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.severity = ErrorSeverity.MEDIUM


class ModelLoadError(BenchmarkError):
    """Error loading a model."""
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(message, context={"model_name": model_name, **kwargs})
        self.severity = ErrorSeverity.HIGH


class DatasetLoadError(BenchmarkError):
    """Error loading a dataset."""
    def __init__(self, message: str, dataset_name: Optional[str] = None, **kwargs):
        super().__init__(message, context={"dataset_name": dataset_name, **kwargs})
        self.severity = ErrorSeverity.MEDIUM


class InferenceError(BenchmarkError):
    """Error during inference."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, context=kwargs)
        self.severity = ErrorSeverity.HIGH


class ValidationError(BenchmarkError):
    """Error during validation."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, context={"field": field, **kwargs})
        self.severity = ErrorSeverity.LOW


class TimeoutError(BenchmarkError):
    """Timeout error."""
    def __init__(self, message: str, timeout: Optional[float] = None, **kwargs):
        super().__init__(message, context={"timeout": timeout, **kwargs})
        self.severity = ErrorSeverity.MEDIUM


@dataclass
class ErrorCollector:
    """Collects and manages errors."""
    errors: List[ErrorInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(
        self,
        error: Union[Exception, str],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an error."""
        if isinstance(error, Exception):
            error_info = ErrorInfo(
                message=str(error),
                exception_type=type(error).__name__,
                severity=severity,
                context=context or {},
                traceback=traceback.format_exc(),
            )
        else:
            error_info = ErrorInfo(
                message=error,
                exception_type="Error",
                severity=severity,
                context=context or {},
            )
        
        self.errors.append(error_info)
        logger.error(f"Error collected: {error_info.message}", extra=error_info.context)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)
        logger.warning(warning)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def get_critical_errors(self) -> List[ErrorInfo]:
        """Get critical errors."""
        return [e for e in self.errors if e.severity == ErrorSeverity.CRITICAL]
    
    def get_high_severity_errors(self) -> List[ErrorInfo]:
        """Get high severity errors."""
        return [e for e in self.errors if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]]
    
    def clear(self) -> None:
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


@contextmanager
def error_context(
    operation: str,
    collector: Optional[ErrorCollector] = None,
    reraise: bool = True,
    default_return: Any = None,
):
    """
    Context manager for error handling.
    
    Args:
        operation: Description of operation
        collector: Optional error collector
        reraise: Whether to reraise exceptions
        default_return: Default return value on error
    
    Example:
        >>> with error_context("Loading model", collector) as ctx:
        >>>     model = load_model()
    """
    try:
        yield
    except Exception as e:
        error_info = ErrorInfo(
            message=f"{operation} failed: {str(e)}",
            exception_type=type(e).__name__,
            context={"operation": operation},
            traceback=traceback.format_exc(),
        )
        
        if collector:
            collector.add_error(e, context=error_info.context)
        
        logger.error(f"{operation} failed", exc_info=True)
        
        if reraise:
            raise
        else:
            return default_return


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    on_error: Optional[Callable[[Exception], Any]] = None,
    **kwargs,
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default_return: Default return value on error
        on_error: Optional error handler callback
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_return on error
    
    Example:
        >>> result = safe_execute(risky_function, arg1, arg2, default_return={})
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        if on_error:
            return on_error(e)
        return default_return


def format_error(error: Exception, include_traceback: bool = False) -> str:
    """
    Format an error for display.
    
    Args:
        error: Exception to format
        include_traceback: Whether to include traceback
    
    Returns:
        Formatted error string
    """
    error_str = f"{type(error).__name__}: {str(error)}"
    if include_traceback:
        error_str += f"\n{traceback.format_exc()}"
    return error_str


def get_error_summary(errors: List[ErrorInfo]) -> str:
    """
    Get summary of errors.
    
    Args:
        errors: List of error info
    
    Returns:
        Summary string
    """
    if not errors:
        return "No errors"
    
    by_severity = {}
    for error in errors:
        severity = error.severity.value
        by_severity.setdefault(severity, []).append(error)
    
    summary = f"Total errors: {len(errors)}\n"
    for severity in ["critical", "high", "medium", "low"]:
        if severity in by_severity:
            count = len(by_severity[severity])
            summary += f"  {severity.capitalize()}: {count}\n"
    
    return summary.strip()


__all__ = [
    "ErrorSeverity",
    "ErrorInfo",
    "BenchmarkError",
    "ModelLoadError",
    "DatasetLoadError",
    "InferenceError",
    "ValidationError",
    "TimeoutError",
    "ErrorCollector",
    "error_context",
    "safe_execute",
    "format_error",
    "get_error_summary",
]












