"""
Errors Module - Re-export error handling components.

This module groups all error-related functionality:
- Error Handling
- Error Recovery
"""

from .error_handling import (
    ErrorSeverity as ErrorSeverityLevel,
    ErrorInfo,
    ErrorContext as ErrorContextInfo,
    ErrorHandler,
    ErrorAggregator,
    format_error,
    log_error,
    handle_error,
)

from .error_recovery import (
    ErrorSeverity,
    RecoveryStrategy,
    ErrorContext,
    RecoveryPolicy,
    ErrorRecoveryManager,
    recover_from_error,
    classify_error,
)

__all__ = [
    # Error Handling
    "ErrorSeverityLevel",
    "ErrorInfo",
    "ErrorContextInfo",
    "ErrorHandler",
    "ErrorAggregator",
    "format_error",
    "log_error",
    "handle_error",
    # Error Recovery
    "ErrorSeverity",
    "RecoveryStrategy",
    "ErrorContext",
    "RecoveryPolicy",
    "ErrorRecoveryManager",
    "recover_from_error",
    "classify_error",
]




