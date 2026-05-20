"""
Structured logging configuration.
"""

import sys
from typing import Any, Dict
import structlog
from structlog.stdlib import LoggerFactory

from .config import get_settings


def configure_logging() -> None:
    """Configure structured logging."""
    settings = get_settings()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" 
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_function_call(func_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Log function call with parameters."""
    return {
        "function": func_name,
        "parameters": kwargs,
        "event": "function_call"
    }


def log_function_result(func_name: str, result: Any, **kwargs: Any) -> Dict[str, Any]:
    """Log function result."""
    return {
        "function": func_name,
        "result": result,
        "parameters": kwargs,
        "event": "function_result"
    }


def log_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Log error with context."""
    return {
        "error": str(error),
        "error_type": type(error).__name__,
        "context": context or {},
        "event": "error"
    }




