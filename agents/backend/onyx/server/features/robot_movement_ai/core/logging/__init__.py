"""
Structured Logging
==================

Sistema de logging estructurado.
"""

from .structured_logger import (
    StructuredLogger,
    setup_logging,
    get_logger,
    log_route_request,
    log_route_response,
    log_training_step,
    log_inference,
    log_error
)

from .formatters import (
    JSONFormatter,
    ColoredFormatter,
    StructuredFormatter
)

__all__ = [
    "StructuredLogger",
    "setup_logging",
    "get_logger",
    "log_route_request",
    "log_route_response",
    "log_training_step",
    "log_inference",
    "log_error",
    "JSONFormatter",
    "ColoredFormatter",
    "StructuredFormatter"
]

