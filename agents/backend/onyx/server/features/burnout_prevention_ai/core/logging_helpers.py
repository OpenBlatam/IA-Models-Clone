"""
Logging Helpers
===============
Helper functions for consistent logging across the application.

Centralizes logging operations to ensure consistent formatting,
error message truncation, and context handling throughout the app.
"""

from typing import Any, Dict, Optional
from .constants import MAX_ERROR_MESSAGE_LENGTH

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def log_error(
    message: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    exc_info: bool = True
) -> None:
    """
    Log error with consistent formatting.
    
    Args:
        message: Error message
        error: Exception object
        context: Optional context dictionary
        exc_info: Whether to include exception info
    """
    error_msg = truncate_error_message(error)
    log_data = {
        "error": error_msg,
        "error_type": type(error).__name__
    }
    if context:
        log_data.update(context)
    
    logger.error(message, **log_data, exc_info=exc_info)


def truncate_error_message(error: Exception, max_length: Optional[int] = None) -> str:
    """
    Truncate error message to max length.
    
    Args:
        error: Exception to truncate
        max_length: Maximum length (defaults to MAX_ERROR_MESSAGE_LENGTH)
        
    Returns:
        Truncated error message
    """
    if max_length is None:
        max_length = MAX_ERROR_MESSAGE_LENGTH
    return str(error)[:max_length]


def log_warning(
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log warning with consistent formatting.
    
    Args:
        message: Warning message
        context: Optional context dictionary
    """
    if context:
        logger.warning(message, **context)
    else:
        logger.warning(message)


def log_info(
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log info with consistent formatting.
    
    Args:
        message: Info message
        context: Optional context dictionary
    """
    if context:
        logger.info(message, **context)
    else:
        logger.info(message)


def log_debug(
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log debug with consistent formatting.
    
    Args:
        message: Debug message
        context: Optional context dictionary
    """
    if context:
        logger.debug(message, **context)
    else:
        logger.debug(message)

