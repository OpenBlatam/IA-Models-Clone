"""
Error handlers for OpenRouter client.

Centralizes error handling logic to eliminate duplication.
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


def handle_openrouter_error(
    error: Exception,
    timeout: Optional[float] = None,
    operation_name: str = "OpenRouter API request"
) -> Exception:
    """
    Handle OpenRouter API errors consistently.
    
    Args:
        error: The exception that occurred
        timeout: Optional timeout value for timeout errors
        operation_name: Name of operation for error messages
        
    Returns:
        Exception with appropriate error message
        
    Examples:
        >>> try:
        ...     response = await client.post(...)
        ... except httpx.HTTPStatusError as e:
        ...     raise handle_openrouter_error(e, timeout=60.0)
        >>> except httpx.TimeoutException as e:
        ...     raise handle_openrouter_error(e, timeout=60.0)
    """
    if isinstance(error, httpx.HTTPStatusError):
        error_msg = f"OpenRouter API error: {error.response.status_code}"
        try:
            error_data = error.response.json()
            error_detail = error_data.get("error", {})
            error_msg = error_detail.get("message", error_msg)
        except Exception:
            pass
        
        logger.error(f"{operation_name} error: {error_msg}")
        return Exception(error_msg)
    
    elif isinstance(error, httpx.TimeoutException):
        timeout_msg = f" after {timeout}s" if timeout else ""
        logger.error(f"{operation_name} timeout{timeout_msg}")
        timeout_detail = f"Request timeout after {timeout}s" if timeout else "Request timeout"
        return Exception(timeout_detail)
    
    else:
        logger.error(f"{operation_name} error: {error}", exc_info=True)
        return error

