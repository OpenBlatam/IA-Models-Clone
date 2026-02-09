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
    operation: str = "OpenRouter operation",
    timeout: Optional[float] = None
) -> Exception:
    """
    Handle OpenRouter API errors consistently.
    
    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        timeout: Optional timeout value for timeout errors
        
    Returns:
        Exception with appropriate error message
        
    Examples:
        >>> try:
        ...     response = await client.post(...)
        ... except httpx.HTTPStatusError as e:
        ...     raise handle_openrouter_error(e, "extract content")
        >>> except httpx.TimeoutException as e:
        ...     raise handle_openrouter_error(e, "extract content", timeout=60.0)
    """
    if isinstance(error, httpx.HTTPStatusError):
        error_msg = f"Error de API OpenRouter: {error.response.status_code}"
        try:
            error_data = error.response.json()
            error_detail = error_data.get("error", {})
            error_msg = error_detail.get("message", error_msg)
        except Exception:
            pass
        
        logger.error(f"Error OpenRouter ({error.response.status_code}): {error_msg}")
        return Exception(f"Error al procesar con OpenRouter: {error_msg}")
    
    elif isinstance(error, httpx.TimeoutException):
        timeout_msg = f"después de {timeout}s" if timeout else ""
        logger.error(f"Timeout al llamar OpenRouter {timeout_msg}")
        return Exception(f"Timeout al procesar con OpenRouter")
    
    else:
        logger.error(f"Error llamando OpenRouter: {error}", exc_info=True)
        return error

