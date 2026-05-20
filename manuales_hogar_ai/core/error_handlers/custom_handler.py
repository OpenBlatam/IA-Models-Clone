"""
Custom Exception Handler
=======================

Manejador de excepciones personalizadas de la aplicación.
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse

from ..exceptions import (
    ManualesHogarAIException,
    ValidationError,
    RateLimitError,
    DatabaseError,
    CacheError,
    ExternalServiceError,
    CircuitBreakerOpenError,
)

logger = logging.getLogger(__name__)


async def custom_exception_handler(
    request: Request, exc: ManualesHogarAIException
) -> JSONResponse:
    """Handle custom application exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Map exception types to HTTP status codes
    status_code_map = {
        ValidationError: status.HTTP_400_BAD_REQUEST,
        RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
        DatabaseError: status.HTTP_503_SERVICE_UNAVAILABLE,
        CacheError: status.HTTP_503_SERVICE_UNAVAILABLE,
        ExternalServiceError: status.HTTP_502_BAD_GATEWAY,
        CircuitBreakerOpenError: status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    
    status_code = status_code_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    logger.error(
        f"Application error: {exc.error_code}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "error_code": exc.error_code,
            "message": exc.message,
        },
    )
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.message,
            "error_code": exc.error_code or type(exc).__name__,
            "request_id": request_id,
        },
    )

