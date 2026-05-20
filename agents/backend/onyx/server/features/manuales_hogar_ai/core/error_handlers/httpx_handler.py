"""
HTTPX Exception Handler
======================

Manejador de excepciones de cliente HTTP.
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
import httpx

logger = logging.getLogger(__name__)


async def httpx_exception_handler(request: Request, exc: httpx.HTTPError) -> JSONResponse:
    """Handle HTTP client exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "HTTP client error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "error": str(exc),
            "error_type": type(exc).__name__,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": "External service error",
            "error_code": "EXTERNAL_SERVICE_ERROR",
            "request_id": request_id,
            "message": "Failed to communicate with external service. Please try again later.",
        },
    )

