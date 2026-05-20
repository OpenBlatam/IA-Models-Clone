"""
Global Exception Handler
=======================

Manejador global de excepciones no capturadas.
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse

from ...core.base.service_base import BaseService

logger = logging.getLogger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "Unhandled exception",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
            "error_type": type(exc).__name__,
        },
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "request_id": request_id,
            "message": "An unexpected error occurred. Please try again later.",
        },
    )

