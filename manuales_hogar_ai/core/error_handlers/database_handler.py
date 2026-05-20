"""
Database Exception Handler
=========================

Manejador de excepciones de base de datos.
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


async def database_exception_handler(
    request: Request, exc: SQLAlchemyError
) -> JSONResponse:
    """Handle database exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "Database error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "error": str(exc),
            "error_type": type(exc).__name__,
        },
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Database error",
            "error_code": "DATABASE_ERROR",
            "request_id": request_id,
            "message": "Database operation failed. Please try again later.",
        },
    )

