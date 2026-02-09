"""
Enhanced error handling and validation utilities.
"""
import logging
from typing import Any, Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class APIError(HTTPException):
    """Base API error with structured response."""
    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.error_code = error_code or f"ERR_{status_code}"
        self.details = details or {}


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    logger.warning(
        "Validation error",
        extra={
            "path": str(request.url.path),
            "errors": errors,
            "request_id": request.headers.get("X-Request-ID"),
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "error_code": "ERR_422",
            "message": "Request validation failed",
            "details": {"errors": errors},
            "request_id": request.headers.get("X-Request-ID"),
        },
    )


async def api_exception_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API errors."""
    logger.error(
        "API error",
        extra={
            "path": str(request.url.path),
            "status_code": exc.status_code,
            "error_code": exc.error_code,
            "message": exc.detail,
            "request_id": request.headers.get("X-Request-ID"),
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "error_code": exc.error_code,
            "message": exc.detail,
            "details": exc.details,
            "request_id": request.headers.get("X-Request-ID"),
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    logger.exception(
        "Unhandled exception",
        extra={
            "path": str(request.url.path),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "request_id": request_id,
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "error_code": "ERR_500",
            "message": "An unexpected error occurred",
            "request_id": request_id,
        },
    )


