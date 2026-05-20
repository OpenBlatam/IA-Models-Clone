"""
Exception handler middleware for custom exceptions.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

from ..exceptions import LovableException

logger = logging.getLogger(__name__)


async def lovable_exception_handler(request: Request, exc: LovableException) -> JSONResponse:
    """Handle Lovable custom exceptions."""
    logger.error(
        f"LovableException: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors."""
    errors = exc.errors()
    logger.warning(
        f"Validation error: {errors}",
        extra={"path": request.url.path}
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Request validation failed",
            "details": errors
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(
        f"HTTPException: {exc.detail}",
        extra={"status_code": exc.status_code, "path": request.url.path}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"path": request.url.path}
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "status_code": 500
        }
    )






