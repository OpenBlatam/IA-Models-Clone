"""
Error handling utilities

This module provides exception handlers for FastAPI application,
ensuring consistent error response formatting across all endpoints.
"""

from typing import Dict, Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import traceback

from utils.exceptions import LogisticsException

logger = logging.getLogger(__name__)


async def logistics_exception_handler(
    request: Request,
    exc: LogisticsException
) -> JSONResponse:
    """
    Handle custom logistics exceptions
    
    Args:
        request: FastAPI request object
        exc: Logistics exception to handle
        
    Returns:
        JSONResponse with error details
    """
    # Use the exception's to_dict method if available
    if hasattr(exc, 'to_dict'):
        error_content = exc.to_dict()
    else:
        error_content = {
            "error": {
                "code": exc.error_code or "LOGISTICS_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        }
    
    # Add request path
    error_content["error"]["path"] = str(request.url.path)
    error_content["error"]["method"] = request.method
    
    logger.warning(
        f"Logistics exception: {exc.error_code} - {exc.detail} "
        f"at {request.method} {request.url.path}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_content,
        headers=exc.headers if hasattr(exc, 'headers') and exc.headers else None
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle validation errors from Pydantic
    
    Args:
        request: FastAPI request object
        exc: Validation error exception
        
    Returns:
        JSONResponse with validation error details
    """
    errors = []
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    error_content = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "errors": errors,
            "path": str(request.url.path),
            "method": request.method
        }
    }
    
    logger.warning(
        f"Validation error at {request.method} {request.url.path}: "
        f"{len(errors)} field(s) failed validation"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_content
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle general unhandled exceptions
    
    Args:
        request: FastAPI request object
        exc: Exception to handle
        
    Returns:
        JSONResponse with generic error message
    """
    # Log full exception details
    logger.error(
        f"Unhandled exception at {request.method} {request.url.path}: {exc}",
        exc_info=True,
        extra={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "path": str(request.url.path),
            "method": request.method,
            "traceback": traceback.format_exc()
        }
    )
    
    # In production, don't expose internal error details
    error_content: Dict[str, Any] = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An internal server error occurred",
            "path": str(request.url.path),
            "method": request.method
        }
    }
    
    # In development, include more details
    import os
    if os.getenv("ENVIRONMENT", "production") == "development":
        error_content["error"]["details"] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_content
    )

