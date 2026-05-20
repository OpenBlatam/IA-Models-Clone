"""
Centralized Error Handlers
Global exception handlers for FastAPI application
"""

import os
import traceback
import time
from typing import Union
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.exceptions import APIException
from core.config import settings
from core.logging_config import get_logger

logger = get_logger(__name__)


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Handle custom API exceptions"""
    logger.warning(
        f"API Exception: {exc.error_type} - {exc.message}",
        extra={
            "error_type": exc.error_type,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )
    
    headers = {}
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": {
                "message": exc.detail if isinstance(exc.detail, str) else "HTTP error occurred",
                "status_code": exc.status_code,
                "type": "HTTPException",
                "detail": exc.detail if isinstance(exc.detail, dict) else None
            },
            "timestamp": time.time()
        },
        headers=headers
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors"""
    logger.warning(
        f"Validation Error: {exc.errors()}",
        extra={
            "path": request.url.path,
            "errors": exc.errors()
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "data": None,
            "error": {
                "message": "Validation error",
                "status_code": 422,
                "type": "ValidationError",
                "detail": {
                    "errors": exc.errors(),
                    "body": exc.body if hasattr(exc, 'body') else None
                }
            },
            "timestamp": time.time()
        }
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    # Don't expose internal errors in production
    environment = settings.environment.lower()
    dev_mode = environment == "development" or settings.debug
    
    error_message = "An unexpected error occurred"
    error_detail = None
    
    if dev_mode:
        error_message = str(exc)
        error_detail = {
            "type": exc.__class__.__name__,
            "traceback": traceback.format_exc().split("\n")[:-1] if traceback else None
        }
    
    logger.error(
        f"Unexpected error: {exc}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "data": None,
            "error": {
                "message": error_message,
                "status_code": 500,
                "type": "InternalServerError",
                "detail": error_detail
            },
            "timestamp": time.time()
        },
        headers={"Content-Type": "application/json"}
    )


def register_exception_handlers(app):
    """Register all exception handlers with FastAPI app"""
    # Custom API exceptions
    app.add_exception_handler(APIException, api_exception_handler)
    
    # HTTP exceptions
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Validation exceptions
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Global exception handler (catch-all)
    app.add_exception_handler(Exception, global_exception_handler)
    
    logger.info("Exception handlers registered")





