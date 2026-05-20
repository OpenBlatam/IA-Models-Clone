"""
Exception Handlers
Custom exception handlers for the FastAPI application
"""

import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse

from .models import ErrorResponse
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

def create_error_response(error: str, status_code: int) -> dict:
    """Create a standardized error response"""
    return ErrorResponse(
        error=error,
        status_code=status_code,
        timestamp=datetime.now(timezone.utc)
    ).model_dump()

def setup_exception_handlers(app: FastAPI) -> None:
    """Configure exception handlers for the application"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> ORJSONResponse:
        """Handle HTTP exceptions"""
        logger.warning(
            "HTTP exception",
            extra={
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": request.url.path,
                "method": request.method
            }
        )
        return ORJSONResponse(
            status_code=exc.status_code,
            content=create_error_response(exc.detail, exc.status_code)
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> ORJSONResponse:
        """Handle general exceptions"""
        settings = get_settings()
        error_detail = str(exc) if settings.debug else "Internal server error"
        
        logger.error(
            "Unhandled exception",
            extra={
                "path": request.url.path,
                "method": request.method,
                "error_type": type(exc).__name__
            },
            exc_info=True
        )
        
        return ORJSONResponse(
            status_code=500,
            content=create_error_response(error_detail, 500)
        )
    
    logger.info("Exception handlers configured")
