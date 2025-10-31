"""
Exception Handlers - Frontend-friendly error responses
"""

import logging
import traceback
import time
from typing import Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os

from ..core.config import get_settings
from ..core.logging_config import get_logger

logger = get_logger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions with frontend-friendly format"""
        logger.warning(
            f"HTTP {exc.status_code}: {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method
            }
        )
        
        headers = {"Content-Type": "application/json"}
        if exc.status_code == 401:
            headers["WWW-Authenticate"] = "Bearer"
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "data": None,
                "error": {
                    "message": exc.detail,
                    "status_code": exc.status_code,
                    "type": "HTTPException"
                },
                "timestamp": time.time()
            },
            headers=headers
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions"""
        settings = get_settings()
        dev_mode = settings.debug or settings.environment == "development"
        
        error_message = "An unexpected error occurred"
        error_detail: Any = None
        
        if dev_mode:
            error_message = str(exc)
            error_detail = {
                "type": exc.__class__.__name__,
                "traceback": traceback.format_exc().split("\n")[:-1]
            }
        
        logger.error(
            f"Unhandled exception: {exc}",
            exc_info=True,
            extra={
                "path": request.url.path,
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=500,
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






