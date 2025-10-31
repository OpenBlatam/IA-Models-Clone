from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Any, List, Dict, Optional
import asyncio
"""
Exception Handlers
=================

Custom exception handlers for the FastAPI application.
"""


logger = logging.getLogger(__name__)


def setup_exception_handlers(app: FastAPI):
    """Setup custom exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "error_code": f"HTTP_{exc.status_code}",
                "timestamp": time.time(),
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation exceptions"""
        logger.warning(f"Validation Error: {exc.errors()}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "error_code": "VALIDATION_ERROR",
                "details": exc.errors(),
                "timestamp": time.time(),
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": time.time(),
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions"""
        logger.warning(f"Value Error: {exc}")
        
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid value",
                "error_code": "VALUE_ERROR",
                "message": str(exc),
                "timestamp": time.time(),
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(request: Request, exc: TimeoutError):
        """Handle timeout exceptions"""
        logger.warning(f"Timeout Error: {exc}")
        
        return JSONResponse(
            status_code=408,
            content={
                "error": "Request timeout",
                "error_code": "TIMEOUT_ERROR",
                "message": "Request timed out",
                "timestamp": time.time(),
                "path": str(request.url)
            }
        ) 