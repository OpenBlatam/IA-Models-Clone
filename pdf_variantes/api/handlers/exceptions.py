"""
PDF Variantes API - Exception Handlers
Centralized exception handling with consistent responses
"""

import os
import traceback
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from ..exceptions import BaseAPIException, format_error_response
from ...utils.response_helpers import (
    get_request_id,
    create_error_response,
    create_not_found_response,
    create_unauthorized_response
)
from ...utils.structured_logging import get_logger

logger = get_logger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup all exception handlers for the application"""
    
    @app.exception_handler(BaseAPIException)
    async def api_exception_handler(request: Request, exc: BaseAPIException):
        """Handle custom API exceptions with consistent format"""
        logger.error(f"API Exception: {exc.error_code} - {exc.detail}")
        
        headers = _build_error_headers(exc)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=format_error_response(exc),
            headers=headers
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent frontend-friendly format"""
        request_id = get_request_id(request)
        
        logger.error(
            f"[{request_id}] HTTP Exception: {exc.status_code} - {exc.detail}"
        )
        
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": request_id
        }
        
        error_response = _build_http_error_response(exc, request_id, headers)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
            headers=headers
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions with consistent frontend-friendly format"""
        request_id = get_request_id(request)
        
        logger.error(
            f"[{request_id}] Unhandled Exception: {str(exc)}",
            exc_info=True
        )
        
        error_response = _build_general_error_response(exc, request_id)
        
        return JSONResponse(
            status_code=500,
            content=error_response,
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": request_id
            }
        )


def _build_error_headers(exc: BaseAPIException) -> Dict[str, str]:
    """Build headers for error responses"""
    headers = {"Content-Type": "application/json"}
    
    if exc.status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    elif exc.status_code == 429:
        if "retry_after" in exc.metadata:
            headers["Retry-After"] = str(exc.metadata["retry_after"])
    
    return headers


def _build_http_error_response(
    exc: HTTPException,
    request_id: str,
    headers: Dict[str, str]
) -> Dict[str, Any]:
    """Build error response for HTTP exceptions"""
    if exc.status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
        return create_unauthorized_response(
            message=str(exc.detail),
            request_id=request_id
        )
    
    if exc.status_code == 404:
        error_response = create_not_found_response(
            resource="Resource",
            request_id=request_id
        )
        error_response["error"]["message"] = str(exc.detail)
        return error_response
    
    return create_error_response(
        message=str(exc.detail),
        status_code=exc.status_code,
        error_type="HTTPException",
        request_id=request_id
    )


def _build_general_error_response(
    exc: Exception,
    request_id: str
) -> Dict[str, Any]:
    """Build error response for general exceptions"""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    dev_mode = environment == "development" or os.getenv("DEBUG", "false").lower() == "true"
    
    error_message = "An unexpected error occurred"
    error_detail = None
    
    if dev_mode:
        error_message = str(exc)
        error_detail = {
            "type": exc.__class__.__name__,
            "traceback": traceback.format_exc().split("\n")[:-1] if traceback else None
        }
    
    return create_error_response(
        message=error_message,
        status_code=500,
        error_type="InternalServerError",
        details=error_detail,
        request_id=request_id
    )






