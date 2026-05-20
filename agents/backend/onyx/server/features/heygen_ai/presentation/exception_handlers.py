from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import structlog
from ..domain.exceptions.domain_errors import (
import time 
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Exception Handlers

FastAPI exception handlers for converting domain exceptions to HTTP responses.
"""


    DomainError,
    UserValidationError,
    VideoValidationError,
    BusinessRuleViolationError,
    DomainNotFoundException,
    DomainConflictError,
    DomainForbiddenError,
    ValueObjectValidationError
)

logger = structlog.get_logger()


def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: Any = None,
    request_id: str = None
) -> JSONResponse:
    """Create standardized error response."""
    content = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": str(int(time.time())),
            "request_id": request_id
        }
    }
    
    if details:
        content["error"]["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )


async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    """Handle domain errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        "Domain error occurred",
        request_id=request_id,
        error_code=exc.error_code,
        message=exc.message,
        context=exc.context
    )
    
    # Map domain errors to HTTP status codes
    status_code_map = {
        "USER_VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "VIDEO_VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "VALUE_OBJECT_VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "BUSINESS_RULE_VIOLATION": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "DOMAIN_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "DOMAIN_CONFLICT": status.HTTP_409_CONFLICT,
        "DOMAIN_FORBIDDEN": status.HTTP_403_FORBIDDEN,
    }
    
    status_code = status_code_map.get(exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return create_error_response(
        status_code=status_code,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.context if exc.context else None,
        request_id=request_id
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        "Validation error occurred",
        request_id=request_id,
        errors=exc.errors()
    )
    
    # Format validation errors
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"validation_errors": errors},
        request_id=request_id
    )


async async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        "HTTP exception occurred",
        request_id=request_id,
        status_code=exc.status_code,
        detail=exc.detail
    )
    
    return create_error_response(
        status_code=exc.status_code,
        error_code=f"HTTP_{exc.status_code}",
        message=exc.detail if isinstance(exc.detail, str) else "HTTP error occurred",
        details=exc.detail if not isinstance(exc.detail, str) else None,
        request_id=request_id
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "Unexpected error occurred",
        request_id=request_id,
        error=str(exc),
        error_type=type(exc).__name__,
        exc_info=True
    )
    
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred. Please try again later.",
        request_id=request_id
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Setup exception handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    logger.info("Setting up exception handlers")
    
    # Domain exception handlers
    app.add_exception_handler(DomainError, domain_error_handler)
    app.add_exception_handler(UserValidationError, domain_error_handler)
    app.add_exception_handler(VideoValidationError, domain_error_handler)
    app.add_exception_handler(BusinessRuleViolationError, domain_error_handler)
    app.add_exception_handler(DomainNotFoundException, domain_error_handler)
    app.add_exception_handler(DomainConflictError, domain_error_handler)
    app.add_exception_handler(DomainForbiddenError, domain_error_handler)
    app.add_exception_handler(ValueObjectValidationError, domain_error_handler)
    
    # FastAPI exception handlers
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    
    # General exception handler (catch-all)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Exception handlers setup completed")


# Add missing import