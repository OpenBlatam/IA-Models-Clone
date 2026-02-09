"""
Centralized exception handlers for Multi-Model API
Provides consistent error responses across all endpoints
"""

import logging
from typing import Union
from fastapi import Request, status
from fastapi.responses import JSONResponse

from .exceptions import (
    MultiModelAPIException,
    ModelExecutionException,
    RateLimitExceededException,
    CacheException,
    ValidationException,
    ModelNotFoundException,
    StrategyNotFoundException,
    TimeoutException
)

logger = logging.getLogger(__name__)


async def multi_model_exception_handler(
    request: Request,
    exc: MultiModelAPIException
) -> JSONResponse:
    """Handle Multi-Model API exceptions"""
    logger.error(
        f"Multi-Model API exception: {exc.__class__.__name__} - {exc.message}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "details": exc.details
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "path": request.url.path,
            "details": exc.details
        }
    )


async def rate_limit_exception_handler(
    request: Request,
    exc: RateLimitExceededException
) -> JSONResponse:
    """Handle rate limit exceptions"""
    logger.warning(
        f"Rate limit exceeded: {request.url.path}",
        extra={
            "retry_after": exc.retry_after,
            "limit": exc.limit
        }
    )
    
    headers = {
        "Retry-After": str(exc.retry_after)
    }
    
    if exc.limit is not None:
        headers["X-RateLimit-Limit"] = str(exc.limit)
        headers["X-RateLimit-Remaining"] = str(exc.remaining or 0)
        if exc.reset_at:
            headers["X-RateLimit-Reset"] = str(int(exc.reset_at))
    
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "RateLimitExceeded",
            "message": exc.message,
            "retry_after": exc.retry_after,
            "path": request.url.path
        },
        headers=headers
    )


async def validation_exception_handler(
    request: Request,
    exc: ValidationException
) -> JSONResponse:
    """Handle validation exceptions"""
    logger.warning(
        f"Validation error: {exc.message}",
        extra={
            "field": exc.field,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "ValidationError",
            "message": exc.message,
            "field": exc.field,
            "path": request.url.path,
            "details": exc.details
        }
    )


async def model_not_found_exception_handler(
    request: Request,
    exc: ModelNotFoundException
) -> JSONResponse:
    """Handle model not found exceptions"""
    logger.warning(
        f"Model not found: {exc.model_type}",
        extra={
            "model_type": exc.model_type,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "ModelNotFound",
            "message": exc.message,
            "model_type": exc.model_type,
            "path": request.url.path
        }
    )


async def timeout_exception_handler(
    request: Request,
    exc: TimeoutException
) -> JSONResponse:
    """Handle timeout exceptions"""
    logger.error(
        f"Request timeout: {exc.timeout}s",
        extra={
            "timeout": exc.timeout,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={
            "error": "Timeout",
            "message": exc.message,
            "timeout": exc.timeout,
            "path": request.url.path
        }
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions"""
    logger.exception(
        f"Unexpected error: {type(exc).__name__}",
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "path": request.url.path
        }
    )


def register_exception_handlers(app):
    """Register all exception handlers with FastAPI app"""
    app.add_exception_handler(MultiModelAPIException, multi_model_exception_handler)
    app.add_exception_handler(RateLimitExceededException, rate_limit_exception_handler)
    app.add_exception_handler(ValidationException, validation_exception_handler)
    app.add_exception_handler(ModelNotFoundException, model_not_found_exception_handler)
    app.add_exception_handler(TimeoutException, timeout_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)




