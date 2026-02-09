from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
import traceback
from typing import Any, Dict, Union
from datetime import datetime
from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from .http_exception_system import (
from .error_system import (
    import functools
    import asyncio
    from fastapi import FastAPI, APIRouter
from typing import Any, List, Dict, Optional
"""
🔧 HTTPException Handlers for FastAPI
====================================

FastAPI exception handlers that integrate with the HTTPException system
and provide consistent error responses across the application.
"""



    OnyxHTTPException, HTTPExceptionFactory, HTTPExceptionMapper,
    HTTPExceptionHandler, http_exception_handler
)
    OnyxBaseError, ValidationError, AuthenticationError, AuthorizationError,
    DatabaseError, CacheError, NetworkError, ExternalServiceError,
    ResourceNotFoundError, RateLimitError, TimeoutError,
    SerializationError, BusinessLogicError, SystemError
)

logger = logging.getLogger(__name__)


async async def onyx_http_exception_handler(request: Request, exc: OnyxHTTPException) -> JSONResponse:
    """
    Handler for OnyxHTTPException with detailed error information.
    """
    # Extract error details
    error_detail = exc.detail.get("error", {})
    
    # Log the error
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {error_detail.get('error_code', 'UNKNOWN')} - "
        f"{error_detail.get('message', 'Unknown error')} - "
        f"Path: {request.url.path} - Method: {request.method}"
    )
    
    # Return JSON response with error details
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
        headers=exc.headers
    )


async def onyx_base_error_handler(request: Request, exc: OnyxBaseError) -> JSONResponse:
    """
    Handler for OnyxBaseError that converts it to HTTP exception.
    """
    # Convert Onyx error to HTTP exception
    http_exception = HTTPExceptionMapper.map_onyx_error_to_http_exception(exc)
    
    # Log the error
    logger.error(
        f"Onyx Error: {exc.error_code} - {exc.message} - "
        f"Path: {request.url.path} - Method: {request.method}",
        exc_info=True
    )
    
    # Return JSON response
    return JSONResponse(
        status_code=http_exception.status_code,
        content=http_exception.detail,
        headers=http_exception.headers
    )


async async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handler for FastAPI request validation errors.
    """
    # Extract validation errors
    validation_errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        validation_errors.append(f"{field}: {message}")
    
    # Create HTTP exception
    http_exception = HTTPExceptionFactory.unprocessable_entity(
        message="Request validation failed",
        error_code="VALIDATION_ERROR",
        validation_errors=validation_errors,
        additional_data={
            "raw_errors": exc.errors(),
            "body": exc.body
        }
    )
    
    # Log the validation error
    logger.warning(
        f"Validation Error: {len(validation_errors)} validation errors - "
        f"Path: {request.url.path} - Method: {request.method}"
    )
    
    # Return JSON response
    return JSONResponse(
        status_code=http_exception.status_code,
        content=http_exception.detail,
        headers=http_exception.headers
    )


async def pydantic_validation_error_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """
    Handler for Pydantic validation errors.
    """
    # Extract validation errors
    validation_errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        validation_errors.append(f"{field}: {message}")
    
    # Create HTTP exception
    http_exception = HTTPExceptionFactory.unprocessable_entity(
        message="Data validation failed",
        error_code="PYDANTIC_VALIDATION_ERROR",
        validation_errors=validation_errors,
        additional_data={
            "raw_errors": exc.errors(),
            "model": exc.model.__name__ if hasattr(exc, 'model') else None
        }
    )
    
    # Log the validation error
    logger.warning(
        f"Pydantic Validation Error: {len(validation_errors)} validation errors - "
        f"Path: {request.url.path} - Method: {request.method}"
    )
    
    # Return JSON response
    return JSONResponse(
        status_code=http_exception.status_code,
        content=http_exception.detail,
        headers=http_exception.headers
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handler for unexpected exceptions.
    """
    # Convert to HTTP exception
    http_exception = HTTPExceptionMapper.map_exception_to_http_exception(exc)
    
    # Log the unexpected error
    logger.error(
        f"Unexpected Error: {type(exc).__name__} - {str(exc)} - "
        f"Path: {request.url.path} - Method: {request.method}",
        exc_info=True
    )
    
    # Return JSON response
    return JSONResponse(
        status_code=http_exception.status_code,
        content=http_exception.detail,
        headers=http_exception.headers
    )


class ExceptionHandlerRegistry:
    """
    Registry for managing exception handlers across the application.
    """
    
    def __init__(self) -> Any:
        self.handlers: Dict[type, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_handler(self, exception_type: type, handler: Any):
        """Register an exception handler"""
        self.handlers[exception_type] = handler
        self.logger.info(f"Registered handler for {exception_type.__name__}")
    
    def get_handler(self, exception_type: type) -> Optional[Dict[str, Any]]:
        """Get handler for exception type"""
        return self.handlers.get(exception_type)
    
    def register_default_handlers(self, app) -> Any:
        """Register default exception handlers for FastAPI app"""
        
        # Register OnyxHTTPException handler
        app.add_exception_handler(OnyxHTTPException, onyx_http_exception_handler)
        
        # Register OnyxBaseError handler
        app.add_exception_handler(OnyxBaseError, onyx_base_error_handler)
        
        # Register validation error handlers
        app.add_exception_handler(RequestValidationError, request_validation_error_handler)
        app.add_exception_handler(PydanticValidationError, pydantic_validation_error_handler)
        
        # Register general exception handler (should be last)
        app.add_exception_handler(Exception, general_exception_handler)
        
        self.logger.info("Registered default exception handlers")
    
    def register_custom_handlers(self, app) -> Any:
        """Register custom exception handlers for specific error types"""
        
        # Register handlers for specific Onyx error types
        app.add_exception_handler(ValidationError, onyx_base_error_handler)
        app.add_exception_handler(AuthenticationError, onyx_base_error_handler)
        app.add_exception_handler(AuthorizationError, onyx_base_error_handler)
        app.add_exception_handler(DatabaseError, onyx_base_error_handler)
        app.add_exception_handler(CacheError, onyx_base_error_handler)
        app.add_exception_handler(NetworkError, onyx_base_error_handler)
        app.add_exception_handler(ExternalServiceError, onyx_base_error_handler)
        app.add_exception_handler(ResourceNotFoundError, onyx_base_error_handler)
        app.add_exception_handler(RateLimitError, onyx_base_error_handler)
        app.add_exception_handler(TimeoutError, onyx_base_error_handler)
        app.add_exception_handler(SerializationError, onyx_base_error_handler)
        app.add_exception_handler(BusinessLogicError, onyx_base_error_handler)
        app.add_exception_handler(SystemError, onyx_base_error_handler)
        
        self.logger.info("Registered custom exception handlers")


# Global registry instance
exception_handler_registry = ExceptionHandlerRegistry()


def setup_exception_handlers(app) -> Any:
    """
    Setup all exception handlers for a FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Register default handlers
    exception_handler_registry.register_default_handlers(app)
    
    # Register custom handlers
    exception_handler_registry.register_custom_handlers(app)
    
    logger.info("Exception handlers setup completed")


# Decorator for automatic error handling
async def handle_http_exceptions(func) -> Any:
    """
    Decorator that automatically handles exceptions and converts them to HTTP exceptions.
    """
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except OnyxBaseError as e:
            # Convert Onyx error to HTTP exception
            http_exception = HTTPExceptionMapper.map_onyx_error_to_http_exception(e)
            raise http_exception
        except Exception as e:
            # Convert general exception to HTTP exception
            http_exception = HTTPExceptionMapper.map_exception_to_http_exception(e)
            raise http_exception
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except OnyxBaseError as e:
            # Convert Onyx error to HTTP exception
            http_exception = HTTPExceptionMapper.map_onyx_error_to_http_exception(e)
            raise http_exception
        except Exception as e:
            # Convert general exception to HTTP exception
            http_exception = HTTPExceptionMapper.map_exception_to_http_exception(e)
            raise http_exception
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Example FastAPI app with exception handlers
def create_app_with_exception_handlers():
    """
    Example of creating a FastAPI app with exception handlers.
    """
    
    app = FastAPI(title="Example App with Exception Handlers")
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Create router
    router = APIRouter()
    
    @router.get("/users/{user_id}")
    @handle_http_exceptions
    async def get_user(user_id: str):
        """Example endpoint with automatic error handling"""
        if user_id == "invalid":
            raise ValidationError(
                message="Invalid user ID format",
                field="user_id",
                value=user_id
            )
        
        if user_id == "not_found":
            raise ResourceNotFoundError(
                message="User not found",
                resource_type="user",
                resource_id=user_id
            )
        
        if user_id == "unauthorized":
            raise AuthenticationError(
                message="Invalid authentication token"
            )
        
        return {"user_id": user_id, "name": "John Doe"}
    
    @router.post("/users")
    @handle_http_exceptions
    async def create_user(user_data: dict):
        """Example endpoint with validation error handling"""
        if not user_data.get("email"):
            raise ValidationError(
                message="Email is required",
                field="email"
            )
        
        if user_data.get("email") == "duplicate@example.com":
            raise BusinessLogicError(
                message="User with this email already exists",
                business_rule="unique_email"
            )
        
        return {"user_id": "123", "email": user_data["email"]}
    
    app.include_router(router)
    return app


# Example usage
def example_usage():
    """Example of how to use the exception handlers"""
    
    # Create app with exception handlers
    app = create_app_with_exception_handlers()
    
    # Test different error scenarios
    test_cases = [
        ("/users/invalid", "Validation Error"),
        ("/users/not_found", "Not Found Error"),
        ("/users/unauthorized", "Authentication Error"),
    ]
    
    for path, description in test_cases:
        print(f"Testing {description} at {path}")
        # In a real scenario, you would make HTTP requests to test these endpoints


match __name__:
    case "__main__":
    example_usage() 