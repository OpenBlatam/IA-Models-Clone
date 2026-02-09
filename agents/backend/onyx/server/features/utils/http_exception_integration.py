from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import logging
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from functools import wraps
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from .http_exception_system import (
from .http_exception_handlers import (
from .http_response_models import (
from .error_system import (
        import asyncio
    from fastapi import APIRouter
import re
from pathlib import Path
from .http_exception_system import (
from typing import Any, List, Dict, Optional
"""
🔗 HTTPException Integration Guide
==================================

Integration module showing how to integrate the HTTPException system
with existing FastAPI applications, middleware, and external services.
"""



    OnyxHTTPException, HTTPExceptionFactory, HTTPExceptionMapper,
    HTTPExceptionHandler, http_exception_handler
)
    setup_exception_handlers, handle_http_exceptions
)
    SuccessResponse, ErrorResponse, ResponseFactory
)
    OnyxBaseError, ValidationError, AuthenticationError, AuthorizationError,
    ResourceNotFoundError, BusinessLogicError, DatabaseError,
    ErrorContext, ErrorFactory
)

logger = logging.getLogger(__name__)


class HTTPExceptionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling HTTP exceptions and providing consistent error responses.
    """
    
    def __init__(self, app, enable_logging: bool = True, enable_metrics: bool = True):
        
    """__init__ function."""
super().__init__(app)
        self.enable_logging = enable_logging
        self.enable_metrics = enable_metrics
        self.error_counts = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and handle exceptions"""
        start_time = datetime.utcnow()
        request_id = request.headers.get("X-Request-ID", str(datetime.utcnow().timestamp()))
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log successful requests
            if self.enable_logging:
                self._log_request(request, response, start_time, request_id)
            
            return response
            
        except OnyxHTTPException as e:
            # Handle Onyx HTTP exceptions
            return await self._handle_onyx_http_exception(request, e, start_time, request_id)
            
        except OnyxBaseError as e:
            # Convert Onyx errors to HTTP exceptions
            http_exception = HTTPExceptionMapper.map_onyx_error_to_http_exception(e)
            return await self._handle_onyx_http_exception(request, http_exception, start_time, request_id)
            
        except Exception as e:
            # Handle unexpected exceptions
            return await self._handle_unexpected_exception(request, e, start_time, request_id)
    
    async async def _handle_onyx_http_exception(
        self, 
        request: Request, 
        exc: OnyxHTTPException, 
        start_time: datetime,
        request_id: str
    ) -> JSONResponse:
        """Handle Onyx HTTP exceptions"""
        # Update metrics
        if self.enable_metrics:
            self._update_error_metrics(exc.status_code)
        
        # Log the error
        if self.enable_logging:
            self._log_error(request, exc, start_time, request_id)
        
        # Return JSON response
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers=exc.headers
        )
    
    async def _handle_unexpected_exception(
        self, 
        request: Request, 
        exc: Exception, 
        start_time: datetime,
        request_id: str
    ) -> JSONResponse:
        """Handle unexpected exceptions"""
        # Convert to HTTP exception
        http_exception = HTTPExceptionMapper.map_exception_to_http_exception(exc)
        
        # Update metrics
        if self.enable_metrics:
            self._update_error_metrics(http_exception.status_code)
        
        # Log the error
        if self.enable_logging:
            self._log_error(request, http_exception, start_time, request_id)
        
        # Return JSON response
        return JSONResponse(
            status_code=http_exception.status_code,
            content=http_exception.detail,
            headers=http_exception.headers
        )
    
    def _log_request(self, request: Request, response: Response, start_time: datetime, request_id: str):
        """Log successful request"""
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {duration:.3f}s - "
            f"Request ID: {request_id}"
        )
    
    def _log_error(self, request: Request, exc: OnyxHTTPException, start_time: datetime, request_id: str):
        """Log error request"""
        duration = (datetime.utcnow() - start_time).total_seconds()
        error_detail = exc.detail.get("error", {})
        
        logger.error(
            f"Request failed: {request.method} {request.url.path} - "
            f"Status: {exc.status_code} - Error: {error_detail.get('error_code', 'UNKNOWN')} - "
            f"Duration: {duration:.3f}s - Request ID: {request_id}"
        )
    
    def _update_error_metrics(self, status_code: int):
        """Update error metrics"""
        status_category = f"{status_code // 100}xx"
        self.error_counts[status_category] = self.error_counts.get(status_category, 0) + 1
    
    def get_error_metrics(self) -> Dict[str, int]:
        """Get error metrics"""
        return self.error_counts.copy()


class HTTPExceptionIntegration:
    """
    Integration helper for adding HTTPException handling to existing FastAPI applications.
    """
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
        self.middleware_added = False
        self.handlers_setup = False
    
    def setup_exception_handling(
        self,
        enable_middleware: bool = True,
        enable_logging: bool = True,
        enable_metrics: bool = True,
        custom_handlers: Optional[Dict[type, Callable]] = None
    ):
        """Setup complete exception handling for the application"""
        
        # Setup exception handlers
        if not self.handlers_setup:
            setup_exception_handlers(self.app)
            self.handlers_setup = True
        
        # Add custom handlers if provided
        if custom_handlers:
            for exception_type, handler in custom_handlers.items():
                self.app.add_exception_handler(exception_type, handler)
        
        # Add middleware
        if enable_middleware and not self.middleware_added:
            self.app.add_middleware(
                HTTPExceptionMiddleware,
                enable_logging=enable_logging,
                enable_metrics=enable_metrics
            )
            self.middleware_added = True
        
        logger.info("HTTPException handling setup completed")
    
    def add_custom_error_handler(self, exception_type: type, handler: Callable):
        """Add custom error handler"""
        self.app.add_exception_handler(exception_type, handler)
        logger.info(f"Added custom handler for {exception_type.__name__}")
    
    def create_error_monitoring_endpoint(self) -> Any:
        """Create endpoint for monitoring error metrics"""
        
        @self.app.get("/health/errors")
        async def get_error_metrics():
            """Get error metrics"""
            middleware = None
            for m in self.app.user_middleware:
                if isinstance(m.cls, HTTPExceptionMiddleware):
                    middleware = m.cls
                    break
            
            if middleware and hasattr(middleware, 'get_error_metrics'):
                metrics = middleware.get_error_metrics()
            else:
                metrics = {}
            
            return {
                "error_metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            }


class ErrorHandlingDecorator:
    """
    Decorator for automatic error handling in service functions.
    """
    
    def __init__(self, default_status_code: int = 500):
        
    """__init__ function."""
self.default_status_code = default_status_code
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except OnyxBaseError as e:
                http_exception = HTTPExceptionMapper.map_onyx_error_to_http_exception(e)
                raise http_exception
            except Exception as e:
                http_exception = HTTPExceptionFactory.internal_server_error(
                    message=str(e),
                    error_code="UNEXPECTED_ERROR"
                )
                raise http_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except OnyxBaseError as e:
                http_exception = HTTPExceptionMapper.map_onyx_error_to_http_exception(e)
                raise http_exception
            except Exception as e:
                http_exception = HTTPExceptionFactory.internal_server_error(
                    message=str(e),
                    error_code="UNEXPECTED_ERROR"
                )
                raise http_exception
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# Integration examples
def integrate_with_existing_app(app: FastAPI) -> HTTPExceptionIntegration:
    """
    Integrate HTTPException handling with an existing FastAPI application.
    
    Args:
        app: Existing FastAPI application
        
    Returns:
        HTTPExceptionIntegration instance
    """
    integration = HTTPExceptionIntegration(app)
    
    # Setup basic exception handling
    integration.setup_exception_handling(
        enable_middleware=True,
        enable_logging=True,
        enable_metrics=True
    )
    
    # Add error monitoring endpoint
    integration.create_error_monitoring_endpoint()
    
    return integration


def create_integration_example():
    """
    Create an example showing integration with existing applications.
    """
    
    # Create existing app
    app = FastAPI(title="Integration Example")
    
    # Create integration
    integration = integrate_with_existing_app(app)
    
    # Create router with existing endpoints
    router = APIRouter()
    
    # Example existing endpoint without error handling
    @router.get("/legacy/users/{user_id}")
    async def get_legacy_user(user_id: str):
        """Legacy endpoint without proper error handling"""
        if user_id == "invalid":
            # This will be caught by the middleware and converted to proper HTTP exception
            raise ValueError("Invalid user ID")
        
        if user_id == "not_found":
            # This will be caught and converted to 404
            raise ResourceNotFoundError(
                message="User not found",
                resource_type="user",
                resource_id=user_id
            )
        
        return {"user_id": user_id, "name": "Legacy User"}
    
    # Example endpoint with custom error handling
    @router.get("/custom/users/{user_id}")
    async def get_custom_user(user_id: str):
        """Endpoint with custom error handling"""
        try:
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
            
            return {"user_id": user_id, "name": "Custom User"}
            
        except OnyxBaseError as e:
            # Convert to HTTP exception manually
            http_exception = HTTPExceptionMapper.map_onyx_error_to_http_exception(e)
            raise http_exception
    
    # Example service function with decorator
    @ErrorHandlingDecorator()
    async def service_function(user_id: str):
        """Service function with automatic error handling"""
        if user_id == "error":
            raise DatabaseError(
                message="Database connection failed",
                operation="query"
            )
        
        return {"result": "success"}
    
    @router.get("/service/{user_id}")
    async def get_service_result(user_id: str):
        """Endpoint using service function with error handling"""
        result = await service_function(user_id)
        return result
    
    # Include router
    app.include_router(router)
    
    return app


# Migration guide functions
def migrate_existing_http_exceptions(app: FastAPI):
    """
    Migrate existing HTTPException usage to the new system.
    
    Args:
        app: FastAPI application to migrate
    """
    
    # Example migration patterns
    migration_examples = {
        "old_pattern": """
        # Old pattern
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        """,
        
        "new_pattern": """
        # New pattern
        if not user:
            raise_not_found(
                message="User not found",
                resource_type="user",
                resource_id=user_id
            )
        """,
        
        "old_validation": """
        # Old validation pattern
        if not email or '@' not in email:
            raise HTTPException(status_code=400, detail="Invalid email")
        """,
        
        "new_validation": """
        # New validation pattern
        if not email or '@' not in email:
            raise_bad_request(
                message="Invalid email format",
                error_code="INVALID_EMAIL",
                field="email",
                value=email
            )
        """,
        
        "old_auth": """
        # Old authentication pattern
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        """,
        
        "new_auth": """
        # New authentication pattern
        if not token:
            raise_unauthorized(
                message="Authentication required",
                error_code="MISSING_TOKEN"
            )
        """
    }
    
    logger.info("Migration examples provided. Update your code to use the new patterns.")
    return migration_examples


def create_migration_script():
    """
    Create a script to help migrate existing code.
    """
    
    migration_script = """
# Migration Script for HTTPException System
# =========================================


def migrate_file(file_path: str):
    
    """migrate_file function."""
'''Migrate a single file to use the new HTTPException system'''
    
    with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    # Migration patterns
    patterns = [
        # HTTPException with 404
        (
            r'raise HTTPException\(status_code=404, detail="([^"]+)"\)',
            r'raise_not_found(message="\\1")'
        ),
        
        # HTTPException with 400
        (
            r'raise HTTPException\(status_code=400, detail="([^"]+)"\)',
            r'raise_bad_request(message="\\1")'
        ),
        
        # HTTPException with 401
        (
            r'raise HTTPException\(status_code=401, detail="([^"]+)"\)',
            r'raise_unauthorized(message="\\1")'
        ),
        
        # HTTPException with 403
        (
            r'raise HTTPException\(status_code=403, detail="([^"]+)"\)',
            r'raise_forbidden(message="\\1")'
        ),
        
        # HTTPException with 409
        (
            r'raise HTTPException\(status_code=409, detail="([^"]+)"\)',
            r'raise_conflict(message="\\1")'
        ),
        
        # HTTPException with 422
        (
            r'raise HTTPException\(status_code=422, detail="([^"]+)"\)',
            r'raise_unprocessable_entity(message="\\1")'
        ),
        
        # HTTPException with 429
        (
            r'raise HTTPException\(status_code=429, detail="([^"]+)"\)',
            r'raise_too_many_requests(message="\\1")'
        ),
        
        # HTTPException with 500
        (
            r'raise HTTPException\(status_code=500, detail="([^"]+)"\)',
            r'raise_internal_server_error(message="\\1")'
        ),
        
        # HTTPException with 503
        (
            r'raise HTTPException\(status_code=503, detail="([^"]+)"\)',
            r'raise_service_unavailable(message="\\1")'
        ),
    ]
    
    # Apply patterns
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Add imports if needed
    if 'raise_not_found' in content and 'from .http_exception_system import' not in content:
        import_statement = '''
    raise_bad_request, raise_unauthorized, raise_forbidden,
    raise_not_found, raise_conflict, raise_unprocessable_entity,
    raise_too_many_requests, raise_internal_server_error,
    raise_service_unavailable
)
'''
        # Find the right place to add imports
        lines = content.split('\\n')
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                continue
            if line.strip() == '':
                continue
            lines.insert(i, import_statement.strip())
            break
        
        content = '\\n'.join(lines)
    
    # Write back to file
    with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Migrated {file_path}")

def migrate_directory(directory: str):
    
    """migrate_directory function."""
'''Migrate all Python files in a directory'''
    
    for file_path in Path(directory).rglob("*.py"):
        if "migration" not in str(file_path):
            try:
                migrate_file(str(file_path))
            except Exception as e:
                print(f"Error migrating {file_path}: {e}")

# Usage
# migrate_directory("./your_project_directory")
"""
    
    return migration_script


# Example usage
def example_integration():
    """Example of integration with existing applications"""
    
    # Create integration example
    app = create_integration_example()
    
    # Create migration guide
    migration_examples = migrate_existing_http_exceptions(app)
    
    # Create migration script
    migration_script = create_migration_script()
    
    print("HTTPException Integration Example")
    print("=" * 50)
    print("1. Integration with existing apps completed")
    print("2. Migration examples provided")
    print("3. Migration script created")
    print("\nTo use the integration:")
    print("- Call integrate_with_existing_app(your_app)")
    print("- Use the migration script to update existing code")
    print("- Add @ErrorHandlingDecorator() to service functions")


match __name__:
    case "__main__":
    example_integration() 