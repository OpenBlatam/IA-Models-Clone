from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, APIRouter, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from .enhanced_middleware_system import (
from .middleware_monitoring import (
from .http_exception_system import (
from .http_exception_handlers import handle_http_exceptions
from .error_system import (
        import redis.asyncio as redis
        import random
        import random
        import redis.asyncio as redis
        import random
from typing import Any, List, Dict, Optional
"""
🔗 Middleware Integration Examples
==================================

Comprehensive examples showing how to integrate the enhanced middleware system
with FastAPI applications for error handling, logging, and monitoring.
"""



# Import our middleware systems
    EnhancedMiddlewareManager, EnhancedMiddlewareConfig,
    create_enhanced_middleware_config, create_production_enhanced_config,
    create_development_enhanced_config, setup_enhanced_middleware
)
    MonitoringManager, MonitoringConfig,
    create_monitoring_config, create_production_monitoring_config,
    create_development_monitoring_config, setup_monitoring
)
    HTTPExceptionFactory, raise_bad_request, raise_not_found,
    raise_unauthorized, raise_forbidden, raise_internal_server_error
)
    ValidationError, AuthenticationError, AuthorizationError,
    ResourceNotFoundError, BusinessLogicError, DatabaseError,
    ErrorFactory
)

logger = logging.getLogger(__name__)

# =============================================================================
# EXAMPLE 1: BASIC MIDDLEWARE INTEGRATION
# =============================================================================

def create_basic_middleware_app() -> FastAPI:
    """
    Example 1: Basic middleware integration with error handling and logging.
    """
    
    # Create FastAPI app
    app = FastAPI(
        title="Basic Middleware Example",
        description="Example with basic middleware integration",
        version="1.0.0"
    )
    
    # Create middleware configuration
    config = create_enhanced_middleware_config(
        environment="development",
        logging_enabled=True,
        error_handling_enabled=True,
        monitoring_enabled=True
    )
    
    # Setup middleware
    middleware_manager = setup_enhanced_middleware(app, config)
    
    # Create router
    router = APIRouter()
    
    @router.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "Basic middleware example", "timestamp": datetime.utcnow().isoformat()}
    
    @router.get("/users/{user_id}")
    @handle_http_exceptions
    async def get_user(user_id: str):
        """Get user by ID with automatic error handling."""
        if not user_id.isalnum():
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
        
        if user_id == "error":
            raise DatabaseError(
                message="Database connection failed",
                operation="query"
            )
        
        return {"user_id": user_id, "name": "John Doe", "email": "john@example.com"}
    
    @router.post("/users")
    @handle_http_exceptions
    async def create_user(user_data: dict):
        """Create user with validation."""
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
        
        return {"user_id": "123", "email": user_data["email"], "status": "created"}
    
    @router.get("/test/error")
    async def test_error():
        """Test endpoint that raises an error."""
        raise ValueError("This is a test error for middleware")
    
    @router.get("/test/slow")
    async def test_slow():
        """Test endpoint that simulates a slow request."""
        await asyncio.sleep(2)  # Simulate slow processing
        return {"message": "Slow request completed"}
    
    # Add monitoring endpoints
    @router.get("/health")
    async def health():
        """Health check endpoint."""
        return middleware_manager.get_health_status()
    
    @router.get("/metrics")
    async def metrics():
        """Metrics endpoint."""
        return middleware_manager.get_metrics()
    
    @router.get("/performance")
    async def performance():
        """Performance summary endpoint."""
        return middleware_manager.get_performance_summary()
    
    @router.get("/errors")
    async def errors():
        """Error summary endpoint."""
        return middleware_manager.get_error_summary()
    
    # Include router
    app.include_router(router)
    
    return app

# =============================================================================
# EXAMPLE 2: PRODUCTION MIDDLEWARE INTEGRATION
# =============================================================================

async def create_production_middleware_app() -> FastAPI:
    """
    Example 2: Production middleware integration with Redis and comprehensive monitoring.
    """
    
    # Create FastAPI app
    app = FastAPI(
        title="Production Middleware Example",
        description="Example with production-optimized middleware",
        version="1.0.0"
    )
    
    # Create production middleware configuration
    middleware_config = create_production_enhanced_config()
    
    # Create production monitoring configuration
    monitoring_config = create_production_monitoring_config()
    
    # Setup Redis (optional)
    redis_client = None
    try:
        redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        middleware_config.redis_enabled = False
        monitoring_config.redis_enabled = False
    
    # Setup middleware
    middleware_manager = setup_enhanced_middleware(app, middleware_config, redis_client)
    
    # Setup monitoring
    monitoring_manager = await setup_monitoring(app, monitoring_config, redis_client)
    
    # Create router
    router = APIRouter()
    
    @router.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Production middleware example",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": "production"
        }
    
    @router.get("/api/users/{user_id}")
    @handle_http_exceptions
    async def get_user(user_id: str, request: Request):
        """Get user with comprehensive error handling."""
        # Simulate authentication check
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise AuthenticationError(
                message="Authentication required",
                auth_method="bearer"
            )
        
        # Simulate authorization check
        if user_id == "admin" and auth_header != "Bearer admin_token":
            raise AuthorizationError(
                message="Admin access required",
                required_permission="admin"
            )
        
        # Simulate validation
        if not user_id.isalnum():
            raise_bad_request(
                message="Invalid user ID format",
                field="user_id",
                value=user_id
            )
        
        # Simulate not found
        if user_id == "not_found":
            raise_not_found(
                message="User not found",
                resource_type="user",
                resource_id=user_id
            )
        
        # Simulate database error
        if user_id == "db_error":
            raise DatabaseError(
                message="Database connection failed",
                operation="query",
                table="users"
            )
        
        return {
            "user_id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "created_at": datetime.utcnow().isoformat()
        }
    
    @router.post("/api/users")
    @handle_http_exceptions
    async def create_user(user_data: dict):
        """Create user with validation and business logic."""
        # Validate required fields
        if not user_data.get("email"):
            raise ValidationError(
                message="Email is required",
                field="email"
            )
        
        if not user_data.get("name"):
            raise ValidationError(
                message="Name is required",
                field="name"
            )
        
        # Validate email format
        if "@" not in user_data["email"]:
            raise ValidationError(
                message="Invalid email format",
                field="email",
                value=user_data["email"]
            )
        
        # Check business rules
        if user_data["email"] == "duplicate@example.com":
            raise BusinessLogicError(
                message="User with this email already exists",
                business_rule="unique_email"
            )
        
        if len(user_data.get("name", "")) < 2:
            raise BusinessLogicError(
                message="Name must be at least 2 characters long",
                business_rule="name_length"
            )
        
        return {
            "user_id": "123",
            "email": user_data["email"],
            "name": user_data["name"],
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }
    
    @router.get("/api/test/performance")
    async def test_performance():
        """Test endpoint for performance monitoring."""
        start_time = time.time()
        
        # Simulate some processing
        await asyncio.sleep(0.1)
        
        # Simulate database query
        await asyncio.sleep(0.2)
        
        # Simulate external API call
        await asyncio.sleep(0.3)
        
        duration = time.time() - start_time
        
        return {
            "message": "Performance test completed",
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.get("/api/test/errors")
    async def test_errors():
        """Test endpoint for error monitoring."""
        
        error_types = [
            ValueError("Random validation error"),
            TypeError("Random type error"),
            ConnectionError("Random connection error"),
            PermissionError("Random permission error"),
            RuntimeError("Random runtime error")
        ]
        
        # Randomly raise an error
        if random.random() < 0.3:  # 30% chance of error
            raise random.choice(error_types)
        
        return {"message": "No error this time", "timestamp": datetime.utcnow().isoformat()}
    
    @router.get("/api/test/slow")
    async def test_slow():
        """Test endpoint for slow request monitoring."""
        # Simulate slow processing
        await asyncio.sleep(3)
        
        return {
            "message": "Slow request completed",
            "duration": 3.0,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Add monitoring endpoints
    @router.get("/monitoring/health")
    async def health():
        """Health check endpoint."""
        return monitoring_manager.get_health_status()
    
    @router.get("/monitoring/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return monitoring_manager.get_metrics()
    
    @router.get("/monitoring/performance")
    async def performance():
        """Performance summary endpoint."""
        return monitoring_manager.get_performance_summary()
    
    @router.get("/monitoring/errors")
    async def errors():
        """Error summary endpoint."""
        return monitoring_manager.get_error_summary()
    
    @router.get("/monitoring/alerts")
    async def alerts(hours: int = 24):
        """Alerts endpoint."""
        return monitoring_manager.get_alerts(hours)
    
    # Include router
    app.include_router(router)
    
    return app

# =============================================================================
# EXAMPLE 3: DEVELOPMENT MIDDLEWARE INTEGRATION
# =============================================================================

def create_development_middleware_app() -> FastAPI:
    """
    Example 3: Development middleware integration with detailed logging and debugging.
    """
    
    # Create FastAPI app
    app = FastAPI(
        title="Development Middleware Example",
        description="Example with development-optimized middleware",
        version="1.0.0"
    )
    
    # Create development middleware configuration
    config = create_development_enhanced_config()
    
    # Setup middleware
    middleware_manager = setup_enhanced_middleware(app, config)
    
    # Create router
    router = APIRouter()
    
    @router.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Development middleware example",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": "development",
            "debug": True
        }
    
    @router.get("/debug/users/{user_id}")
    @handle_http_exceptions
    async def get_user_debug(user_id: str, request: Request):
        """Get user with detailed debugging information."""
        # Log request details
        logger.info("Processing user request", user_id=user_id, headers=dict(request.headers))
        
        # Simulate various scenarios
        if user_id == "validation_error":
            raise ValidationError(
                message="Invalid user ID format",
                field="user_id",
                value=user_id,
                validation_errors=["Must be alphanumeric", "Must be 3-20 characters"]
            )
        
        if user_id == "auth_error":
            raise AuthenticationError(
                message="Invalid authentication token",
                auth_method="bearer"
            )
        
        if user_id == "authz_error":
            raise AuthorizationError(
                message="Insufficient permissions",
                required_permission="admin",
                user_permissions=["read"]
            )
        
        if user_id == "not_found":
            raise ResourceNotFoundError(
                message="User not found in database",
                resource_type="user",
                resource_id=user_id
            )
        
        if user_id == "business_error":
            raise BusinessLogicError(
                message="User account is suspended",
                business_rule="account_status"
            )
        
        if user_id == "db_error":
            raise DatabaseError(
                message="Database connection timeout",
                operation="query",
                table="users",
                constraint="user_id"
            )
        
        if user_id == "unexpected":
            # This will be caught by the general exception handler
            raise RuntimeError("Unexpected runtime error")
        
        return {
            "user_id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "debug_info": {
                "request_id": request.headers.get("X-Request-ID"),
                "user_agent": request.headers.get("User-Agent"),
                "client_ip": request.client.host if request.client else None
            }
        }
    
    @router.post("/debug/users")
    @handle_http_exceptions
    async def create_user_debug(user_data: dict, request: Request):
        """Create user with detailed validation."""
        logger.info("Creating user", user_data=user_data)
        
        # Detailed validation
        validation_errors = []
        
        if not user_data.get("email"):
            validation_errors.append("Email is required")
        elif "@" not in user_data["email"]:
            validation_errors.append("Invalid email format")
        
        if not user_data.get("name"):
            validation_errors.append("Name is required")
        elif len(user_data["name"]) < 2:
            validation_errors.append("Name must be at least 2 characters")
        
        if not user_data.get("age"):
            validation_errors.append("Age is required")
        elif not isinstance(user_data["age"], int):
            validation_errors.append("Age must be a number")
        elif user_data["age"] < 0 or user_data["age"] > 150:
            validation_errors.append("Age must be between 0 and 150")
        
        if validation_errors:
            raise ValidationError(
                message="Multiple validation errors",
                validation_errors=validation_errors
            )
        
        return {
            "user_id": "123",
            "email": user_data["email"],
            "name": user_data["name"],
            "age": user_data["age"],
            "status": "created",
            "debug_info": {
                "request_id": request.headers.get("X-Request-ID"),
                "validation_passed": True
            }
        }
    
    @router.get("/debug/test/errors")
    async def test_errors_debug():
        """Test various error types for debugging."""
        
        error_scenarios = [
            ("validation", lambda: raise_bad_request("Test validation error", field="test")),
            ("not_found", lambda: raise_not_found("Test resource not found", resource_type="test")),
            ("unauthorized", lambda: raise_unauthorized("Test authentication error")),
            ("forbidden", lambda: raise_forbidden("Test authorization error")),
            ("internal", lambda: raise_internal_server_error("Test internal server error")),
            ("runtime", lambda: exec("raise RuntimeError('Test runtime error')")),
            ("type", lambda: exec("raise TypeError('Test type error')")),
            ("value", lambda: exec("raise ValueError('Test value error')")),
        ]
        
        scenario_name, error_func = random.choice(error_scenarios)
        
        logger.info("Testing error scenario", scenario=scenario_name)
        
        try:
            error_func()
        except Exception as e:
            logger.error("Error scenario triggered", scenario=scenario_name, error=str(e))
            raise
        
        return {"message": "No error triggered", "scenario": scenario_name}
    
    @router.get("/debug/test/performance")
    async def test_performance_debug():
        """Test performance monitoring with detailed logging."""
        start_time = time.time()
        
        logger.info("Starting performance test")
        
        # Simulate various processing steps
        steps = [
            ("database_query", 0.1),
            ("external_api_call", 0.2),
            ("data_processing", 0.15),
            ("cache_operation", 0.05),
            ("response_generation", 0.1)
        ]
        
        for step_name, duration in steps:
            logger.info(f"Processing step: {step_name}")
            await asyncio.sleep(duration)
        
        total_duration = time.time() - start_time
        
        logger.info("Performance test completed", duration=total_duration)
        
        return {
            "message": "Performance test completed",
            "duration": total_duration,
            "steps": steps,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Add debugging endpoints
    @router.get("/debug/middleware/status")
    async def middleware_status():
        """Get middleware status for debugging."""
        return {
            "middleware_enabled": config.enabled,
            "logging_enabled": config.logging.log_requests,
            "error_handling_enabled": config.error_handling.catch_unexpected_errors,
            "monitoring_enabled": config.monitoring.collect_metrics,
            "health_status": middleware_manager.get_health_status()
        }
    
    @router.get("/debug/middleware/performance")
    async def middleware_performance():
        """Get middleware performance data."""
        return middleware_manager.get_performance_summary()
    
    @router.get("/debug/middleware/errors")
    async def middleware_errors():
        """Get middleware error data."""
        return middleware_manager.get_error_summary()
    
    # Include router
    app.include_router(router)
    
    return app

# =============================================================================
# EXAMPLE 4: CUSTOM MIDDLEWARE INTEGRATION
# =============================================================================

def create_custom_middleware_app() -> FastAPI:
    """
    Example 4: Custom middleware integration with specific requirements.
    """
    
    # Create FastAPI app
    app = FastAPI(
        title="Custom Middleware Example",
        description="Example with custom middleware configuration",
        version="1.0.0"
    )
    
    # Create custom middleware configuration
    config = EnhancedMiddlewareConfig(
        environment="custom",
        enabled=True,
        logging=LoggingConfig(
            log_requests=True,
            log_responses=True,
            log_errors=True,
            log_request_headers=True,
            log_request_body=True,
            use_structured_logging=True,
            include_request_id=True,
            include_user_context=True
        ),
        error_handling=ErrorHandlingConfig(
            catch_unexpected_errors=True,
            log_full_traceback=True,
            sanitize_error_messages=False,  # Show full errors in custom setup
            include_error_codes=True,
            error_sampling_rate=1.0,
            error_alert_threshold=5,
            slow_request_threshold=0.5,  # Lower threshold for custom setup
            critical_request_threshold=2.0
        ),
        monitoring=MonitoringConfig(
            collect_metrics=True,
            track_response_times=True,
            track_memory_usage=True,
            track_cpu_usage=True,
            track_error_rates=True,
            track_error_types=True,
            track_slow_requests=True,
            enable_alerts=True,
            alert_thresholds={
                "error_rate": 0.1,  # Higher threshold for custom setup
                "response_time_p95": 1.0,  # Lower threshold
                "memory_usage": 0.7,
                "cpu_usage": 0.7
            }
        ),
        security_enabled=True,
        rate_limiting_enabled=True,
        rate_limit_requests=50,  # Lower rate limit for custom setup
        rate_limit_window=60,
        cors_enabled=True,
        compression_enabled=True
    )
    
    # Setup middleware
    middleware_manager = setup_enhanced_middleware(app, config)
    
    # Create router
    router = APIRouter()
    
    @router.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Custom middleware example",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": "custom",
            "features": [
                "Detailed logging",
                "Custom error handling",
                "Performance monitoring",
                "Rate limiting",
                "Security headers"
            ]
        }
    
    @router.get("/custom/users/{user_id}")
    @handle_http_exceptions
    async def get_custom_user(user_id: str, request: Request):
        """Get user with custom error handling."""
        # Custom validation logic
        if len(user_id) < 3:
            raise ValidationError(
                message="User ID must be at least 3 characters",
                field="user_id",
                value=user_id
            )
        
        if user_id.startswith("test_"):
            raise BusinessLogicError(
                message="Test users are not accessible",
                business_rule="test_user_access"
            )
        
        # Custom business logic
        if user_id == "premium_user":
            # Check if user has premium access
            auth_header = request.headers.get("Authorization")
            if not auth_header or "premium" not in auth_header:
                raise AuthorizationError(
                    message="Premium access required",
                    required_permission="premium"
                )
        
        return {
            "user_id": user_id,
            "name": "Custom User",
            "email": f"{user_id}@custom.example.com",
            "custom_field": "Custom value",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.post("/custom/process")
    @handle_http_exceptions
    async def custom_process(data: dict):
        """Custom processing endpoint with specific error handling."""
        # Custom validation
        if not data.get("type"):
            raise ValidationError(
                message="Processing type is required",
                field="type"
            )
        
        if data["type"] not in ["standard", "premium", "enterprise"]:
            raise ValidationError(
                message="Invalid processing type",
                field="type",
                value=data["type"],
                validation_errors=["Must be one of: standard, premium, enterprise"]
            )
        
        # Custom business logic
        if data["type"] == "enterprise" and not data.get("enterprise_key"):
            raise BusinessLogicError(
                message="Enterprise key required for enterprise processing",
                business_rule="enterprise_validation"
            )
        
        # Simulate processing
        await asyncio.sleep(0.5)  # Simulate processing time
        
        return {
            "status": "processed",
            "type": data["type"],
            "result": f"Processed with {data['type']} logic",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.get("/custom/test/rate-limit")
    async def test_rate_limit():
        """Test rate limiting."""
        return {
            "message": "Rate limit test successful",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.get("/custom/test/slow")
    async def test_slow_custom():
        """Test slow request detection."""
        await asyncio.sleep(1.5)  # Should trigger slow request alert
        return {
            "message": "Slow request completed",
            "duration": 1.5,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Add custom monitoring endpoints
    @router.get("/custom/monitoring/status")
    async def custom_status():
        """Custom status endpoint."""
        return {
            "status": "custom",
            "middleware_config": config.dict(),
            "health": middleware_manager.get_health_status(),
            "performance": middleware_manager.get_performance_summary(),
            "errors": middleware_manager.get_error_summary()
        }
    
    # Include router
    app.include_router(router)
    
    return app

# =============================================================================
# EXAMPLE 5: COMPLETE INTEGRATION EXAMPLE
# =============================================================================

async def create_complete_integration_app() -> FastAPI:
    """
    Example 5: Complete integration with all middleware features.
    """
    
    # Create FastAPI app
    app = FastAPI(
        title="Complete Middleware Integration",
        description="Complete example with all middleware features",
        version="1.0.0"
    )
    
    # Create comprehensive configuration
    middleware_config = create_production_enhanced_config()
    monitoring_config = create_production_monitoring_config()
    
    # Setup Redis
    redis_client = None
    try:
        redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected for complete integration")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        middleware_config.redis_enabled = False
        monitoring_config.redis_enabled = False
    
    # Setup middleware
    middleware_manager = setup_enhanced_middleware(app, middleware_config, redis_client)
    
    # Setup monitoring
    monitoring_manager = await setup_monitoring(app, monitoring_config, redis_client)
    
    # Create comprehensive router
    router = APIRouter()
    
    @router.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Complete middleware integration example",
            "timestamp": datetime.utcnow().isoformat(),
            "features": [
                "Enhanced error handling",
                "Structured logging",
                "Performance monitoring",
                "Real-time alerting",
                "Rate limiting",
                "Security headers",
                "Redis integration",
                "Prometheus metrics"
            ]
        }
    
    # User management endpoints
    @router.get("/api/v1/users/{user_id}")
    @handle_http_exceptions
    async def get_user_v1(user_id: str, request: Request):
        """Get user with comprehensive error handling."""
        # Authentication check
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise AuthenticationError(
                message="Authentication required",
                auth_method="bearer"
            )
        
        # Validation
        if not user_id.isalnum() or len(user_id) < 3:
            raise ValidationError(
                message="Invalid user ID format",
                field="user_id",
                value=user_id
            )
        
        # Business logic
        if user_id == "admin" and auth_header != "Bearer admin_token":
            raise AuthorizationError(
                message="Admin access required",
                required_permission="admin"
            )
        
        # Simulate database operations
        if user_id == "not_found":
            raise ResourceNotFoundError(
                message="User not found",
                resource_type="user",
                resource_id=user_id
            )
        
        if user_id == "db_error":
            raise DatabaseError(
                message="Database connection failed",
                operation="query",
                table="users"
            )
        
        return {
            "user_id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "status": "active",
            "created_at": datetime.utcnow().isoformat()
        }
    
    @router.post("/api/v1/users")
    @handle_http_exceptions
    async def create_user_v1(user_data: dict):
        """Create user with comprehensive validation."""
        # Comprehensive validation
        validation_errors = []
        
        if not user_data.get("email"):
            validation_errors.append("Email is required")
        elif "@" not in user_data["email"]:
            validation_errors.append("Invalid email format")
        
        if not user_data.get("name"):
            validation_errors.append("Name is required")
        elif len(user_data["name"]) < 2:
            validation_errors.append("Name must be at least 2 characters")
        
        if validation_errors:
            raise ValidationError(
                message="Validation failed",
                validation_errors=validation_errors
            )
        
        # Business logic
        if user_data["email"] == "duplicate@example.com":
            raise BusinessLogicError(
                message="User with this email already exists",
                business_rule="unique_email"
            )
        
        return {
            "user_id": "123",
            "email": user_data["email"],
            "name": user_data["name"],
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }
    
    # Test endpoints for monitoring
    @router.get("/api/v1/test/performance")
    async def test_performance_v1():
        """Test performance monitoring."""
        start_time = time.time()
        
        # Simulate various processing steps
        await asyncio.sleep(0.1)  # Database query
        await asyncio.sleep(0.2)  # External API call
        await asyncio.sleep(0.1)  # Data processing
        
        duration = time.time() - start_time
        
        return {
            "message": "Performance test completed",
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.get("/api/v1/test/errors")
    async def test_errors_v1():
        """Test error monitoring."""
        
        error_scenarios = [
            ("validation", lambda: raise_bad_request("Test validation error")),
            ("not_found", lambda: raise_not_found("Test resource not found")),
            ("unauthorized", lambda: raise_unauthorized("Test authentication error")),
            ("forbidden", lambda: raise_forbidden("Test authorization error")),
            ("internal", lambda: raise_internal_server_error("Test internal server error")),
        ]
        
        scenario_name, error_func = random.choice(error_scenarios)
        
        if random.random() < 0.3:  # 30% chance of error
            error_func()
        
        return {
            "message": "No error this time",
            "scenario": scenario_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.get("/api/v1/test/slow")
    async def test_slow_v1():
        """Test slow request monitoring."""
        await asyncio.sleep(3)  # Simulate slow processing
        
        return {
            "message": "Slow request completed",
            "duration": 3.0,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Monitoring endpoints
    @router.get("/api/v1/monitoring/health")
    async def health_v1():
        """Health check endpoint."""
        return monitoring_manager.get_health_status()
    
    @router.get("/api/v1/monitoring/metrics")
    async def metrics_v1():
        """Prometheus metrics endpoint."""
        return monitoring_manager.get_metrics()
    
    @router.get("/api/v1/monitoring/performance")
    async def performance_v1():
        """Performance summary endpoint."""
        return monitoring_manager.get_performance_summary()
    
    @router.get("/api/v1/monitoring/errors")
    async def errors_v1():
        """Error summary endpoint."""
        return monitoring_manager.get_error_summary()
    
    @router.get("/api/v1/monitoring/alerts")
    async def alerts_v1(hours: int = 24):
        """Alerts endpoint."""
        return monitoring_manager.get_alerts(hours)
    
    # Include router
    app.include_router(router)
    
    return app

# =============================================================================
# MAIN EXAMPLE RUNNER
# =============================================================================

async def run_examples():
    """Run all middleware integration examples."""
    
    examples = [
        ("Basic Middleware", create_basic_middleware_app),
        ("Development Middleware", create_development_middleware_app),
        ("Custom Middleware", create_custom_middleware_app),
        ("Production Middleware", create_production_middleware_app),
        ("Complete Integration", create_complete_integration_app)
    ]
    
    print("🔗 Middleware Integration Examples")
    print("=" * 50)
    
    for name, create_func in examples:
        print(f"\n📋 {name}")
        print("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(create_func):
                app = await create_func()
            else:
                app = create_func()
            
            print(f"✅ {name} created successfully")
            print(f"   - Title: {app.title}")
            print(f"   - Version: {app.version}")
            print(f"   - Routes: {len(app.routes)}")
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print(f"\n🎉 All examples processed!")
    print("\nTo run any example:")
    print("1. Import the create function")
    print("2. Call it to create the FastAPI app")
    print("3. Run with uvicorn: uvicorn app:app --reload")

match __name__:
    case "__main__":
    asyncio.run(run_examples()) 