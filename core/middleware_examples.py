from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import random
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog
import redis.asyncio as redis
from ..utils.middleware_system import (
    import uvicorn
from typing import Any, List, Dict, Optional
import logging
"""
Middleware Examples - FastAPI Middleware Usage Examples
Demonstrates how to use the comprehensive middleware system for logging,
error monitoring, and performance optimization.
"""



    MiddlewareManager,
    MiddlewareConfig,
    create_middleware_config,
    create_production_middleware_config,
    create_development_middleware_config
)

logger = structlog.get_logger(__name__)

# =============================================================================
# EXAMPLE 1: BASIC MIDDLEWARE SETUP
# =============================================================================

def create_basic_app() -> FastAPI:
    """Create a FastAPI app with basic middleware setup."""
    
    # Create middleware configuration
    config = create_middleware_config(
        logging_enabled=True,
        performance_monitoring_enabled=True,
        error_monitoring_enabled=True,
        security_enabled=True,
        rate_limiting_enabled=True,
        rate_limit_requests=50,  # 50 requests per minute
        slow_request_threshold=1.0
    )
    
    # Create middleware manager
    middleware_manager = MiddlewareManager(config)
    
    # Create FastAPI app
    app = FastAPI(
        title="Basic Middleware Example",
        version="1.0.0",
        description="Example with basic middleware setup"
    )
    
    # Setup middleware
    middleware_manager.setup_middleware(app)
    
    # Add routes
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "Hello World", "timestamp": time.time()}
    
    @app.get("/slow")
    async def slow_endpoint():
        """Simulate a slow endpoint."""
        await asyncio.sleep(random.uniform(0.5, 2.0))
        return {"message": "Slow response", "duration": "variable"}
    
    @app.get("/error")
    async def error_endpoint():
        """Simulate an error."""
        if random.random() < 0.3:
            raise HTTPException(status_code=500, detail="Random error")
        return {"message": "Sometimes works"}
    
    @app.get("/metrics")
    async def metrics():
        """Get Prometheus metrics."""
        return middleware_manager.get_metrics()
    
    return app

# =============================================================================
# EXAMPLE 2: PRODUCTION MIDDLEWARE SETUP
# =============================================================================

async def create_production_app() -> FastAPI:
    """Create a FastAPI app with production-optimized middleware."""
    
    # Create Redis client for rate limiting and error storage
    redis_client = redis.from_url("redis://localhost:6379")
    
    # Create production middleware configuration
    config = create_production_middleware_config()
    
    # Create middleware manager with Redis
    middleware_manager = MiddlewareManager(config, redis_client)
    
    # Create FastAPI app
    app = FastAPI(
        title="Production Middleware Example",
        version="1.0.0",
        description="Example with production middleware setup"
    )
    
    # Setup middleware
    middleware_manager.setup_middleware(app)
    
    # Add production routes
    @app.get("/api/v1/health")
    async def health_check():
        
    """health_check function."""
return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0"
        }
    
    @app.get("/api/v1/users/{user_id}")
    async def get_user(user_id: int):
        """Get user information."""
        # Simulate database query
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        if user_id > 1000:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com"
        }
    
    @app.post("/api/v1/users")
    async def create_user(user_data: Dict[str, Any]):
        """Create a new user."""
        # Simulate processing
        await asyncio.sleep(random.uniform(0.2, 1.0))
        
        return {
            "id": random.randint(1, 1000),
            "name": user_data.get("name"),
            "email": user_data.get("email"),
            "created_at": time.time()
        }
    
    @app.get("/api/v1/performance")
    async def performance_summary():
        """Get performance monitoring summary."""
        return middleware_manager.get_performance_summary()
    
    @app.get("/api/v1/errors")
    async def error_summary():
        """Get error monitoring summary."""
        return middleware_manager.get_error_summary()
    
    return app

# =============================================================================
# EXAMPLE 3: DEVELOPMENT MIDDLEWARE SETUP
# =============================================================================

def create_development_app() -> FastAPI:
    """Create a FastAPI app with development-optimized middleware."""
    
    # Create development middleware configuration
    config = create_development_middleware_config()
    
    # Create middleware manager
    middleware_manager = MiddlewareManager(config)
    
    # Create FastAPI app
    app = FastAPI(
        title="Development Middleware Example",
        version="1.0.0-dev",
        description="Example with development middleware setup"
    )
    
    # Setup middleware
    middleware_manager.setup_middleware(app)
    
    # Add development routes with detailed logging
    @app.get("/debug/request-info")
    async def debug_request_info(request: Request):
        """Debug endpoint showing request information."""
        return {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "query_params": dict(request.query_params)
        }
    
    @app.post("/debug/test-error")
    async def test_error(error_type: str = "validation"):
        """Test different types of errors."""
        if error_type == "validation":
            raise HTTPException(status_code=422, detail="Validation error")
        elif error_type == "server":
            raise HTTPException(status_code=500, detail="Server error")
        elif error_type == "not_found":
            raise HTTPException(status_code=404, detail="Not found")
        else:
            return {"message": "No error"}
    
    @app.get("/debug/slow/{duration}")
    async def debug_slow_request(duration: float):
        """Test slow request monitoring."""
        await asyncio.sleep(duration)
        return {"message": f"Request took {duration} seconds"}
    
    return app

# =============================================================================
# EXAMPLE 4: CUSTOM MIDDLEWARE CONFIGURATION
# =============================================================================

def create_custom_middleware_app() -> FastAPI:
    """Create a FastAPI app with custom middleware configuration."""
    
    # Create custom middleware configuration
    config = MiddlewareConfig(
        # Logging - detailed for debugging
        logging_enabled=True,
        log_request_body=True,
        log_response_body=True,
        log_headers=True,
        sensitive_headers=["authorization", "x-api-key", "cookie"],
        
        # Performance monitoring - aggressive thresholds
        performance_monitoring_enabled=True,
        slow_request_threshold=0.5,  # 500ms threshold
        performance_metrics_enabled=True,
        
        # Error monitoring - sample all errors
        error_monitoring_enabled=True,
        error_sampling_rate=1.0,
        error_retention_days=7,
        
        # Security - strict settings
        security_enabled=True,
        rate_limiting_enabled=True,
        rate_limit_requests=30,  # 30 requests per minute
        rate_limit_window=60,
        
        # CORS - restrictive
        cors_enabled=True,
        cors_origins=["https://app.example.com"],
        cors_credentials=True,
        cors_methods=["GET", "POST", "PUT", "DELETE"],
        cors_headers=["Content-Type", "Authorization"],
        
        # Compression - aggressive
        compression_enabled=True,
        compression_min_size=500,
        
        # Trusted hosts
        trusted_hosts_enabled=True,
        trusted_hosts=["app.example.com", "api.example.com"],
        
        # Redis integration
        redis_enabled=True,
        redis_url="redis://localhost:6379",
        redis_ttl=1800
    )
    
    # Create middleware manager
    middleware_manager = MiddlewareManager(config)
    
    # Create FastAPI app
    app = FastAPI(
        title="Custom Middleware Example",
        version="1.0.0",
        description="Example with custom middleware configuration"
    )
    
    # Setup middleware
    middleware_manager.setup_middleware(app)
    
    # Add custom routes
    @app.get("/api/v1/secure")
    async def secure_endpoint():
        """Secure endpoint with strict rate limiting."""
        return {"message": "Secure endpoint", "timestamp": time.time()}
    
    @app.post("/api/v1/data")
    async def process_data(data: Dict[str, Any]):
        """Process data with detailed logging."""
        # Simulate processing
        await asyncio.sleep(random.uniform(0.1, 0.8))
        
        return {
            "processed": True,
            "data_size": len(str(data)),
            "timestamp": time.time()
        }
    
    return app

# =============================================================================
# EXAMPLE 5: MIDDLEWARE WITH LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with middleware setup."""
    logger.info("Starting application with middleware")
    
    # Initialize Redis connection
    redis_client = redis.from_url("redis://localhost:6379")
    await redis_client.ping()
    logger.info("Redis connection established")
    
    # Store Redis client in app state
    app.state.redis_client = redis_client
    
    yield
    
    # Cleanup
    await redis_client.close()
    logger.info("Application shutting down")

def create_lifespan_middleware_app() -> FastAPI:
    """Create a FastAPI app with middleware and lifespan management."""
    
    # Create middleware configuration
    config = create_production_middleware_config()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="Lifespan Middleware Example",
        version="1.0.0",
        description="Example with middleware and lifespan management",
        lifespan=lifespan
    )
    
    # Create middleware manager (will be configured in lifespan)
    app.state.middleware_manager = None
    
    # Add routes
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "Hello with lifespan middleware"}
    
    @app.get("/redis-test")
    async def redis_test():
        """Test Redis connection."""
        redis_client = app.state.redis_client
        await redis_client.set("test_key", "test_value")
        value = await redis_client.get("test_key")
        return {"redis_test": value.decode() if value else None}
    
    return app

# =============================================================================
# EXAMPLE 6: MIDDLEWARE WITH DEPENDENCY INJECTION
# =============================================================================

class UserService:
    """Example service for dependency injection."""
    
    def __init__(self) -> Any:
        self.users = {}
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        await asyncio.sleep(random.uniform(0.1, 0.3))
        return self.users.get(user_id)
    
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user."""
        await asyncio.sleep(random.uniform(0.2, 0.5))
        user_id = len(self.users) + 1
        user = {"id": user_id, **user_data}
        self.users[user_id] = user
        return user

def create_dependency_middleware_app() -> FastAPI:
    """Create a FastAPI app with middleware and dependency injection."""
    
    # Create middleware configuration
    config = create_middleware_config(
        logging_enabled=True,
        performance_monitoring_enabled=True,
        error_monitoring_enabled=True,
        rate_limiting_enabled=True,
        rate_limit_requests=100
    )
    
    # Create middleware manager
    middleware_manager = MiddlewareManager(config)
    
    # Create FastAPI app
    app = FastAPI(
        title="Dependency Middleware Example",
        version="1.0.0",
        description="Example with middleware and dependency injection"
    )
    
    # Setup middleware
    middleware_manager.setup_middleware(app)
    
    # Create service instance
    user_service = UserService()
    
    # Dependency
    def get_user_service() -> UserService:
        return user_service
    
    # Add routes with dependencies
    @app.get("/users/{user_id}")
    async def get_user(
        user_id: int,
        service: UserService = Depends(get_user_service)
    ):
        """Get user with dependency injection."""
        user = await service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    
    @app.post("/users")
    async def create_user(
        user_data: Dict[str, Any],
        service: UserService = Depends(get_user_service)
    ):
        """Create user with dependency injection."""
        user = await service.create_user(user_data)
        return user
    
    return app

# =============================================================================
# EXAMPLE 7: MIDDLEWARE WITH CUSTOM METRICS
# =============================================================================

def create_custom_metrics_app() -> FastAPI:
    """Create a FastAPI app with custom metrics and middleware."""
    
    # Create middleware configuration
    config = create_middleware_config(
        performance_monitoring_enabled=True,
        performance_metrics_enabled=True,
        logging_enabled=True
    )
    
    # Create middleware manager
    middleware_manager = MiddlewareManager(config)
    
    # Create FastAPI app
    app = FastAPI(
        title="Custom Metrics Example",
        version="1.0.0",
        description="Example with custom metrics and middleware"
    )
    
    # Setup middleware
    middleware_manager.setup_middleware(app)
    
    # Custom metrics tracking
    request_counts = {}
    
    @app.middleware("http")
    async def custom_metrics_middleware(request: Request, call_next):
        """Custom metrics middleware."""
        endpoint = request.url.path
        
        # Track request count
        request_counts[endpoint] = request_counts.get(endpoint, 0) + 1
        
        # Process request
        response = await call_next(request)
        
        # Add custom headers
        response.headers["X-Request-Count"] = str(request_counts[endpoint])
        response.headers["X-Endpoint"] = endpoint
        
        return response
    
    # Add routes
    @app.get("/api/v1/metrics/custom")
    async def custom_metrics():
        """Get custom metrics."""
        return {
            "request_counts": request_counts,
            "total_requests": sum(request_counts.values()),
            "endpoints": list(request_counts.keys())
        }
    
    @app.get("/api/v1/metrics/prometheus")
    async def prometheus_metrics():
        """Get Prometheus metrics."""
        return middleware_manager.get_metrics()
    
    return app

# =============================================================================
# EXAMPLE 8: MIDDLEWARE TESTING
# =============================================================================

def create_test_app() -> FastAPI:
    """Create a FastAPI app for testing middleware."""
    
    # Create test middleware configuration
    config = MiddlewareConfig(
        logging_enabled=True,
        log_request_body=True,
        log_response_body=True,
        performance_monitoring_enabled=True,
        error_monitoring_enabled=True,
        rate_limiting_enabled=False,  # Disable rate limiting for tests
        security_enabled=True,
        cors_enabled=True,
        compression_enabled=False  # Disable compression for tests
    )
    
    # Create middleware manager
    middleware_manager = MiddlewareManager(config)
    
    # Create FastAPI app
    app = FastAPI(
        title="Test Middleware Example",
        version="1.0.0-test",
        description="Example for testing middleware functionality"
    )
    
    # Setup middleware
    middleware_manager.setup_middleware(app)
    
    # Test routes
    @app.get("/test/health")
    async def test_health():
        """Health check for testing."""
        return {"status": "healthy", "timestamp": time.time()}
    
    @app.get("/test/slow")
    async def test_slow():
        """Slow endpoint for testing performance monitoring."""
        await asyncio.sleep(1.5)  # Always slow
        return {"message": "Slow response"}
    
    @app.get("/test/error")
    async def test_error():
        """Error endpoint for testing error monitoring."""
        raise HTTPException(status_code=500, detail="Test error")
    
    @app.post("/test/data")
    async def test_data(data: Dict[str, Any]):
        """Data endpoint for testing request/response logging."""
        return {"received": data, "processed": True}
    
    return app

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

async def demo_middleware_functionality():
    """Demonstrate middleware functionality."""
    print("\n🔧 MIDDLEWARE SYSTEM DEMO")
    print("=========================")
    
    # Create different app configurations
    apps = {
        "Basic": create_basic_app(),
        "Development": create_development_app(),
        "Production": await create_production_app(),
        "Custom": create_custom_middleware_app(),
        "Test": create_test_app()
    }
    
    for name, app in apps.items():
        print(f"\n📋 {name} App Configuration:")
        print(f"   - Title: {app.title}")
        print(f"   - Version: {app.version}")
        print(f"   - Routes: {len(app.routes)}")
        
        # Check for middleware-specific endpoints
        middleware_endpoints = [
            route.path for route in app.routes 
            if any(keyword in route.path for keyword in ["metrics", "health", "debug"])
        ]
        
        if middleware_endpoints:
            print(f"   - Middleware endpoints: {middleware_endpoints}")
    
    print("\n✅ Middleware system demonstration completed")

# Example usage
if __name__ == "__main__":
    
    # Run the demo
    asyncio.run(demo_middleware_functionality())
    
    # Start a test server
    app = create_test_app()
    uvicorn.run(app, host="0.0.0.0", port=8000) 