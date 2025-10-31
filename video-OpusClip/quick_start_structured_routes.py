"""
🚀 Quick Start Guide for Structured Routes and Dependencies

This script provides a quick introduction to the structured routing system
with practical examples and common usage patterns.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import structured routes components
from structured_routes import (
    RouteCategory, RoutePriority, RouteConfig,
    BaseRouter, CommonDependencies, RouterFactory, RouteRegistry,
    create_structured_app
)

# Import Video-OpusClip components
from fastapi_dependency_injection import DependencyContainer, get_dependency_container
from async_database import AsyncVideoDatabase, AsyncBatchDatabaseOperations

logger = structlog.get_logger(__name__)

# =============================================================================
# Quick Start Examples
# =============================================================================

def example_1_basic_router_creation():
    """Example 1: Basic router creation and usage."""
    print("\n=== Example 1: Basic Router Creation ===")
    
    # Create a basic router
    router = APIRouter(prefix="/api/v1/example", tags=["example"])
    
    # Define a simple endpoint
    @router.get("/hello")
    async def hello_world():
        return {"message": "Hello, World!", "timestamp": datetime.now().isoformat()}
    
    # Define an endpoint with parameters
    @router.get("/greet/{name}")
    async def greet_user(name: str, age: Optional[int] = Query(None)):
        message = f"Hello, {name}!"
        if age:
            message += f" You are {age} years old."
        
        return {
            "message": message,
            "name": name,
            "age": age,
            "timestamp": datetime.now().isoformat()
        }
    
    print("✅ Created basic router with endpoints:")
    print("  - GET /api/v1/example/hello")
    print("  - GET /api/v1/example/greet/{name}")
    
    return router

def example_2_route_configuration():
    """Example 2: Route configuration and metadata."""
    print("\n=== Example 2: Route Configuration ===")
    
    # Create route configuration
    config = RouteConfig(
        path="/api/v1/videos",
        method="POST",
        tags=["video", "processing"],
        summary="Create video processing request",
        description="Create a new video processing request with AI analysis",
        status_code=201,
        category=RouteCategory.VIDEO,
        priority=RoutePriority.HIGH,
        rate_limit=100,  # 100 requests per minute
        cache_ttl=300    # 5 minutes cache
    )
    
    print("✅ Route configuration created:")
    print(f"  - Path: {config.path}")
    print(f"  - Method: {config.method}")
    print(f"  - Category: {config.category}")
    print(f"  - Priority: {config.priority}")
    print(f"  - Rate Limit: {config.rate_limit} req/min")
    print(f"  - Cache TTL: {config.cache_ttl} seconds")

def example_3_dependency_injection():
    """Example 3: Dependency injection patterns."""
    print("\n=== Example 3: Dependency Injection ===")
    
    # Mock dependency container
    class MockDependencyContainer:
        async def get_db_session_dependency(self):
            async def mock_db_session():
                return {"connection": "mock_db"}
            return mock_db_session
        
        async def get_cache_client_dependency(self):
            async def mock_cache_client():
                return {"connection": "mock_cache"}
            return mock_cache_client
    
    # Create common dependencies
    container = MockDependencyContainer()
    deps = CommonDependencies(container)
    
    # Create router with dependencies
    router = APIRouter(prefix="/api/v1/deps", tags=["dependencies"])
    
    @router.get("/user")
    async def get_user_info(
        current_user: Dict[str, Any] = Depends(deps.get_current_user)
    ):
        if current_user:
            return {
                "user": current_user,
                "message": "User authenticated"
            }
        else:
            return {
                "user": None,
                "message": "No authentication"
            }
    
    @router.get("/database")
    async def check_database(
        db_session = Depends(deps.get_db_session)
    ):
        return {
            "database": "connected",
            "session": db_session,
            "timestamp": datetime.now().isoformat()
        }
    
    print("✅ Created router with dependency injection:")
    print("  - GET /api/v1/deps/user (with authentication)")
    print("  - GET /api/v1/deps/database (with database session)")

def example_4_route_handlers():
    """Example 4: Route handlers with business logic."""
    print("\n=== Example 4: Route Handlers ===")
    
    # Mock dependencies
    class MockDependencies:
        async def get_current_user(self):
            return {"id": "user_123", "email": "user@example.com"}
        
        async def get_video_database(self):
            return MockVideoDatabase()
    
    class MockVideoDatabase:
        async def create_video_record(self, video_data: Dict[str, Any]) -> int:
            return 1
        
        async def get_video_by_id(self, video_id: int) -> Optional[Dict[str, Any]]:
            return {
                "id": video_id,
                "title": "Sample Video",
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
    
    # Create route handlers
    deps = MockDependencies()
    
    # Video route handlers
    class VideoHandlers:
        def __init__(self, dependencies):
            self.deps = dependencies
        
        async def create_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
            """Create a new video processing request."""
            current_user = await self.deps.get_current_user()
            video_db = await self.deps.get_video_database()
            
            # Add user ID to video data
            video_data["user_id"] = current_user["id"]
            video_data["created_at"] = datetime.now().isoformat()
            
            # Create video record
            video_id = await video_db.create_video_record(video_data)
            
            return {
                "id": video_id,
                "title": video_data["title"],
                "status": "pending",
                "user_id": current_user["id"],
                "created_at": video_data["created_at"],
                "message": "Video processing request created successfully"
            }
        
        async def get_video(self, video_id: int) -> Dict[str, Any]:
            """Get video by ID."""
            video_db = await self.deps.get_video_database()
            video = await video_db.get_video_by_id(video_id)
            
            if not video:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Video not found"
                )
            
            return {
                "video": video,
                "message": "Video retrieved successfully"
            }
    
    # Create router with handlers
    router = APIRouter(prefix="/api/v1/videos", tags=["video"])
    handlers = VideoHandlers(deps)
    
    @router.post("/")
    async def create_video(video_data: Dict[str, Any]):
        return await handlers.create_video(video_data)
    
    @router.get("/{video_id}")
    async def get_video(video_id: int):
        return await handlers.get_video(video_id)
    
    print("✅ Created route handlers with business logic:")
    print("  - POST /api/v1/videos/ (create video)")
    print("  - GET /api/v1/videos/{video_id} (get video)")

def example_5_router_factory():
    """Example 5: Router factory pattern."""
    print("\n=== Example 5: Router Factory ===")
    
    # Mock dependencies
    class MockDependencies:
        async def get_current_user(self):
            return {"id": "user_123", "email": "user@example.com"}
        
        async def get_video_database(self):
            return MockVideoDatabase()
        
        async def get_batch_database(self):
            return MockBatchDatabase()
    
    class MockVideoDatabase:
        async def create_video_record(self, video_data: Dict[str, Any]) -> int:
            return 1
        
        async def get_videos_by_status(self, status: str) -> List[Dict[str, Any]]:
            return [
                {"id": 1, "title": "Video 1", "status": status},
                {"id": 2, "title": "Video 2", "status": status}
            ]
    
    class MockBatchDatabase:
        async def batch_insert_videos(self, videos: List[Dict[str, Any]]) -> List[int]:
            return [1, 2, 3]
    
    # Create router factory
    deps = MockDependencies()
    factory = RouterFactory(deps)
    
    # Create different types of routers
    video_router = factory.create_video_router()
    batch_router = factory.create_batch_router()
    analytics_router = factory.create_analytics_router()
    health_router = factory.create_health_router()
    
    print("✅ Created routers using factory pattern:")
    print(f"  - Video Router: {len(video_router.routes)} routes")
    print(f"  - Batch Router: {len(batch_router.routes)} routes")
    print(f"  - Analytics Router: {len(analytics_router.routes)} routes")
    print(f"  - Health Router: {len(health_router.routes)} routes")

def example_6_route_registry():
    """Example 6: Route registry and management."""
    print("\n=== Example 6: Route Registry ===")
    
    # Create FastAPI app
    app = FastAPI(title="Quick Start Example", version="1.0.0")
    
    # Mock dependency container
    class MockContainer:
        async def get_db_session_dependency(self):
            async def mock_db():
                return {"connection": "mock_db"}
            return mock_db
        
        async def get_cache_client_dependency(self):
            async def mock_cache():
                return {"connection": "mock_cache"}
            return mock_cache
    
    # Create route registry
    container = MockContainer()
    registry = RouteRegistry(app, container)
    
    # Register routers
    registry.register_router("video", APIRouter(prefix="/api/v1/videos"), RouteConfig(
        path="/api/v1/videos",
        method="*",
        tags=["video"],
        summary="Video processing operations",
        category=RouteCategory.VIDEO,
        priority=RoutePriority.HIGH
    ))
    
    registry.register_router("batch", APIRouter(prefix="/api/v1/batch"), RouteConfig(
        path="/api/v1/batch",
        method="*",
        tags=["batch"],
        summary="Batch processing operations",
        category=RouteCategory.BATCH,
        priority=RoutePriority.NORMAL
    ))
    
    print("✅ Registered routers in registry:")
    for name, router in registry.get_all_routers().items():
        config = registry.get_route_config(name)
        print(f"  - {name}: {config.category.value} (priority: {config.priority.value})")

def example_7_complete_application():
    """Example 7: Complete structured application."""
    print("\n=== Example 7: Complete Structured Application ===")
    
    # Create the complete structured application
    app = create_structured_app()
    
    print("✅ Created complete structured application:")
    print(f"  - Title: {app.title}")
    print(f"  - Version: {app.version}")
    print(f"  - Routes: {len(app.routes)} total routes")
    print(f"  - Middleware: {len(app.user_middleware)} middleware")
    
    # List all routes
    print("\n📋 Available Routes:")
    for route in app.routes:
        if hasattr(route, 'path'):
            methods = ', '.join(route.methods) if hasattr(route, 'methods') else 'GET'
            print(f"  - {methods} {route.path}")
    
    return app

def example_8_custom_router():
    """Example 8: Creating custom routers."""
    print("\n=== Example 8: Custom Router ===")
    
    # Create custom router extending BaseRouter
    class CustomVideoRouter(BaseRouter):
        def __init__(self):
            super().__init__(prefix="/api/v1/custom", tags=["custom"])
        
        def setup_routes(self):
            """Setup custom routes."""
            
            @self.router.get("/videos")
            async def list_custom_videos():
                return {
                    "videos": [
                        {"id": 1, "title": "Custom Video 1"},
                        {"id": 2, "title": "Custom Video 2"}
                    ],
                    "message": "Custom video list"
                }
            
            @self.router.post("/videos")
            async def create_custom_video(video_data: Dict[str, Any]):
                return {
                    "id": 3,
                    "title": video_data.get("title", "Custom Video"),
                    "status": "created",
                    "message": "Custom video created"
                }
            
            @self.router.get("/stats")
            async def get_custom_stats():
                return {
                    "total_videos": 2,
                    "processing": 1,
                    "completed": 1,
                    "message": "Custom statistics"
                }
    
    # Create and setup custom router
    custom_router = CustomVideoRouter()
    custom_router.setup_routes()
    
    print("✅ Created custom router:")
    print(f"  - Prefix: {custom_router.router.prefix}")
    print(f"  - Tags: {custom_router.router.tags}")
    print(f"  - Routes: {len(custom_router.router.routes)} routes")
    
    return custom_router

def example_9_error_handling():
    """Example 9: Error handling patterns."""
    print("\n=== Example 9: Error Handling ===")
    
    # Create router with error handling
    router = APIRouter(prefix="/api/v1/errors", tags=["error-handling"])
    
    @router.get("/divide/{a}/{b}")
    async def divide_numbers(a: float, b: float):
        """Example endpoint with error handling."""
        try:
            if b == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Division by zero is not allowed"
                )
            
            result = a / b
            return {
                "operation": "division",
                "a": a,
                "b": b,
                "result": result,
                "message": "Division completed successfully"
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in division: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )
    
    @router.get("/simulate-error")
    async def simulate_error(error_type: str = Query("validation")):
        """Simulate different types of errors."""
        if error_type == "validation":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Validation error: Invalid input data"
            )
        elif error_type == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found"
            )
        elif error_type == "server_error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
        else:
            return {"message": "No error simulated"}
    
    print("✅ Created router with error handling:")
    print("  - GET /api/v1/errors/divide/{a}/{b} (with validation)")
    print("  - GET /api/v1/errors/simulate-error (error simulation)")
    
    return router

def example_10_performance_monitoring():
    """Example 10: Performance monitoring and metrics."""
    print("\n=== Example 10: Performance Monitoring ===")
    
    # Create router with performance monitoring
    router = APIRouter(prefix="/api/v1/performance", tags=["performance"])
    
    @router.get("/slow-operation")
    async def slow_operation():
        """Simulate a slow operation."""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        process_time = time.time() - start_time
        
        return {
            "message": "Slow operation completed",
            "process_time": process_time,
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/fast-operation")
    async def fast_operation():
        """Simulate a fast operation."""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        process_time = time.time() - start_time
        
        return {
            "message": "Fast operation completed",
            "process_time": process_time,
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/metrics")
    async def get_metrics():
        """Get performance metrics."""
        return {
            "uptime": time.time(),
            "requests_processed": 1000,
            "average_response_time": 0.5,
            "error_rate": 0.01,
            "active_connections": 10,
            "timestamp": datetime.now().isoformat()
        }
    
    print("✅ Created router with performance monitoring:")
    print("  - GET /api/v1/performance/slow-operation")
    print("  - GET /api/v1/performance/fast-operation")
    print("  - GET /api/v1/performance/metrics")
    
    return router

# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Run all quick start examples."""
    print("🚀 Quick Start Guide for Structured Routes and Dependencies")
    print("=" * 70)
    
    # Run examples
    example_1_basic_router_creation()
    example_2_route_configuration()
    example_3_dependency_injection()
    example_4_route_handlers()
    example_5_router_factory()
    example_6_route_registry()
    example_7_complete_application()
    example_8_custom_router()
    example_9_error_handling()
    example_10_performance_monitoring()
    
    print("\n✅ All examples completed successfully!")
    print("\n📚 Next Steps:")
    print("  1. Review the structured_routes.py file for full implementation")
    print("  2. Check STRUCTURED_ROUTES_GUIDE.md for detailed documentation")
    print("  3. Start building your own routes using these patterns")
    print("  4. Customize the dependency injection for your specific needs")

if __name__ == "__main__":
    asyncio.run(main()) 