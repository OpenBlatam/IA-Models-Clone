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
import uuid
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi_dependency_injection import (
from typing import Any, List, Dict, Optional
import logging
"""
FastAPI Dependency Injection Demo for Lazy Loading System

This demo showcases:
- FastAPI dependency injection for state management
- Shared resource lifecycle management
- Configuration management through dependencies
- Request-scoped dependencies
- Performance monitoring
- Error handling with dependencies
- Testing with dependency injection
"""



    DependencyConfig, DependencyManager, create_app,
    get_lazy_manager_dependency, get_loader_dependency,
    get_config, get_stats_dependency, get_request_id,
    get_user_context, get_performance_monitor,
    LazyLoadingService, TestDependencyManager
)


# Pydantic models for API
class ProductRequest(BaseModel):
    """Product request model."""
    product_id: str = Field(..., description="Product ID to retrieve")


class BatchProductRequest(BaseModel):
    """Batch product request model."""
    product_ids: List[str] = Field(..., description="List of product IDs")


class PaginationRequest(BaseModel):
    """Pagination request model."""
    page: int = Field(0, ge=0, description="Page number")
    page_size: int = Field(50, ge=1, le=1000, description="Page size")


class CursorRequest(BaseModel):
    """Cursor request model."""
    cursor: Optional[int] = Field(None, description="Cursor for pagination")
    limit: int = Field(50, ge=1, le=1000, description="Limit of items")


class WindowRequest(BaseModel):
    """Window request model."""
    start: int = Field(0, ge=0, description="Start index")
    size: int = Field(100, ge=1, le=1000, description="Window size")


class PerformanceResponse(BaseModel):
    """Performance response model."""
    request_id: str
    duration: float
    success: bool
    error_message: Optional[str] = None


class SystemStatsResponse(BaseModel):
    """System statistics response model."""
    loaders: Dict[str, Any]
    memory_usage: Dict[str, Any]
    performance: Dict[str, Any]
    uptime: float


# Create FastAPI application with dependency injection
app = create_app()

# Global performance monitor
performance_monitor = get_performance_monitor()


# Middleware for performance tracking
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Track request performance."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        performance_monitor.record_request(duration, success=True)
        
        # Add performance headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = str(duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        performance_monitor.record_request(duration, success=False)
        raise


# Dependency injection examples

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "FastAPI Dependency Injection Demo",
        "version": "1.0.0",
        "features": [
            "Dependency Injection",
            "State Management",
            "Resource Lifecycle",
            "Performance Monitoring",
            "Error Handling"
        ]
    }


@app.get("/health")
async def health_check(
    stats: Dict[str, Any] = Depends(get_stats_dependency),
    config: DependencyConfig = Depends(get_config)
):
    """Health check with dependency injection."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "config": {
            "default_strategy": config.default_strategy.value,
            "default_batch_size": config.default_batch_size,
            "enable_monitoring": config.enable_monitoring
        },
        "stats": stats
    }


# Product endpoints with dependency injection

@app.get("/products/{product_id}")
async def get_product(
    product_id: str,
    request: Request,
    service: LazyLoadingService = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get product using on-demand loading with dependency injection."""
    try:
        product = await service.get_product(product_id)
        
        return {
            "product": product,
            "request_id": get_request_id(request),
            "user_context": user_context,
            "cached": service.lazy_manager.get_loader("products_on_demand").stats.cache_hits > 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/products/batch")
async def get_products_batch(
    request: BatchProductRequest,
    background_tasks: BackgroundTasks,
    service: LazyLoadingService = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get multiple products using background loading."""
    try:
        products = await service.get_products_batch(request.product_ids)
        
        # Add background task for cleanup
        background_tasks.add_task(cleanup_old_cache, service.lazy_manager)
        
        return {
            "products": products,
            "requested_count": len(request.product_ids),
            "loaded_count": len(products),
            "user_context": user_context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# User endpoints with dependency injection

@app.get("/users")
async def get_users(
    page: int = Query(0, ge=0),
    page_size: int = Query(50, ge=1, le=1000),
    service: LazyLoadingService = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get users using paginated loading."""
    try:
        result = await service.get_users_paginated(page, page_size)
        
        return {
            **result,
            "user_context": user_context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/users/cursor")
async def get_users_cursor(
    cursor: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    service: LazyLoadingService = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get users using cursor-based loading."""
    try:
        result = await service.get_users_cursor(cursor, limit)
        
        return {
            **result,
            "user_context": user_context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Item endpoints with dependency injection

@app.get("/items/stream")
async def get_items_stream(
    limit: int = Query(100, ge=1, le=1000),
    service: LazyLoadingService = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get items using streaming loading."""
    try:
        items = await service.get_items_streaming(limit)
        
        return {
            "items": items,
            "count": len(items),
            "limit": limit,
            "user_context": user_context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/items/window")
async def get_items_window(
    start: int = Query(0, ge=0),
    size: int = Query(100, ge=1, le=1000),
    service: LazyLoadingService = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get items using windowed loading."""
    try:
        result = await service.get_items_window(start, size)
        
        return {
            **result,
            "user_context": user_context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/items/window/slide")
async def slide_items_window(
    direction: str = Query("forward", regex="^(forward|backward)$"),
    size: int = Query(50, ge=1, le=1000),
    service: LazyLoadingService = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Slide the items window."""
    try:
        loader = service.lazy_manager.get_loader("items_windowed")
        items = await loader.slide_window(direction, size)
        
        return {
            "items": items,
            "direction": direction,
            "size": size,
            "count": len(items),
            "user_context": user_context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# System monitoring endpoints

@app.get("/stats/system")
async def get_system_stats(
    service: LazyLoadingService = Depends(),
    performance_stats: Dict[str, Any] = Depends(lambda: performance_monitor.get_stats())
):
    """Get comprehensive system statistics."""
    try:
        lazy_stats = await service.get_system_stats()
        
        return SystemStatsResponse(
            loaders=lazy_stats,
            memory_usage=lazy_stats.get("memory", {}),
            performance=performance_stats,
            uptime=time.time() - lazy_stats.get("start_time", time.time())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/stats/performance")
async def get_performance_stats():
    """Get performance statistics."""
    return performance_monitor.get_stats()


@app.get("/stats/loaders")
async def get_loader_stats(
    lazy_manager = Depends(get_lazy_manager_dependency)
):
    """Get loader-specific statistics."""
    return lazy_manager.get_stats()


# Configuration endpoints

@app.get("/config")
async def get_configuration(
    config: DependencyConfig = Depends(get_config)
):
    """Get current configuration."""
    return config.dict()


@app.post("/config/reload")
async def reload_configuration(
    background_tasks: BackgroundTasks,
    dependency_manager = Depends(lambda: get_dependency_manager())
):
    """Reload configuration (admin only)."""
    try:
        # In a real application, you would reload from external source
        background_tasks.add_task(dependency_manager._cleanup_resources)
        
        return {
            "message": "Configuration reload initiated",
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")


# Utility functions

async def cleanup_old_cache(lazy_manager) -> Any:
    """Background task to cleanup old cache entries."""
    try:
        # Simulate cleanup
        await asyncio.sleep(1)
        # logger.info("Cache cleanup completed") # Assuming logger is available
    except Exception as e:
        # logger.error(f"Cache cleanup failed: {e}") # Assuming logger is available
        pass


# Error handling with dependencies

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError with dependency injection."""
    performance_monitor.record_request(0, success=False)
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "detail": str(exc),
            "request_id": get_request_id(request),
            "timestamp": time.time()
        }
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Handle RuntimeError with dependency injection."""
    performance_monitor.record_request(0, success=False)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "request_id": get_request_id(request),
            "timestamp": time.time()
        }
    )


# Testing endpoints

@app.get("/test/dependencies")
async def test_dependencies(
    config: DependencyConfig = Depends(get_config),
    lazy_manager = Depends(get_lazy_manager_dependency),
    stats: Dict[str, Any] = Depends(get_stats_dependency),
    request_id: str = Depends(get_request_id)
):
    """Test dependency injection system."""
    return {
        "config": config.dict(),
        "lazy_manager_initialized": lazy_manager is not None,
        "stats": stats,
        "request_id": request_id,
        "timestamp": time.time()
    }


@app.get("/test/loaders")
async def test_loaders(
    products_loader = Depends(get_loader_dependency("products_on_demand")),
    users_loader = Depends(get_loader_dependency("users_paginated")),
    items_loader = Depends(get_loader_dependency("items_streaming"))
):
    """Test loader dependencies."""
    return {
        "products_loader": type(products_loader).__name__,
        "users_loader": type(users_loader).__name__,
        "items_loader": type(items_loader).__name__,
        "all_initialized": all([
            products_loader is not None,
            users_loader is not None,
            items_loader is not None
        ])
    }


# Demo functions

async def demonstrate_dependency_injection():
    """Demonstrate dependency injection features."""
    print("\n=== FastAPI Dependency Injection Demo ===")
    
    # Create test configuration
    config = DependencyConfig(
        default_strategy=LoadingStrategy.ON_DEMAND,
        default_batch_size=50,
        default_cache_ttl=180,
        enable_monitoring=True,
        enable_cleanup=True
    )
    
    # Create dependency manager
    async with TestDependencyManager(config) as manager:
        print("1. Testing dependency manager initialization...")
        assert manager.state.is_initialized
        assert manager.state.lazy_manager is not None
        print("   ✅ Dependency manager initialized successfully")
        
        # Test loaders
        print("\n2. Testing loader dependencies...")
        products_loader = manager.get_loader("products_on_demand")
        users_loader = manager.get_loader("users_paginated")
        
        assert products_loader is not None
        assert users_loader is not None
        print("   ✅ Loaders created successfully")
        
        # Test data sources
        print("\n3. Testing data source dependencies...")
        products_source = manager.get_data_source("products")
        users_source = manager.get_data_source("users")
        
        assert products_source is not None
        assert users_source is not None
        print("   ✅ Data sources created successfully")
        
        # Test lazy loading
        print("\n4. Testing lazy loading with dependencies...")
        product = await products_loader.get_item("prod_0000")
        assert product is not None
        print(f"   ✅ Loaded product: {product['title']}")
        
        # Test statistics
        print("\n5. Testing statistics with dependencies...")
        stats = manager.get_stats()
        assert "loaders" in stats
        assert "data_sources" in stats
        print("   ✅ Statistics collected successfully")
        
        print("\n✅ Dependency injection demo completed!")


async def demonstrate_request_scoped_dependencies():
    """Demonstrate request-scoped dependencies."""
    print("\n=== Request-Scoped Dependencies Demo ===")
    
    # Simulate request context
    class MockRequest:
        def __init__(self) -> Any:
            self.state = type('State', (), {})()
            self.state.request_id = str(uuid.uuid4())
            self.state.user_id = "user_123"
            self.state.session_id = "session_456"
    
    request = MockRequest()
    
    print("1. Testing request ID generation...")
    request_id = get_request_id(request)
    assert request_id == request.state.request_id
    print(f"   ✅ Request ID: {request_id}")
    
    print("\n2. Testing user context...")
    user_context = get_user_context(request)
    assert user_context["user_id"] == "user_123"
    assert user_context["session_id"] == "session_456"
    print(f"   ✅ User context: {user_context}")
    
    print("\n3. Testing performance monitoring...")
    monitor = get_performance_monitor()
    monitor.record_request(0.5, success=True)
    monitor.record_request(1.2, success=False)
    
    stats = monitor.get_stats()
    assert stats["total_requests"] == 2
    assert stats["success_count"] == 1
    assert stats["error_count"] == 1
    print(f"   ✅ Performance stats: {stats}")
    
    print("\n✅ Request-scoped dependencies demo completed!")


async def demonstrate_configuration_dependencies():
    """Demonstrate configuration dependencies."""
    print("\n=== Configuration Dependencies Demo ===")
    
    print("1. Testing default configuration...")
    default_config = get_default_config()
    assert default_config.default_strategy == LoadingStrategy.ON_DEMAND
    assert default_config.default_batch_size == 100
    print("   ✅ Default configuration loaded")
    
    print("\n2. Testing custom configuration...")
    custom_config = get_custom_config(
        strategy=LoadingStrategy.PAGINATED,
        batch_size=200,
        cache_ttl=600
    )
    assert custom_config.default_strategy == LoadingStrategy.PAGINATED
    assert custom_config.default_batch_size == 200
    assert custom_config.default_cache_ttl == 600
    print("   ✅ Custom configuration created")
    
    print("\n3. Testing configuration caching...")
    # Second call should use cached version
    cached_config = get_default_config()
    assert cached_config is default_config
    print("   ✅ Configuration caching working")
    
    print("\n✅ Configuration dependencies demo completed!")


async def demonstrate_error_handling():
    """Demonstrate error handling with dependencies."""
    print("\n=== Error Handling Demo ===")
    
    async with TestDependencyManager() as manager:
        print("1. Testing invalid loader access...")
        try:
            manager.get_loader("nonexistent_loader")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"   ✅ Caught expected error: {e}")
        
        print("\n2. Testing invalid data source access...")
        try:
            manager.get_data_source("nonexistent_source")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"   ✅ Caught expected error: {e}")
        
        print("\n3. Testing uninitialized manager access...")
        try:
            get_lazy_manager_dependency()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            print(f"   ✅ Caught expected error: {e}")
        
        print("\n✅ Error handling demo completed!")


async def run_comprehensive_demo():
    """Run comprehensive dependency injection demo."""
    print("🚀 Starting FastAPI Dependency Injection Demo")
    print("=" * 60)
    
    try:
        await demonstrate_dependency_injection()
        await demonstrate_request_scoped_dependencies()
        await demonstrate_configuration_dependencies()
        await demonstrate_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All dependency injection demos completed successfully!")
        
        print("\n📋 Next Steps:")
        print("1. Run the FastAPI app: uvicorn fastapi_dependency_demo:app --reload")
        print("2. Test endpoints: http://localhost:8000/docs")
        print("3. Monitor performance: http://localhost:8000/stats/performance")
        print("4. Check system stats: http://localhost:8000/stats/system")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo()) 