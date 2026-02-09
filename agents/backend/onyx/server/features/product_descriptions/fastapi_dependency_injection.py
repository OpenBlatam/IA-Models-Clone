from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, TypeVar, Union
from functools import lru_cache
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from lazy_loading_manager import (
        import time
        import uuid
from typing import Any, List, Dict, Optional
"""
FastAPI Dependency Injection System for Lazy Loading

This module provides a comprehensive dependency injection system for managing
lazy loading state and shared resources with proper lifecycle management,
configuration, and testing support.
"""



    LazyLoadingConfig, LoadingStrategy, LoadingStats,
    OnDemandLoader, PaginatedLoader, StreamingLoader, BackgroundLoader,
    CursorBasedLoader, WindowedLoader, LazyLoadingManager,
    MockDataSource, get_lazy_loading_manager, close_lazy_loading_manager
)

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')


class DependencyConfig(BaseModel):
    """Configuration for dependency injection system."""
    
    # Lazy loading configurations
    default_strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND
    default_batch_size: int = 100
    default_cache_ttl: int = 300
    default_max_memory: int = 1024 * 1024 * 100  # 100MB
    
    # Resource management
    enable_cleanup: bool = True
    cleanup_interval: int = 60
    max_connections: int = 100
    
    # Monitoring
    enable_monitoring: bool = True
    enable_metrics: bool = True
    
    # Performance
    connection_timeout: float = 30.0
    request_timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class ResourceState(BaseModel):
    """State management for shared resources."""
    
    lazy_manager: Optional[LazyLoadingManager] = None
    data_sources: Dict[str, Any] = Field(default_factory=dict)
    loaders: Dict[str, Any] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)
    is_initialized: bool = False
    is_shutting_down: bool = False


class DependencyManager:
    """
    Central dependency manager for FastAPI application.
    
    Manages the lifecycle of shared resources and provides
    dependency injection functions for FastAPI routes.
    """
    
    def __init__(self, config: DependencyConfig):
        """
        Initialize dependency manager.
        
        Args:
            config: Configuration for dependency injection
        """
        self.config = config
        self.state = ResourceState()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialization_lock = asyncio.Lock()
        
        logger.info(f"Initialized DependencyManager with config: {config}")
    
    async def initialize(self) -> None:
        """Initialize all shared resources."""
        async with self._initialization_lock:
            if self.state.is_initialized:
                return
            
            try:
                logger.info("Initializing dependency manager...")
                
                # Initialize lazy loading manager
                self.state.lazy_manager = get_lazy_loading_manager()
                
                # Initialize data sources
                await self._initialize_data_sources()
                
                # Initialize loaders
                await self._initialize_loaders()
                
                # Start cleanup task if enabled
                if self.config.enable_cleanup:
                    self._start_cleanup_task()
                
                self.state.is_initialized = True
                logger.info("Dependency manager initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize dependency manager: {e}")
                raise
    
    async def _initialize_data_sources(self) -> None:
        """Initialize data sources."""
        logger.info("Initializing data sources...")
        
        # Initialize mock data sources for demo
        self.state.data_sources["products"] = MockDataSource("products")
        self.state.data_sources["users"] = MockDataSource("users")
        self.state.data_sources["items"] = MockDataSource("items")
        
        logger.info(f"Initialized {len(self.state.data_sources)} data sources")
    
    async def _initialize_loaders(self) -> None:
        """Initialize lazy loaders."""
        logger.info("Initializing lazy loaders...")
        
        if not self.state.lazy_manager:
            raise RuntimeError("Lazy manager not initialized")
        
        # Create configurations for different strategies
        on_demand_config = LazyLoadingConfig(
            strategy=LoadingStrategy.ON_DEMAND,
            batch_size=self.config.default_batch_size,
            cache_ttl=self.config.default_cache_ttl,
            max_memory=self.config.default_max_memory,
            enable_cleanup=self.config.enable_cleanup,
            cleanup_interval=self.config.cleanup_interval
        )
        
        paginated_config = LazyLoadingConfig(
            strategy=LoadingStrategy.PAGINATED,
            batch_size=self.config.default_batch_size,
            cache_ttl=self.config.default_cache_ttl,
            max_memory=self.config.default_max_memory,
            enable_cleanup=self.config.enable_cleanup,
            cleanup_interval=self.config.cleanup_interval
        )
        
        streaming_config = LazyLoadingConfig(
            strategy=LoadingStrategy.STREAMING,
            window_size=200,
            cache_ttl=self.config.default_cache_ttl,
            max_memory=self.config.default_max_memory,
            enable_cleanup=self.config.enable_cleanup,
            cleanup_interval=self.config.cleanup_interval
        )
        
        background_config = LazyLoadingConfig(
            strategy=LoadingStrategy.BACKGROUND,
            batch_size=25,
            prefetch_size=50,
            cache_ttl=self.config.default_cache_ttl,
            max_memory=self.config.default_max_memory,
            enable_cleanup=self.config.enable_cleanup,
            cleanup_interval=self.config.cleanup_interval
        )
        
        cursor_config = LazyLoadingConfig(
            strategy=LoadingStrategy.CURSOR_BASED,
            batch_size=75,
            cache_ttl=self.config.default_cache_ttl,
            max_memory=self.config.default_max_memory,
            enable_cleanup=self.config.enable_cleanup,
            cleanup_interval=self.config.cleanup_interval
        )
        
        windowed_config = LazyLoadingConfig(
            strategy=LoadingStrategy.WINDOWED,
            window_size=150,
            batch_size=50,
            cache_ttl=self.config.default_cache_ttl,
            max_memory=self.config.default_max_memory,
            enable_cleanup=self.config.enable_cleanup,
            cleanup_interval=self.config.cleanup_interval
        )
        
        # Create loaders
        self.state.loaders["products_on_demand"] = OnDemandLoader(
            self.state.data_sources["products"], on_demand_config
        )
        
        self.state.loaders["users_paginated"] = PaginatedLoader(
            self.state.data_sources["users"], paginated_config
        )
        
        self.state.loaders["items_streaming"] = StreamingLoader(
            self.state.data_sources["items"], streaming_config
        )
        
        self.state.loaders["products_background"] = BackgroundLoader(
            self.state.data_sources["products"], background_config
        )
        
        self.state.loaders["users_cursor"] = CursorBasedLoader(
            self.state.data_sources["users"], cursor_config
        )
        
        self.state.loaders["items_windowed"] = WindowedLoader(
            self.state.data_sources["items"], windowed_config
        )
        
        # Register loaders with manager
        for name, loader in self.state.loaders.items():
            self.state.lazy_manager.register_loader(name, loader)
        
        logger.info(f"Initialized {len(self.state.loaders)} lazy loaders")
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started cleanup task")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self.state.is_shutting_down:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_resources()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_resources(self) -> None:
        """Cleanup expired resources."""
        if self.state.lazy_manager:
            # Update stats
            self.state.stats = self.state.lazy_manager.get_stats()
            
            # Log cleanup info
            logger.debug(f"Cleanup completed. Stats: {self.state.stats}")
    
    async def shutdown(self) -> None:
        """Shutdown dependency manager and cleanup resources."""
        logger.info("Shutting down dependency manager...")
        
        self.state.is_shutting_down = True
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close lazy loading manager
        if self.state.lazy_manager:
            await self.state.lazy_manager.close_all()
        
        # Clear state
        self.state.data_sources.clear()
        self.state.loaders.clear()
        self.state.stats.clear()
        
        logger.info("Dependency manager shutdown completed")
    
    def get_lazy_manager(self) -> LazyLoadingManager:
        """Get lazy loading manager."""
        if not self.state.lazy_manager:
            raise RuntimeError("Lazy manager not initialized")
        return self.state.lazy_manager
    
    def get_loader(self, name: str) -> Optional[Dict[str, Any]]:
        """Get lazy loader by name."""
        if name not in self.state.loaders:
            raise ValueError(f"Loader '{name}' not found")
        return self.state.loaders[name]
    
    def get_data_source(self, name: str) -> Optional[Dict[str, Any]]:
        """Get data source by name."""
        if name not in self.state.data_sources:
            raise ValueError(f"Data source '{name}' not found")
        return self.state.data_sources[name]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "state": self.state.dict(),
            "config": self.config.dict(),
            "loaders": list(self.state.loaders.keys()),
            "data_sources": list(self.state.data_sources.keys())
        }


# Global dependency manager instance
_dependency_manager: Optional[DependencyManager] = None


def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager instance."""
    global _dependency_manager
    if _dependency_manager is None:
        raise RuntimeError("Dependency manager not initialized")
    return _dependency_manager


def set_dependency_manager(manager: DependencyManager) -> None:
    """Set global dependency manager instance."""
    global _dependency_manager
    _dependency_manager = manager


# Dependency injection functions
def get_config() -> DependencyConfig:
    """Get dependency configuration."""
    return get_dependency_manager().config


def get_lazy_manager_dependency() -> LazyLoadingManager:
    """Get lazy loading manager dependency."""
    return get_dependency_manager().get_lazy_manager()


def get_loader_dependency(loader_name: str):
    """Get specific loader dependency."""
    return get_dependency_manager().get_loader(loader_name)


def get_data_source_dependency(source_name: str):
    """Get specific data source dependency."""
    return get_dependency_manager().get_data_source(source_name)


def get_stats_dependency() -> Dict[str, Any]:
    """Get statistics dependency."""
    return get_dependency_manager().get_stats()


# Specific loader dependencies
def get_products_on_demand_loader():
    """Get products on-demand loader."""
    return get_loader_dependency("products_on_demand")


def get_users_paginated_loader():
    """Get users paginated loader."""
    return get_loader_dependency("users_paginated")


def get_items_streaming_loader():
    """Get items streaming loader."""
    return get_loader_dependency("items_streaming")


def get_products_background_loader():
    """Get products background loader."""
    return get_loader_dependency("products_background")


def get_users_cursor_loader():
    """Get users cursor-based loader."""
    return get_loader_dependency("users_cursor")


def get_items_windowed_loader():
    """Get items windowed loader."""
    return get_loader_dependency("items_windowed")


# Configuration dependencies
@lru_cache()
def get_default_config() -> DependencyConfig:
    """Get default configuration (cached)."""
    return DependencyConfig()


def get_custom_config(
    strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND,
    batch_size: int = 100,
    cache_ttl: int = 300
) -> DependencyConfig:
    """Get custom configuration."""
    return DependencyConfig(
        default_strategy=strategy,
        default_batch_size=batch_size,
        default_cache_ttl=cache_ttl
    )


# Request-scoped dependencies
async def get_request_id(request: Request) -> str:
    """Get unique request ID."""
    return getattr(request.state, "request_id", "unknown")


def get_user_context(request: Request) -> Dict[str, Any]:
    """Get user context from request."""
    return {
        "user_id": getattr(request.state, "user_id", None),
        "session_id": getattr(request.state, "session_id", None),
        "request_id": get_request_id(request)
    }


# Error handling dependencies
def get_error_handler():
    """Get error handler dependency."""
    async def handle_errors(exc: Exception) -> JSONResponse:
        """Handle exceptions and return appropriate responses."""
        if isinstance(exc, ValueError):
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "detail": str(exc)}
            )
        elif isinstance(exc, RuntimeError):
            return JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error", "detail": str(exc)}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Unknown Error", "detail": str(exc)}
            )
    
    return handle_errors


# Performance monitoring dependencies
def get_performance_monitor():
    """Get performance monitor dependency."""
    class PerformanceMonitor:
        def __init__(self) -> Any:
            self.request_times: List[float] = []
            self.error_count = 0
            self.success_count = 0
        
        def record_request(self, duration: float, success: bool = True):
            """Record request performance."""
            self.request_times.append(duration)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
        
        def get_stats(self) -> Dict[str, Any]:
            """Get performance statistics."""
            if not self.request_times:
                return {"error": "No requests recorded"}
            
            return {
                "total_requests": len(self.request_times),
                "success_count": self.success_count,
                "error_count": self.error_count,
                "avg_response_time": sum(self.request_times) / len(self.request_times),
                "min_response_time": min(self.request_times),
                "max_response_time": max(self.request_times)
            }
    
    return PerformanceMonitor()


# FastAPI application factory
def create_app(config: Optional[DependencyConfig] = None) -> FastAPI:
    """Create FastAPI application with dependency injection."""
    
    # Use provided config or default
    if config is None:
        config = get_default_config()
    
    # Create dependency manager
    dependency_manager = DependencyManager(config)
    set_dependency_manager(dependency_manager)
    
    # Create FastAPI app with lifespan management
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        
    """lifespan function."""
# Startup
        await dependency_manager.initialize()
        yield
        # Shutdown
        await dependency_manager.shutdown()
    
    app = FastAPI(
        title="Lazy Loading API with Dependency Injection",
        description="Advanced lazy loading system with FastAPI dependency injection",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add middleware for request tracking
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        """Add request context and performance monitoring."""
        
        # Generate request ID
        request.state.request_id = str(uuid.uuid4())
        request.state.start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Record performance
        duration = time.time() - request.state.start_time
        logger.info(f"Request {request.state.request_id} completed in {duration:.3f}s")
        
        return response
    
    # Add exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        
    """value_error_handler function."""
return JSONResponse(
            status_code=400,
            content={"error": "Bad Request", "detail": str(exc)}
        )
    
    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        
    """runtime_error_handler function."""
return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": str(exc)}
        )
    
    return app


# Service layer with dependency injection
class LazyLoadingService:
    """Service layer using dependency injection."""
    
    def __init__(
        self,
        lazy_manager: LazyLoadingManager = Depends(get_lazy_manager_dependency),
        config: DependencyConfig = Depends(get_config)
    ):
        self.lazy_manager = lazy_manager
        self.config = config
    
    async def get_product(
        self,
        product_id: str,
        loader = Depends(get_products_on_demand_loader)
    ) -> Dict[str, Any]:
        """Get product using on-demand loading."""
        try:
            data = await loader.get_item(product_id)
            if data is None:
                raise HTTPException(status_code=404, detail="Product not found")
            return data
        except Exception as e:
            logger.error(f"Failed to get product {product_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to load product")
    
    async def get_users_paginated(
        self,
        page: int = 0,
        page_size: int = 50,
        loader = Depends(get_users_paginated_loader)
    ) -> Dict[str, Any]:
        """Get users using paginated loading."""
        try:
            users = await loader.get_page(page, page_size)
            total_count = await loader.get_total_count()
            
            return {
                "users": users,
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "has_more": (page + 1) * page_size < total_count
            }
        except Exception as e:
            logger.error(f"Failed to get users page {page}: {e}")
            raise HTTPException(status_code=500, detail="Failed to load users")
    
    async def get_items_streaming(
        self,
        limit: int = 100,
        loader = Depends(get_items_streaming_loader)
    ) -> List[Dict[str, Any]]:
        """Get items using streaming loading."""
        try:
            await loader.start_streaming()
            
            items = []
            for _ in range(limit):
                item = await loader.get_next_item()
                if item is None:
                    break
                items.append(item)
            
            return items
        except Exception as e:
            logger.error(f"Failed to stream items: {e}")
            raise HTTPException(status_code=500, detail="Failed to stream items")
    
    async def get_products_batch(
        self,
        product_ids: List[str],
        loader = Depends(get_products_background_loader)
    ) -> List[Dict[str, Any]]:
        """Get products using background loading."""
        try:
            await loader.start_background_loading(product_ids)
            
            products = []
            for product_id in product_ids:
                try:
                    data = await loader.get_item(product_id)
                    if data:
                        products.append(data)
                except Exception as e:
                    logger.warning(f"Failed to load product {product_id}: {e}")
            
            return products
        except Exception as e:
            logger.error(f"Failed to load products batch: {e}")
            raise HTTPException(status_code=500, detail="Failed to load products")
    
    async def get_users_cursor(
        self,
        cursor: Optional[int] = None,
        limit: int = 50,
        loader = Depends(get_users_cursor_loader)
    ) -> Dict[str, Any]:
        """Get users using cursor-based loading."""
        try:
            result = await loader.get_page_with_cursor(cursor, limit)
            return result
        except Exception as e:
            logger.error(f"Failed to get users with cursor: {e}")
            raise HTTPException(status_code=500, detail="Failed to load users")
    
    async def get_items_window(
        self,
        start: int = 0,
        size: int = 100,
        loader = Depends(get_items_windowed_loader)
    ) -> Dict[str, Any]:
        """Get items using windowed loading."""
        try:
            items = await loader.get_window(start, size)
            return {
                "items": items,
                "start": start,
                "size": size,
                "count": len(items)
            }
        except Exception as e:
            logger.error(f"Failed to get items window: {e}")
            raise HTTPException(status_code=500, detail="Failed to load items")
    
    async def get_system_stats(
        self,
        lazy_manager: LazyLoadingManager = Depends(get_lazy_manager_dependency)
    ) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            return lazy_manager.get_stats()
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            raise HTTPException(status_code=500, detail="Failed to get statistics")


# Testing utilities
class TestDependencyManager:
    """Test dependency manager for unit testing."""
    
    def __init__(self, config: Optional[DependencyConfig] = None):
        
    """__init__ function."""
self.config = config or DependencyConfig()
        self.manager = DependencyManager(self.config)
    
    async def __aenter__(self) -> Any:
        """Async context manager entry."""
        await self.manager.initialize()
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit."""
        await self.manager.shutdown()


def get_test_dependencies():
    """Get test dependencies for unit testing."""
    config = DependencyConfig(
        default_strategy=LoadingStrategy.ON_DEMAND,
        default_batch_size=10,
        default_cache_ttl=60,
        enable_cleanup=False
    )
    
    return TestDependencyManager(config)


# Example usage
if __name__ == "__main__":
    # Create app with dependency injection
    app = create_app()
    
    # Add routes (these would be in a separate router file)
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "Lazy Loading API with Dependency Injection"}
    
    @app.get("/health")
    async def health_check(
        stats: Dict[str, Any] = Depends(get_stats_dependency)
    ):
        return {"status": "healthy", "stats": stats}
    
    print("FastAPI app created with dependency injection!")
    print("Run with: uvicorn fastapi_dependency_injection:app --reload") 