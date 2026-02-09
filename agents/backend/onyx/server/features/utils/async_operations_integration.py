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
import logging
import functools
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TypeVar, Generic, Awaitable, Type
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import weakref
import contextlib
import structlog
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
from .async_database_operations import AsyncDatabaseManager, DatabaseConfig, DatabaseType
from .async_api_client import AsyncAPIClient, APIConfig, ClientType, AuthType
                import psutil
from typing import Any, List, Dict, Optional
"""
🔗 Async Operations Integration
==============================

Integration module for async database and API operations with FastAPI:
- Dependency injection for async operations
- Middleware for async request handling
- Route decorators for async operations
- Background task integration
- Error handling and logging
- Performance monitoring
- Health checks
- Configuration management
- Service layer patterns
- Repository pattern implementation
"""




logger = structlog.get_logger(__name__)

T = TypeVar('T')
ModelT = TypeVar('ModelT', bound=BaseModel)

class ServiceType(Enum):
    """Service types"""
    DATABASE = "database"
    API = "api"
    CACHE = "cache"
    BACKGROUND = "background"
    MONITORING = "monitoring"

@dataclass
class IntegrationConfig:
    """Integration configuration"""
    database_config: DatabaseConfig
    api_configs: Dict[str, APIConfig] = field(default_factory=dict)
    enable_health_checks: bool = True
    enable_metrics: bool = True
    enable_caching: bool = True
    enable_background_tasks: bool = True
    enable_monitoring: bool = True
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    gzip_enabled: bool = True

class AsyncOperationsManager:
    """Main async operations manager"""
    
    def __init__(self, config: IntegrationConfig):
        
    """__init__ function."""
self.config = config
        self.database_manager = None
        self.api_clients: Dict[str, AsyncAPIClient] = {}
        self.redis_client = None
        self.background_tasks = []
        self.health_status = {"status": "unknown", "last_check": None}
        
        # Performance tracking
        self.operation_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.request_history: deque = deque(maxlen=10000)
        
        logger.info("Async Operations Manager initialized")
    
    async def initialize(self) -> Any:
        """Initialize all async operations"""
        try:
            # Initialize database manager
            self.database_manager = AsyncDatabaseManager(self.config.database_config)
            await self.database_manager.initialize()
            
            # Initialize API clients
            for name, api_config in self.config.api_configs.items():
                api_client = AsyncAPIClient(api_config)
                await api_client.initialize()
                self.api_clients[name] = api_client
            
            # Initialize Redis for caching
            if self.config.enable_caching:
                self.redis_client = redis.from_url("redis://localhost:6379")
                await self.redis_client.ping()
            
            # Start background tasks
            if self.config.enable_background_tasks:
                await self._start_background_tasks()
            
            # Start monitoring
            if self.config.enable_monitoring:
                await self._start_monitoring()
            
            logger.info("Async Operations Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Async Operations Manager: {e}")
            raise
    
    async def _start_background_tasks(self) -> Any:
        """Start background tasks"""
        # Cache cleanup task
        asyncio.create_task(self._cache_cleanup_task())
        
        # Metrics aggregation task
        asyncio.create_task(self._metrics_aggregation_task())
        
        # Health check task
        if self.config.enable_health_checks:
            asyncio.create_task(self._health_check_task())
    
    async def _start_monitoring(self) -> Any:
        """Start monitoring tasks"""
        # Performance monitoring
        asyncio.create_task(self._performance_monitoring_task())
        
        # Resource monitoring
        asyncio.create_task(self._resource_monitoring_task())
    
    async def _cache_cleanup_task(self) -> Any:
        """Background cache cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                if self.redis_client:
                    # Clean up expired keys
                    await self.redis_client.eval("""
                        local keys = redis.call('keys', 'cache:*')
                        for i, key in ipairs(keys) do
                            if redis.call('ttl', key) == -1 then
                                redis.call('del', key)
                            end
                        end
                    """, 0)
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _metrics_aggregation_task(self) -> Any:
        """Background metrics aggregation task"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Aggregate database metrics
                if self.database_manager:
                    db_metrics = self.database_manager.get_performance_metrics()
                    self.operation_metrics["database"] = db_metrics
                
                # Aggregate API metrics
                api_metrics = {}
                for name, client in self.api_clients.items():
                    api_metrics[name] = client.get_performance_summary()
                self.operation_metrics["api"] = api_metrics
                
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
    
    async def _health_check_task(self) -> Any:
        """Background health check task"""
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                health_status = await self._perform_health_checks()
                self.health_status = health_status
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                self.health_status = {
                    "status": "error",
                    "last_check": datetime.now(),
                    "error": str(e)
                }
    
    async def _performance_monitoring_task(self) -> Any:
        """Background performance monitoring task"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Monitor database performance
                if self.database_manager:
                    db_metrics = self.database_manager.get_performance_metrics()
                    if db_metrics.get("avg_execution_time", 0) > 1.0:
                        logger.warning("Database performance degradation detected")
                
                # Monitor API performance
                for name, client in self.api_clients.items():
                    api_metrics = client.get_performance_summary()
                    if api_metrics.get("overall", {}).get("avg_response_time", 0) > 2.0:
                        logger.warning(f"API {name} performance degradation detected")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _resource_monitoring_task(self) -> Any:
        """Background resource monitoring task"""
        while True:
            try:
                await asyncio.sleep(120)  # Every 2 minutes
                
                # Monitor memory usage
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 80:
                    logger.warning(f"High memory usage: {memory_usage}%")
                
                # Monitor disk usage
                disk_usage = psutil.disk_usage('/').percent
                if disk_usage > 90:
                    logger.warning(f"High disk usage: {disk_usage}%")
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform health checks"""
        checks = {
            "database": False,
            "api_clients": {},
            "redis": False,
            "overall": False
        }
        
        # Database health check
        try:
            if self.database_manager:
                # Simple query to test database
                await self.database_manager.select_one("sqlite_master", {"type": "table"})
                checks["database"] = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        # API clients health check
        for name, client in self.api_clients.items():
            try:
                # Simple GET request to test API
                response = await client.get("/health")
                checks["api_clients"][name] = response.status_code == 200
            except Exception as e:
                logger.error(f"API {name} health check failed: {e}")
                checks["api_clients"][name] = False
        
        # Redis health check
        try:
            if self.redis_client:
                await self.redis_client.ping()
                checks["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        # Overall health
        checks["overall"] = (
            checks["database"] and 
            all(checks["api_clients"].values()) and 
            checks["redis"]
        )
        
        return {
            "status": "healthy" if checks["overall"] else "unhealthy",
            "last_check": datetime.now(),
            "checks": checks
        }
    
    def get_database_manager(self) -> AsyncDatabaseManager:
        """Get database manager"""
        if not self.database_manager:
            raise RuntimeError("Database manager not initialized")
        return self.database_manager
    
    async def get_api_client(self, name: str) -> AsyncAPIClient:
        """Get API client by name"""
        if name not in self.api_clients:
            raise ValueError(f"API client '{name}' not found")
        return self.api_clients[name]
    
    def get_redis_client(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        return self.redis_client
    
    async def cleanup(self) -> Any:
        """Cleanup all resources"""
        try:
            # Cleanup database manager
            if self.database_manager:
                await self.database_manager.cleanup()
            
            # Cleanup API clients
            for client in self.api_clients.values():
                await client.cleanup()
            
            # Cleanup Redis client
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Async Operations Manager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# FastAPI Integration

def create_async_app(config: IntegrationConfig) -> FastAPI:
    """Create FastAPI app with async operations integration"""
    app = FastAPI(
        title="Async Operations API",
        description="API with integrated async database and external API operations",
        version="1.0.0"
    )
    
    # Create operations manager
    operations_manager = AsyncOperationsManager(config)
    
    # Store in app state
    app.state.operations_manager = operations_manager
    
    # Add middleware
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    if config.gzip_enabled:
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add async middleware
    @app.middleware("http")
    async def async_middleware(request: Request, call_next):
        
    """async_middleware function."""
start_time = time.time()
        
        # Add request to history
        operations_manager.request_history.append({
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now(),
            "start_time": start_time
        })
        
        try:
            response = await call_next(request)
            
            # Record response time
            response_time = time.time() - start_time
            operations_manager.request_history[-1]["response_time"] = response_time
            operations_manager.request_history[-1]["status_code"] = response.status_code
            
            return response
            
        except Exception as e:
            # Record error
            operations_manager.request_history[-1]["error"] = str(e)
            raise
    
    # Setup startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        
    """startup_event function."""
await operations_manager.initialize()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        
    """shutdown_event function."""
await operations_manager.cleanup()
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": operations_manager.health_status["status"],
            "last_check": operations_manager.health_status["last_check"],
            "checks": operations_manager.health_status.get("checks", {})
        }
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def get_metrics():
        """Get performance metrics"""
        return {
            "database": operations_manager.operation_metrics.get("database", {}),
            "api": operations_manager.operation_metrics.get("api", {}),
            "requests": {
                "total": len(operations_manager.request_history),
                "recent": [
                    {
                        "path": req["path"],
                        "method": req["method"],
                        "response_time": req.get("response_time", 0),
                        "status_code": req.get("status_code", 0),
                        "timestamp": req["timestamp"].isoformat()
                    }
                    for req in list(operations_manager.request_history)[-10:]
                ]
            }
        }
    
    return app

# Dependency Injection

async def get_database_manager() -> AsyncDatabaseManager:
    """Dependency to get database manager"""
    # This would be injected from FastAPI app state
    # For now, we'll create a mock implementation
    config = DatabaseConfig(
        database_type=DatabaseType.SQLITE,
        connection_string=":memory:"
    )
    manager = AsyncDatabaseManager(config)
    await manager.initialize()
    return manager

async async def get_api_client(name: str = "default") -> AsyncAPIClient:
    """Dependency to get API client"""
    # This would be injected from FastAPI app state
    # For now, we'll create a mock implementation
    config = APIConfig(
        base_url="https://api.example.com",
        client_type=ClientType.AIOHTTP
    )
    client = AsyncAPIClient(config)
    await client.initialize()
    return client

async def get_redis_client() -> redis.Redis:
    """Dependency to get Redis client"""
    # This would be injected from FastAPI app state
    client = redis.from_url("redis://localhost:6379")
    await client.ping()
    return client

# Repository Pattern

class AsyncRepository:
    """Base async repository class"""
    
    def __init__(self, database_manager: AsyncDatabaseManager, table_name: str):
        
    """__init__ function."""
self.database_manager = database_manager
        self.table_name = table_name
    
    async def find_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        """Find record by ID"""
        return await self.database_manager.select_one(
            self.table_name, 
            {"id": id},
            cache_key=f"{self.table_name}:{id}"
        )
    
    async def find_all(self, conditions: Dict[str, Any] = None, 
                      limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find all records"""
        return await self.database_manager.select_many(
            self.table_name,
            conditions,
            limit,
            offset
        )
    
    async def create(self, data: Dict[str, Any]) -> int:
        """Create new record"""
        return await self.database_manager.insert_one(self.table_name, data)
    
    async def update(self, id: int, data: Dict[str, Any]) -> int:
        """Update record"""
        return await self.database_manager.update_one(
            self.table_name,
            {"id": id},
            data
        )
    
    async def delete(self, id: int) -> int:
        """Delete record"""
        return await self.database_manager.delete_one(self.table_name, {"id": id})
    
    async def count(self, conditions: Dict[str, Any] = None) -> int:
        """Count records"""
        # This would need to be implemented based on the specific database
        records = await self.find_all(conditions, limit=1000000)
        return len(records)

# Service Layer Pattern

class AsyncService:
    """Base async service class"""
    
    def __init__(self, database_manager: AsyncDatabaseManager, 
                 api_clients: Dict[str, AsyncAPIClient] = None):
        
    """__init__ function."""
self.database_manager = database_manager
        self.api_clients = api_clients or {}
    
    async def process_with_database(self, operation: Callable, *args, **kwargs):
        """Process operation with database"""
        return await operation(self.database_manager, *args, **kwargs)
    
    async def process_with_api(self, client_name: str, operation: Callable, *args, **kwargs):
        """Process operation with API client"""
        if client_name not in self.api_clients:
            raise ValueError(f"API client '{client_name}' not found")
        
        return await operation(self.api_clients[client_name], *args, **kwargs)
    
    async def process_with_cache(self, cache_key: str, operation: Callable, *args, **kwargs):
        """Process operation with caching"""
        # This would integrate with Redis caching
        return await operation(*args, **kwargs)

# Route Decorators

def async_database_operation(table_name: str, operation: str = "select"):
    """Decorator for async database operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(database_manager: AsyncDatabaseManager = Depends(get_database_manager), 
                         *args, **kwargs):
            try:
                if operation == "select":
                    return await database_manager.select_one(table_name, kwargs)
                elif operation == "select_many":
                    return await database_manager.select_many(table_name, kwargs)
                elif operation == "insert":
                    return await database_manager.insert_one(table_name, kwargs)
                elif operation == "update":
                    return await database_manager.update_one(table_name, args[0], kwargs)
                elif operation == "delete":
                    return await database_manager.delete_one(table_name, kwargs)
                else:
                    return await func(database_manager, *args, **kwargs)
            except Exception as e:
                logger.error(f"Database operation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return wrapper
    return decorator

def async_api_operation(client_name: str = "default", method: str = "GET", 
                       endpoint: str = None, cache_key: str = None):
    """Decorator for async API operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(api_client: AsyncAPIClient = Depends(get_api_client), 
                         *args, **kwargs):
            try:
                if endpoint:
                    if method == "GET":
                        response = await api_client.get(endpoint, kwargs, cache_key=cache_key)
                    elif method == "POST":
                        response = await api_client.post(endpoint, kwargs)
                    elif method == "PUT":
                        response = await api_client.put(endpoint, kwargs)
                    elif method == "DELETE":
                        response = await api_client.delete(endpoint, kwargs)
                    else:
                        return await func(api_client, *args, **kwargs)
                    
                    if response.error:
                        raise HTTPException(status_code=response.status_code, detail=response.error)
                    
                    return response.data
                else:
                    return await func(api_client, *args, **kwargs)
            except Exception as e:
                logger.error(f"API operation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return wrapper
    return decorator

def async_background_task():
    """Decorator for async background tasks"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(background_tasks: BackgroundTasks, *args, **kwargs):
            
    """wrapper function."""
background_tasks.add_task(func, *args, **kwargs)
            return {"message": "Background task started"}
        
        return wrapper
    return decorator

# Example usage

async def example_integration():
    """Example usage of async operations integration"""
    
    # Create configuration
    database_config = DatabaseConfig(
        database_type=DatabaseType.SQLITE,
        connection_string=":memory:"
    )
    
    api_config = APIConfig(
        base_url="https://api.example.com",
        client_type=ClientType.AIOHTTP
    )
    
    config = IntegrationConfig(
        database_config=database_config,
        api_configs={"default": api_config},
        enable_health_checks=True,
        enable_metrics=True
    )
    
    # Create FastAPI app
    app = create_async_app(config)
    
    # Example routes
    @app.get("/users/{user_id}")
    @async_database_operation("users", "select")
    async def get_user(user_id: int):
        
    """get_user function."""
return {"user_id": user_id}
    
    @app.post("/users")
    @async_database_operation("users", "insert")
    async def create_user(user_data: Dict[str, Any]):
        
    """create_user function."""
return user_data
    
    @app.get("/external-data")
    @async_api_operation("default", "GET", "/data", cache_key="external_data")
    async def get_external_data():
        
    """get_external_data function."""
return {"data": "external"}
    
    @app.post("/background-task")
    @async_background_task()
    async def start_background_task():
        
    """start_background_task function."""
await asyncio.sleep(10)  # Simulate long-running task
        return {"status": "completed"}
    
    return app

if __name__ == "__main__":
    app = asyncio.run(example_integration())
    print("Async Operations Integration example created successfully") 