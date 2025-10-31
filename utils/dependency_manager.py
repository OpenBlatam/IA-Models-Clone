from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
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
from abc import ABC, abstractmethod
import structlog
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
            import aiohttp
from typing import Any, List, Dict, Optional
"""
🔗 Dependency Manager
====================

Comprehensive dependency management system for FastAPI:
- Centralized dependency organization and management
- Dependency scoping and lifecycle management
- Dependency injection patterns and best practices
- Dependency validation and error handling
- Dependency caching and performance optimization
- Dependency testing and mocking utilities
- Dependency documentation and type hints
- Dependency monitoring and metrics
- Dependency versioning and compatibility
- Dependency health checks and diagnostics
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')
DependencyT = TypeVar('DependencyT')

class DependencyScope(Enum):
    """Dependency scopes"""
    REQUEST = "request"
    SESSION = "session"
    APPLICATION = "application"
    DATABASE = "database"
    CACHE = "cache"
    AUTH = "auth"
    API = "api"
    BACKGROUND = "background"
    SINGLETON = "singleton"

class DependencyType(Enum):
    """Dependency types"""
    SERVICE = "service"
    REPOSITORY = "repository"
    CLIENT = "client"
    CONFIG = "config"
    UTILITY = "utility"
    MIDDLEWARE = "middleware"
    VALIDATOR = "validator"
    AUTHENTICATOR = "authenticator"
    AUTHORIZER = "authorizer"
    CACHE = "cache"

class DependencyStatus(Enum):
    """Dependency status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class DependencyConfig:
    """Dependency configuration"""
    name: str
    dependency: Callable
    scope: DependencyScope = DependencyScope.REQUEST
    type: DependencyType = DependencyType.SERVICE
    cache: bool = False
    cache_ttl: int = 300
    retry_attempts: int = 3
    timeout: float = 30.0
    required: bool = True
    description: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    validation_schema: Optional[Dict[str, Any]] = None
    health_check: Optional[Callable] = None
    cleanup: Optional[Callable] = None

@dataclass
class DependencyMetrics:
    """Dependency performance metrics"""
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    status: DependencyStatus = DependencyStatus.UNKNOWN

class DependencyContainer:
    """Dependency injection container"""
    
    def __init__(self) -> Any:
        self.dependencies: Dict[str, DependencyConfig] = {}
        self.instances: Dict[str, Any] = {}
        self.metrics: Dict[str, DependencyMetrics] = defaultdict(
            lambda: DependencyMetrics(name="")
        )
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        
        logger.info("Dependency Container initialized")
    
    def register(self, config: DependencyConfig) -> None:
        """Register a dependency"""
        self.dependencies[config.name] = config
        self.metrics[config.name] = DependencyMetrics(name=config.name)
        
        logger.info(f"Registered dependency: {config.name} ({config.type.value})")
    
    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dependency instance"""
        if name not in self.dependencies:
            raise ValueError(f"Dependency '{name}' not found")
        
        config = self.dependencies[name]
        
        # Check cache first
        if config.cache and name in self.cache:
            cached_value, cached_time = self.cache[name]
            if datetime.now() - cached_time < timedelta(seconds=config.cache_ttl):
                self.metrics[name].cache_hits += 1
                return cached_value
            else:
                del self.cache[name]
        
        # Create or get instance based on scope
        if config.scope == DependencyScope.SINGLETON:
            if name not in self.instances:
                self.instances[name] = self._create_instance(config)
            instance = self.instances[name]
        else:
            instance = self._create_instance(config)
        
        # Cache if enabled
        if config.cache:
            self.cache[name] = (instance, datetime.now())
            self.metrics[name].cache_misses += 1
        
        return instance
    
    def _create_instance(self, config: DependencyConfig) -> Any:
        """Create dependency instance"""
        start_time = time.time()
        
        try:
            # Execute dependency with retry logic
            for attempt in range(config.retry_attempts):
                try:
                    if asyncio.iscoroutinefunction(config.dependency):
                        # Handle async dependencies
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Create task for async dependency
                            instance = asyncio.create_task(config.dependency())
                        else:
                            instance = asyncio.run(config.dependency())
                    else:
                        instance = config.dependency()
                    
                    # Record success metrics
                    execution_time = time.time() - start_time
                    self._record_metrics(config.name, execution_time, True)
                    
                    return instance
                    
                except Exception as e:
                    if attempt == config.retry_attempts - 1:
                        # Record failure metrics
                        execution_time = time.time() - start_time
                        self._record_metrics(config.name, execution_time, False, str(e))
                        raise
                    else:
                        # Wait before retry
                        time.sleep(config.timeout * (2 ** attempt))
            
        except Exception as e:
            logger.error(f"Failed to create dependency '{config.name}': {e}")
            raise
    
    def _record_metrics(self, name: str, execution_time: float, success: bool, error: str = None):
        """Record dependency metrics"""
        metrics = self.metrics[name]
        metrics.total_calls += 1
        metrics.total_execution_time += execution_time
        metrics.min_execution_time = min(metrics.min_execution_time, execution_time)
        metrics.max_execution_time = max(metrics.max_execution_time, execution_time)
        metrics.last_updated = datetime.now()
        
        if success:
            metrics.successful_calls += 1
            metrics.status = DependencyStatus.HEALTHY
        else:
            metrics.failed_calls += 1
            metrics.status = DependencyStatus.UNHEALTHY
    
    def get_metrics(self, name: str = None) -> Dict[str, Any]:
        """Get dependency metrics"""
        if name:
            if name not in self.metrics:
                return {"error": f"Dependency '{name}' not found"}
            
            metrics = self.metrics[name]
            return {
                "name": metrics.name,
                "total_calls": metrics.total_calls,
                "successful_calls": metrics.successful_calls,
                "failed_calls": metrics.failed_calls,
                "success_rate": metrics.successful_calls / metrics.total_calls if metrics.total_calls > 0 else 0.0,
                "avg_execution_time": metrics.total_execution_time / metrics.total_calls if metrics.total_calls > 0 else 0.0,
                "min_execution_time": metrics.min_execution_time,
                "max_execution_time": metrics.max_execution_time,
                "cache_hits": metrics.cache_hits,
                "cache_misses": metrics.cache_misses,
                "cache_hit_rate": metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) if (metrics.cache_hits + metrics.cache_misses) > 0 else 0.0,
                "status": metrics.status.value,
                "last_updated": metrics.last_updated.isoformat()
            }
        
        # Return all metrics
        return {
            name: {
                "total_calls": metrics.total_calls,
                "success_rate": metrics.successful_calls / metrics.total_calls if metrics.total_calls > 0 else 0.0,
                "avg_execution_time": metrics.total_execution_time / metrics.total_calls if metrics.total_calls > 0 else 0.0,
                "status": metrics.status.value
            }
            for name, metrics in self.metrics.items()
        }
    
    def health_check(self, name: str = None) -> Dict[str, Any]:
        """Perform health check on dependencies"""
        if name:
            if name not in self.dependencies:
                return {"error": f"Dependency '{name}' not found"}
            
            config = self.dependencies[name]
            if config.health_check:
                try:
                    health_result = config.health_check()
                    return {"name": name, "status": "healthy", "result": health_result}
                except Exception as e:
                    return {"name": name, "status": "unhealthy", "error": str(e)}
            else:
                return {"name": name, "status": "unknown", "message": "No health check configured"}
        
        # Check all dependencies
        health_results = {}
        for dep_name, config in self.dependencies.items():
            if config.health_check:
                try:
                    health_result = config.health_check()
                    health_results[dep_name] = {"status": "healthy", "result": health_result}
                except Exception as e:
                    health_results[dep_name] = {"status": "unhealthy", "error": str(e)}
            else:
                health_results[dep_name] = {"status": "unknown", "message": "No health check configured"}
        
        return health_results
    
    def cleanup(self) -> None:
        """Cleanup all dependencies"""
        for name, config in self.dependencies.items():
            if config.cleanup:
                try:
                    config.cleanup()
                    logger.info(f"Cleaned up dependency: {name}")
                except Exception as e:
                    logger.error(f"Error cleaning up dependency '{name}': {e}")

class DependencyManager:
    """Main dependency manager"""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
        self.container = DependencyContainer()
        self.dependency_factories: Dict[str, Callable] = {}
        self.dependency_validators: Dict[str, Callable] = {}
        
        logger.info("Dependency Manager initialized")
    
    def register_dependency(self, config: DependencyConfig) -> None:
        """Register a dependency"""
        self.container.register(config)
        
        # Create FastAPI dependency function
        def dependency_func():
            
    """dependency_func function."""
return self.container.get(config.name)
        
        # Add to FastAPI dependency system
        setattr(self.app, f"get_{config.name}", dependency_func)
    
    def create_dependency(self, name: str, factory: Callable, **kwargs) -> DependencyConfig:
        """Create dependency configuration from factory"""
        config = DependencyConfig(
            name=name,
            dependency=factory,
            **kwargs
        )
        
        self.register_dependency(config)
        return config
    
    def get_dependency(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dependency instance"""
        return self.container.get(name)
    
    def add_factory(self, name: str, factory: Callable) -> None:
        """Add dependency factory"""
        self.dependency_factories[name] = factory
    
    def add_validator(self, name: str, validator: Callable) -> None:
        """Add dependency validator"""
        self.dependency_validators[name] = validator
    
    def validate_dependency(self, name: str, value: Any) -> bool:
        """Validate dependency value"""
        if name in self.dependency_validators:
            return self.dependency_validators[name](value)
        return True
    
    def get_metrics(self, name: str = None) -> Dict[str, Any]:
        """Get dependency metrics"""
        return self.container.get_metrics(name)
    
    def health_check(self, name: str = None) -> Dict[str, Any]:
        """Perform health check"""
        return self.container.health_check(name)
    
    def cleanup(self) -> None:
        """Cleanup all dependencies"""
        self.container.cleanup()

# Common Dependency Factories

class DatabaseDependencies:
    """Database-related dependencies"""
    
    @staticmethod
    def database_session() -> Callable:
        """Create database session dependency"""
        async def get_database_session():
            
    """get_database_session function."""
# This would create and return a database session
            # For now, return a mock session
            return {"session": "mock_database_session"}
        
        return get_database_session
    
    @staticmethod
    def database_connection() -> Callable:
        """Create database connection dependency"""
        async def get_database_connection():
            
    """get_database_connection function."""
# This would create and return a database connection
            return {"connection": "mock_database_connection"}
        
        return get_database_connection

class AuthDependencies:
    """Authentication-related dependencies"""
    
    @staticmethod
    def current_user() -> Callable:
        """Create current user dependency"""
        async def get_current_user(request: Request):
            
    """get_current_user function."""
# This would extract and validate user from request
            # For now, return a mock user
            return {"user_id": "mock_user_id", "email": "user@example.com"}
        
        return get_current_user
    
    @staticmethod
    async def api_key() -> Callable:
        """Create API key dependency"""
        async def get_api_key(request: Request):
            
    """get_api_key function."""
# This would extract and validate API key from request
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                raise HTTPException(status_code=401, detail="API key required")
            return api_key
        
        return get_api_key
    
    @staticmethod
    def jwt_token() -> Callable:
        """Create JWT token dependency"""
        async def get_jwt_token(request: Request):
            
    """get_jwt_token function."""
# This would extract and validate JWT token from request
            authorization = request.headers.get("Authorization")
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Bearer token required")
            return authorization.split(" ")[1]
        
        return get_jwt_token

class CacheDependencies:
    """Cache-related dependencies"""
    
    @staticmethod
    def redis_client() -> Callable:
        """Create Redis client dependency"""
        async def get_redis_client():
            
    """get_redis_client function."""
# This would create and return a Redis client
            return redis.from_url("redis://localhost:6379")
        
        return get_redis_client
    
    @staticmethod
    def memory_cache() -> Callable:
        """Create memory cache dependency"""
        async def get_memory_cache():
            
    """get_memory_cache function."""
# This would create and return a memory cache
            return {}
        
        return get_memory_cache

class APIDependencies:
    """API-related dependencies"""
    
    @staticmethod
    async def http_client() -> Callable:
        """Create HTTP client dependency"""
        async def get_http_client():
            
    """get_http_client function."""
# This would create and return an HTTP client
            return aiohttp.ClientSession()
        
        return get_http_client
    
    @staticmethod
    def rate_limiter() -> Callable:
        """Create rate limiter dependency"""
        async def get_rate_limiter():
            
    """get_rate_limiter function."""
# This would create and return a rate limiter
            return {"rate_limiter": "mock_rate_limiter"}
        
        return get_rate_limiter

class ServiceDependencies:
    """Service-related dependencies"""
    
    @staticmethod
    def user_service() -> Callable:
        """Create user service dependency"""
        async def get_user_service():
            
    """get_user_service function."""
# This would create and return a user service
            return {"service": "user_service"}
        
        return get_user_service
    
    @staticmethod
    def content_service() -> Callable:
        """Create content service dependency"""
        async def get_content_service():
            
    """get_content_service function."""
# This would create and return a content service
            return {"service": "content_service"}
        
        return get_content_service
    
    @staticmethod
    def analytics_service() -> Callable:
        """Create analytics service dependency"""
        async def get_analytics_service():
            
    """get_analytics_service function."""
# This would create and return an analytics service
            return {"service": "analytics_service"}
        
        return get_analytics_service

# Dependency Decorators

def dependency(name: str, scope: DependencyScope = DependencyScope.REQUEST, 
              cache: bool = False, retry_attempts: int = 3):
    """Decorator for creating dependencies"""
    def decorator(func: Callable) -> Callable:
        config = DependencyConfig(
            name=name,
            dependency=func,
            scope=scope,
            cache=cache,
            retry_attempts=retry_attempts
        )
        
        # Store configuration for later registration
        func._dependency_config = config
        
        return func
    return decorator

def singleton_dependency(name: str):
    """Decorator for creating singleton dependencies"""
    return dependency(name, scope=DependencyScope.SINGLETON)

def cached_dependency(name: str, ttl: int = 300):
    """Decorator for creating cached dependencies"""
    return dependency(name, cache=True, retry_attempts=1)

def required_dependency(name: str):
    """Decorator for creating required dependencies"""
    return dependency(name, scope=DependencyScope.REQUEST)

# Example usage

def create_dependency_manager(app: FastAPI) -> DependencyManager:
    """Create and configure dependency manager"""
    manager = DependencyManager(app)
    
    # Register database dependencies
    manager.create_dependency(
        "database_session",
        DatabaseDependencies.database_session(),
        scope=DependencyScope.REQUEST,
        type=DependencyType.DATABASE,
        description="Database session for data access"
    )
    
    manager.create_dependency(
        "database_connection",
        DatabaseDependencies.database_connection(),
        scope=DependencyScope.APPLICATION,
        type=DependencyType.DATABASE,
        description="Database connection pool"
    )
    
    # Register authentication dependencies
    manager.create_dependency(
        "current_user",
        AuthDependencies.current_user(),
        scope=DependencyScope.REQUEST,
        type=DependencyType.AUTHENTICATOR,
        cache=True,
        cache_ttl=300,
        description="Current authenticated user"
    )
    
    manager.create_dependency(
        "api_key",
        AuthDependencies.api_key(),
        scope=DependencyScope.REQUEST,
        type=DependencyType.AUTHENTICATOR,
        required=True,
        description="API key for authentication"
    )
    
    manager.create_dependency(
        "jwt_token",
        AuthDependencies.jwt_token(),
        scope=DependencyScope.REQUEST,
        type=DependencyType.AUTHENTICATOR,
        required=True,
        description="JWT token for authentication"
    )
    
    # Register cache dependencies
    manager.create_dependency(
        "redis_client",
        CacheDependencies.redis_client(),
        scope=DependencyScope.APPLICATION,
        type=DependencyType.CACHE,
        description="Redis client for caching"
    )
    
    manager.create_dependency(
        "memory_cache",
        CacheDependencies.memory_cache(),
        scope=DependencyScope.APPLICATION,
        type=DependencyType.CACHE,
        description="In-memory cache"
    )
    
    # Register API dependencies
    manager.create_dependency(
        "http_client",
        APIDependencies.http_client(),
        scope=DependencyScope.APPLICATION,
        type=DependencyType.CLIENT,
        description="HTTP client for external API calls"
    )
    
    manager.create_dependency(
        "rate_limiter",
        APIDependencies.rate_limiter(),
        scope=DependencyScope.REQUEST,
        type=DependencyType.UTILITY,
        description="Rate limiter for API requests"
    )
    
    # Register service dependencies
    manager.create_dependency(
        "user_service",
        ServiceDependencies.user_service(),
        scope=DependencyScope.REQUEST,
        type=DependencyType.SERVICE,
        description="User management service"
    )
    
    manager.create_dependency(
        "content_service",
        ServiceDependencies.content_service(),
        scope=DependencyScope.REQUEST,
        type=DependencyType.SERVICE,
        description="Content management service"
    )
    
    manager.create_dependency(
        "analytics_service",
        ServiceDependencies.analytics_service(),
        scope=DependencyScope.REQUEST,
        type=DependencyType.SERVICE,
        description="Analytics service"
    )
    
    return manager

# Example FastAPI app with dependency management

def create_app_with_dependencies() -> FastAPI:
    """Create FastAPI app with managed dependencies"""
    app = FastAPI(
        title="Dependency Managed API",
        description="API with comprehensive dependency management",
        version="1.0.0"
    )
    
    # Create dependency manager
    manager = create_dependency_manager(app)
    
    # Example routes using managed dependencies
    @app.get("/users/{user_id}")
    async def get_user(
        user_id: str,
        current_user: Dict = Depends(manager.get_dependency("current_user")),
        user_service: Dict = Depends(manager.get_dependency("user_service")),
        db_session: Dict = Depends(manager.get_dependency("database_session"))
    ):
        return {
            "user_id": user_id,
            "current_user": current_user,
            "service": user_service,
            "session": db_session
        }
    
    @app.post("/content")
    async def create_content(
        content_data: Dict,
        current_user: Dict = Depends(manager.get_dependency("current_user")),
        content_service: Dict = Depends(manager.get_dependency("content_service")),
        redis_client: redis.Redis = Depends(manager.get_dependency("redis_client"))
    ):
        return {
            "content": content_data,
            "user": current_user,
            "service": content_service,
            "cache": "redis_client_available"
        }
    
    @app.get("/analytics")
    async def get_analytics(
        analytics_service: Dict = Depends(manager.get_dependency("analytics_service")),
        http_client: Any = Depends(manager.get_dependency("http_client"))
    ):
        return {
            "analytics": analytics_service,
            "http_client": "available"
        }
    
    # Add dependency management endpoints
    @app.get("/dependencies/metrics")
    async def get_dependency_metrics():
        """Get dependency performance metrics"""
        return manager.get_metrics()
    
    @app.get("/dependencies/health")
    async def get_dependency_health():
        """Get dependency health status"""
        return manager.health_check()
    
    @app.get("/dependencies/{name}/metrics")
    async def get_dependency_metrics_by_name(name: str):
        """Get metrics for specific dependency"""
        return manager.get_metrics(name)
    
    return app

if __name__ == "__main__":
    app = create_app_with_dependencies()
    print("FastAPI app with dependency management created successfully") 