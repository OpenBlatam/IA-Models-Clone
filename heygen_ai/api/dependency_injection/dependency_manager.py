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
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic, Type
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator, ConfigDict
from .resource_manager import ResourceManager, ResourceType, ResourceConfig
from .service_container import ServiceContainer
from .cache_manager import CacheManager
from .config_manager import ConfigManager
from .security_manager import SecurityManager
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
FastAPI Dependency Injection Manager for HeyGen AI API
Comprehensive dependency injection system for managing state and shared resources.
"""




logger = structlog.get_logger()

# =============================================================================
# Dependency Types
# =============================================================================

class DependencyScope(Enum):
    """Dependency scope enumeration."""
    REQUEST = "request"
    SESSION = "session"
    APPLICATION = "application"
    SINGLETON = "singleton"

class DependencyPriority(Enum):
    """Dependency priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DependencyConfig:
    """Dependency configuration."""
    scope: DependencyScope = DependencyScope.REQUEST
    priority: DependencyPriority = DependencyPriority.NORMAL
    auto_cleanup: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 300
    enable_monitoring: bool = True
    enable_metrics: bool = True

@dataclass
class DependencyStats:
    """Dependency statistics."""
    name: str
    scope: DependencyScope
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    error_count: int = 0
    total_creation_time_ms: float = 0.0
    average_creation_time_ms: float = 0.0
    is_healthy: bool = True
    dependencies: List[str] = Field(default_factory=list)

# =============================================================================
# Base Dependency Classes
# =============================================================================

class DependencyBase:
    """Base class for all dependencies."""
    
    def __init__(self, config: DependencyConfig):
        
    """__init__ function."""
self.config = config
        self.stats = DependencyStats(
            name=self.__class__.__name__,
            scope=config.scope,
            created_at=datetime.now(timezone.utc)
        )
        self._dependencies: Dict[str, Any] = {}
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the dependency."""
        if self._is_initialized:
            return
        
        start_time = time.time()
        try:
            await self._initialize_internal()
            self._is_initialized = True
            
            creation_time_ms = (time.time() - start_time) * 1000
            self.stats.total_creation_time_ms += creation_time_ms
            self.stats.average_creation_time_ms = self.stats.total_creation_time_ms
            
            logger.info(
                f"Dependency {self.__class__.__name__} initialized successfully",
                creation_time_ms=creation_time_ms
            )
            
        except Exception as e:
            self.stats.error_count += 1
            self.stats.is_healthy = False
            logger.error(f"Failed to initialize dependency {self.__class__.__name__}: {e}")
            raise
    
    async def _initialize_internal(self) -> None:
        """Internal initialization method to be implemented by subclasses."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup the dependency."""
        if not self._is_initialized:
            return
        
        try:
            await self._cleanup_internal()
            self._is_initialized = False
            logger.info(f"Dependency {self.__class__.__name__} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup dependency {self.__class__.__name__}: {e}")
            raise
    
    async def _cleanup_internal(self) -> None:
        """Internal cleanup method to be implemented by subclasses."""
        pass
    
    def add_dependency(self, name: str, dependency: Any) -> None:
        """Add a dependency."""
        self._dependencies[name] = dependency
        self.stats.dependencies.append(name)
    
    def get_dependency(self, name: str) -> Optional[Any]:
        """Get a dependency."""
        return self._dependencies.get(name)
    
    def update_stats(self) -> None:
        """Update dependency statistics."""
        self.stats.last_accessed = datetime.now(timezone.utc)
        self.stats.access_count += 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get dependency health status."""
        return {
            "name": self.stats.name,
            "scope": self.stats.scope.value,
            "is_healthy": self.stats.is_healthy,
            "is_initialized": self._is_initialized,
            "access_count": self.stats.access_count,
            "error_count": self.stats.error_count,
            "average_creation_time_ms": self.stats.average_creation_time_ms,
            "last_accessed": self.stats.last_accessed.isoformat() if self.stats.last_accessed else None,
            "dependencies": self.stats.dependencies
        }

# =============================================================================
# Database Dependencies
# =============================================================================

class DatabaseDependency(DependencyBase):
    """Database dependency for managing database connections."""
    
    def __init__(self, config: DependencyConfig, database_url: str):
        
    """__init__ function."""
super().__init__(config)
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
    
    async def _initialize_internal(self) -> None:
        """Initialize database connection."""
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=20,
            max_overflow=30
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with self.session_factory() as session:
            await session.execute("SELECT 1")
    
    async def _cleanup_internal(self) -> None:
        """Cleanup database connection."""
        if self.engine:
            await self.engine.dispose()
    
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        if not self._is_initialized:
            await self.initialize()
        
        self.update_stats()
        return self.session_factory()
    
    async def get_session_dependency(self) -> AsyncSession:
        """Get database session for dependency injection."""
        async with self.get_session() as session:
            yield session

class RedisDependency(DependencyBase):
    """Redis dependency for managing Redis connections."""
    
    def __init__(self, config: DependencyConfig, redis_url: str):
        
    """__init__ function."""
super().__init__(config)
        self.redis_url = redis_url
        self.client = None
    
    async def _initialize_internal(self) -> None:
        """Initialize Redis connection."""
        self.client = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        await self.client.ping()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup Redis connection."""
        if self.client:
            await self.client.close()
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self._is_initialized:
            await self.initialize()
        
        self.update_stats()
        return self.client

# =============================================================================
# Service Dependencies
# =============================================================================

class ServiceDependency(DependencyBase):
    """Service dependency for managing business logic services."""
    
    def __init__(self, config: DependencyConfig, service_class: Type, **kwargs):
        
    """__init__ function."""
super().__init__(config)
        self.service_class = service_class
        self.service_kwargs = kwargs
        self.service_instance = None
    
    async def _initialize_internal(self) -> None:
        """Initialize service instance."""
        self.service_instance = self.service_class(**self.service_kwargs)
        
        # Initialize service if it has an initialize method
        if hasattr(self.service_instance, 'initialize'):
            await self.service_instance.initialize()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup service instance."""
        if self.service_instance and hasattr(self.service_instance, 'cleanup'):
            await self.service_instance.cleanup()
    
    def get_service(self) -> Optional[Dict[str, Any]]:
        """Get service instance."""
        if not self._is_initialized:
            raise RuntimeError(f"Service {self.service_class.__name__} not initialized")
        
        self.update_stats()
        return self.service_instance

# =============================================================================
# Cache Dependencies
# =============================================================================

class CacheDependency(DependencyBase):
    """Cache dependency for managing caching systems."""
    
    def __init__(self, config: DependencyConfig, cache_manager: CacheManager):
        
    """__init__ function."""
super().__init__(config)
        self.cache_manager = cache_manager
    
    async def _initialize_internal(self) -> None:
        """Initialize cache manager."""
        await self.cache_manager.initialize()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup cache manager."""
        await self.cache_manager.cleanup()
    
    def get_cache_manager(self) -> CacheManager:
        """Get cache manager."""
        if not self._is_initialized:
            raise RuntimeError("Cache manager not initialized")
        
        self.update_stats()
        return self.cache_manager

# =============================================================================
# Security Dependencies
# =============================================================================

class SecurityDependency(DependencyBase):
    """Security dependency for managing authentication and authorization."""
    
    def __init__(self, config: DependencyConfig, security_manager: SecurityManager):
        
    """__init__ function."""
super().__init__(config)
        self.security_manager = security_manager
        self.http_bearer = HTTPBearer()
    
    async def _initialize_internal(self) -> None:
        """Initialize security manager."""
        await self.security_manager.initialize()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup security manager."""
        await self.security_manager.cleanup()
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Any:
        """Get current authenticated user."""
        if not self._is_initialized:
            raise RuntimeError("Security manager not initialized")
        
        self.update_stats()
        return await self.security_manager.authenticate_user(credentials.credentials)
    
    async def require_admin_role(self, current_user: Any = Depends()) -> Any:
        """Require admin role for endpoint access."""
        if not self._is_initialized:
            raise RuntimeError("Security manager not initialized")
        
        if not self.security_manager.has_role(current_user, "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        return current_user

# =============================================================================
# Configuration Dependencies
# =============================================================================

class ConfigDependency(DependencyBase):
    """Configuration dependency for managing application configuration."""
    
    def __init__(self, config: DependencyConfig, config_manager: ConfigManager):
        
    """__init__ function."""
super().__init__(config)
        self.config_manager = config_manager
    
    async def _initialize_internal(self) -> None:
        """Initialize configuration manager."""
        await self.config_manager.initialize()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup configuration manager."""
        await self.config_manager.cleanup()
    
    def get_config_manager(self) -> ConfigManager:
        """Get configuration manager."""
        if not self._is_initialized:
            raise RuntimeError("Configuration manager not initialized")
        
        self.update_stats()
        return self.config_manager
    
    def get_setting(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get configuration setting."""
        return self.config_manager.get_setting(key, default)

# =============================================================================
# Dependency Container
# =============================================================================

class DependencyContainer:
    """Main dependency container for managing all dependencies."""
    
    def __init__(self) -> Any:
        self.dependencies: Dict[str, DependencyBase] = {}
        self.dependency_factories: Dict[str, Callable] = {}
        self.scope_managers: Dict[DependencyScope, Any] = {}
        self._initialized = False
    
    def register_dependency(
        self,
        name: str,
        dependency: DependencyBase,
        factory: Optional[Callable] = None
    ) -> None:
        """Register a dependency."""
        self.dependencies[name] = dependency
        if factory:
            self.dependency_factories[name] = factory
        
        logger.info(f"Registered dependency: {name} ({dependency.__class__.__name__})")
    
    def get_dependency(self, name: str) -> Optional[DependencyBase]:
        """Get a dependency by name."""
        return self.dependencies.get(name)
    
    def create_dependency_function(self, name: str) -> Callable:
        """Create a dependency function for FastAPI."""
        def dependency_function():
            
    """dependency_function function."""
dependency = self.get_dependency(name)
            if not dependency:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Dependency {name} not found"
                )
            
            if dependency.config.scope == DependencyScope.REQUEST:
                # For request-scoped dependencies, return a function that yields
                async def request_dependency():
                    
    """request_dependency function."""
if not dependency._is_initialized:
                        await dependency.initialize()
                    dependency.update_stats()
                    yield dependency
                return request_dependency()
            
            elif dependency.config.scope == DependencyScope.SESSION:
                # For session-scoped dependencies, return the dependency directly
                if not dependency._is_initialized:
                    raise RuntimeError(f"Dependency {name} not initialized")
                dependency.update_stats()
                return dependency
            
            elif dependency.config.scope == DependencyScope.APPLICATION:
                # For application-scoped dependencies, return the dependency directly
                if not dependency._is_initialized:
                    raise RuntimeError(f"Dependency {name} not initialized")
                dependency.update_stats()
                return dependency
            
            elif dependency.config.scope == DependencyScope.SINGLETON:
                # For singleton dependencies, return the dependency directly
                if not dependency._is_initialized:
                    raise RuntimeError(f"Dependency {name} not initialized")
                dependency.update_stats()
                return dependency
        
        return dependency_function
    
    async def initialize_all(self) -> None:
        """Initialize all dependencies."""
        if self._initialized:
            return
        
        logger.info("Initializing all dependencies...")
        
        # Initialize dependencies by priority
        priorities = [
            DependencyPriority.CRITICAL,
            DependencyPriority.HIGH,
            DependencyPriority.NORMAL,
            DependencyPriority.LOW
        ]
        
        for priority in priorities:
            priority_dependencies = [
                dep for dep in self.dependencies.values()
                if dep.config.priority == priority
            ]
            
            for dependency in priority_dependencies:
                try:
                    await dependency.initialize()
                except Exception as e:
                    logger.error(f"Failed to initialize dependency {dependency.__class__.__name__}: {e}")
                    raise
        
        self._initialized = True
        logger.info("All dependencies initialized successfully")
    
    async def cleanup_all(self) -> None:
        """Cleanup all dependencies."""
        if not self._initialized:
            return
        
        logger.info("Cleaning up all dependencies...")
        
        # Cleanup dependencies in reverse priority order
        priorities = [
            DependencyPriority.LOW,
            DependencyPriority.NORMAL,
            DependencyPriority.HIGH,
            DependencyPriority.CRITICAL
        ]
        
        for priority in priorities:
            priority_dependencies = [
                dep for dep in self.dependencies.values()
                if dep.config.priority == priority and dep.config.auto_cleanup
            ]
            
            for dependency in priority_dependencies:
                try:
                    await dependency.cleanup()
                except Exception as e:
                    logger.error(f"Failed to cleanup dependency {dependency.__class__.__name__}: {e}")
        
        self._initialized = False
        logger.info("All dependencies cleaned up successfully")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all dependencies."""
        health_status = {
            "initialized": self._initialized,
            "total_dependencies": len(self.dependencies),
            "dependencies": {}
        }
        
        for name, dependency in self.dependencies.items():
            health_status["dependencies"][name] = dependency.get_health_status()
        
        return health_status

# =============================================================================
# Dependency Decorators
# =============================================================================

def inject_dependency(dependency_name: str):
    """Decorator for injecting dependencies into functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # This would be used with the dependency container
            # The actual injection would happen at the FastAPI level
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def singleton_dependency(cls: Type) -> Type:
    """Decorator for creating singleton dependencies."""
    original_init = cls.__init__
    
    def __init__(self, *args, **kwargs) -> Any:
        if not hasattr(cls, '_instance'):
            original_init(self, *args, **kwargs)
            cls._instance = self
        else:
            self.__dict__ = cls._instance.__dict__
    
    cls.__init__ = __init__
    return cls

# =============================================================================
# FastAPI Integration
# =============================================================================

class FastAPIDependencyManager:
    """FastAPI-specific dependency manager."""
    
    def __init__(self, app) -> Any:
        self.app = app
        self.container = DependencyContainer()
        self._setup_lifecycle_events()
    
    def _setup_lifecycle_events(self) -> Any:
        """Setup FastAPI lifecycle events."""
        @self.app.on_event("startup")
        async def startup_event():
            
    """startup_event function."""
await self.container.initialize_all()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            
    """shutdown_event function."""
await self.container.cleanup_all()
    
    def register_database(
        self,
        name: str,
        database_url: str,
        config: Optional[DependencyConfig] = None
    ) -> None:
        """Register database dependency."""
        if config is None:
            config = DependencyConfig(
                scope=DependencyScope.APPLICATION,
                priority=DependencyPriority.CRITICAL
            )
        
        dependency = DatabaseDependency(config, database_url)
        self.container.register_dependency(name, dependency)
    
    def register_redis(
        self,
        name: str,
        redis_url: str,
        config: Optional[DependencyConfig] = None
    ) -> None:
        """Register Redis dependency."""
        if config is None:
            config = DependencyConfig(
                scope=DependencyScope.APPLICATION,
                priority=DependencyPriority.HIGH
            )
        
        dependency = RedisDependency(config, redis_url)
        self.container.register_dependency(name, dependency)
    
    def register_service(
        self,
        name: str,
        service_class: Type,
        config: Optional[DependencyConfig] = None,
        **kwargs
    ) -> None:
        """Register service dependency."""
        if config is None:
            config = DependencyConfig(
                scope=DependencyScope.REQUEST,
                priority=DependencyPriority.NORMAL
            )
        
        dependency = ServiceDependency(config, service_class, **kwargs)
        self.container.register_dependency(name, dependency)
    
    def register_cache(
        self,
        name: str,
        cache_manager: CacheManager,
        config: Optional[DependencyConfig] = None
    ) -> None:
        """Register cache dependency."""
        if config is None:
            config = DependencyConfig(
                scope=DependencyScope.APPLICATION,
                priority=DependencyPriority.HIGH
            )
        
        dependency = CacheDependency(config, cache_manager)
        self.container.register_dependency(name, dependency)
    
    def register_security(
        self,
        name: str,
        security_manager: SecurityManager,
        config: Optional[DependencyConfig] = None
    ) -> None:
        """Register security dependency."""
        if config is None:
            config = DependencyConfig(
                scope=DependencyScope.APPLICATION,
                priority=DependencyScope.CRITICAL
            )
        
        dependency = SecurityDependency(config, security_manager)
        self.container.register_dependency(name, dependency)
    
    def register_config(
        self,
        name: str,
        config_manager: ConfigManager,
        config: Optional[DependencyConfig] = None
    ) -> None:
        """Register configuration dependency."""
        if config is None:
            config = DependencyConfig(
                scope=DependencyScope.APPLICATION,
                priority=DependencyScope.CRITICAL
            )
        
        dependency = ConfigDependency(config, config_manager)
        self.container.register_dependency(name, dependency)
    
    def get_dependency_function(self, name: str) -> Callable:
        """Get dependency function for FastAPI dependency injection."""
        return self.container.create_dependency_function(name)
    
    def get_health_endpoint(self) -> Optional[Dict[str, Any]]:
        """Get health check endpoint for dependencies."""
        async def health_check():
            
    """health_check function."""
return self.container.get_health_status()
        
        return health_check

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "DependencyScope",
    "DependencyPriority",
    "DependencyConfig",
    "DependencyStats",
    "DependencyBase",
    "DatabaseDependency",
    "RedisDependency",
    "ServiceDependency",
    "CacheDependency",
    "SecurityDependency",
    "ConfigDependency",
    "DependencyContainer",
    "FastAPIDependencyManager",
    "inject_dependency",
    "singleton_dependency",
] 