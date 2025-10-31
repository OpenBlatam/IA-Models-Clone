from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Type, TypeVar
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict, Field
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Service Container for FastAPI Dependency Injection
Manages service dependencies and their lifecycle.
"""


logger = structlog.get_logger()

# =============================================================================
# Service Types
# =============================================================================

class ServiceScope(Enum):
    """Service scope enumeration."""
    REQUEST = "request"
    SESSION = "session"
    APPLICATION = "application"
    SINGLETON = "singleton"

class ServiceStatus(Enum):
    """Service status enumeration."""
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"

@dataclass
class ServiceConfig:
    """Service configuration."""
    name: str
    service_class: Type
    scope: ServiceScope = ServiceScope.REQUEST
    auto_initialize: bool = True
    lazy_loading: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300
    enable_monitoring: bool = True
    enable_metrics: bool = True
    dependencies: List[str] = None
    factory_function: Optional[Callable] = None
    cleanup_on_shutdown: bool = True

@dataclass
class ServiceStats:
    """Service statistics."""
    name: str
    service_class: str
    scope: ServiceScope
    status: ServiceStatus
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    error_count: int = 0
    total_creation_time_ms: float = 0.0
    average_creation_time_ms: float = 0.0
    instance_count: int = 0
    dependencies: List[str] = Field(default_factory=list)

# =============================================================================
# Service Base Classes
# =============================================================================

class ServiceBase:
    """Base class for all services."""
    
    def __init__(self) -> Any:
        self._is_initialized = False
        self._dependencies: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize the service."""
        if self._is_initialized:
            return
        
        try:
            await self._initialize_internal()
            self._is_initialized = True
            logger.info(f"Service {self.__class__.__name__} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service {self.__class__.__name__}: {e}")
            raise
    
    async def _initialize_internal(self) -> None:
        """Internal initialization method to be implemented by subclasses."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup the service."""
        if not self._is_initialized:
            return
        
        try:
            await self._cleanup_internal()
            self._is_initialized = False
            logger.info(f"Service {self.__class__.__name__} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup service {self.__class__.__name__}: {e}")
            raise
    
    async def _cleanup_internal(self) -> None:
        """Internal cleanup method to be implemented by subclasses."""
        pass
    
    def add_dependency(self, name: str, dependency: Any) -> None:
        """Add a dependency to the service."""
        self._dependencies[name] = dependency
    
    def get_dependency(self, name: str) -> Optional[Any]:
        """Get a dependency from the service."""
        return self._dependencies.get(name)
    
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._is_initialized

# =============================================================================
# Service Instance
# =============================================================================

class ServiceInstance:
    """Service instance wrapper."""
    
    def __init__(self, service: Any, config: ServiceConfig, stats: ServiceStats):
        
    """__init__ function."""
self.service = service
        self.config = config
        self.stats = stats
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed = None
    
    def access(self) -> Any:
        """Access the service instance."""
        self.last_accessed = datetime.now(timezone.utc)
        self.stats.access_count += 1
        return self.service
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service instance health status."""
        return {
            "name": self.stats.name,
            "service_class": self.stats.service_class,
            "scope": self.stats.scope.value,
            "status": self.stats.status.value,
            "access_count": self.stats.access_count,
            "error_count": self.stats.error_count,
            "average_creation_time_ms": self.stats.average_creation_time_ms,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat(),
            "dependencies": self.stats.dependencies
        }

# =============================================================================
# Service Container
# =============================================================================

class ServiceContainer:
    """Main service container for managing service dependencies."""
    
    def __init__(self) -> Any:
        self.services: Dict[str, ServiceConfig] = {}
        self.instances: Dict[str, ServiceInstance] = {}
        self.factories: Dict[str, Callable] = {}
        self._initialized = False
        self._cache: Dict[str, Any] = {}
    
    def register_service(
        self,
        name: str,
        service_class: Type,
        scope: ServiceScope = ServiceScope.REQUEST,
        dependencies: Optional[List[str]] = None,
        factory_function: Optional[Callable] = None,
        **kwargs
    ) -> None:
        """Register a service."""
        config = ServiceConfig(
            name=name,
            service_class=service_class,
            scope=scope,
            dependencies=dependencies or [],
            factory_function=factory_function,
            **kwargs
        )
        
        self.services[name] = config
        if factory_function:
            self.factories[name] = factory_function
        
        logger.info(f"Registered service: {name} ({service_class.__name__})")
    
    def get_service_config(self, name: str) -> Optional[ServiceConfig]:
        """Get service configuration."""
        return self.services.get(name)
    
    async def get_service(self, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get a service instance."""
        config = self.get_service_config(name)
        if not config:
            raise RuntimeError(f"Service {name} not registered")
        
        # Check cache for singleton services
        if config.scope == ServiceScope.SINGLETON and name in self._cache:
            return self._cache[name]
        
        # Create or get instance based on scope
        if config.scope == ServiceScope.SINGLETON:
            instance = await self._create_singleton_instance(name, config, **kwargs)
            return instance
        
        elif config.scope == ServiceScope.APPLICATION:
            instance = await self._get_application_instance(name, config, **kwargs)
            return instance
        
        elif config.scope == ServiceScope.SESSION:
            instance = await self._get_session_instance(name, config, **kwargs)
            return instance
        
        else:  # REQUEST scope
            instance = await self._create_request_instance(name, config, **kwargs)
            return instance
    
    async def _create_singleton_instance(self, name: str, config: ServiceConfig, **kwargs) -> Any:
        """Create singleton service instance."""
        if name in self._cache:
            return self._cache[name]
        
        start_time = time.time()
        try:
            instance = await self._create_instance(name, config, **kwargs)
            self._cache[name] = instance
            
            creation_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Created singleton service {name} in {creation_time_ms:.2f}ms")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create singleton service {name}: {e}")
            raise
    
    async def _get_application_instance(self, name: str, config: ServiceConfig, **kwargs) -> Optional[Dict[str, Any]]:
        """Get application-scoped service instance."""
        if name in self.instances:
            service_instance = self.instances[name]
            return service_instance.access()
        
        start_time = time.time()
        try:
            instance = await self._create_instance(name, config, **kwargs)
            service_instance = ServiceInstance(
                instance,
                config,
                ServiceStats(
                    name=name,
                    service_class=config.service_class.__name__,
                    scope=config.scope,
                    status=ServiceStatus.READY,
                    created_at=datetime.now(timezone.utc),
                    dependencies=config.dependencies
                )
            )
            
            self.instances[name] = service_instance
            
            creation_time_ms = (time.time() - start_time) * 1000
            service_instance.stats.total_creation_time_ms = creation_time_ms
            service_instance.stats.average_creation_time_ms = creation_time_ms
            
            logger.info(f"Created application service {name} in {creation_time_ms:.2f}ms")
            
            return service_instance.access()
            
        except Exception as e:
            logger.error(f"Failed to create application service {name}: {e}")
            raise
    
    async def _get_session_instance(self, name: str, config: ServiceConfig, **kwargs) -> Optional[Dict[str, Any]]:
        """Get session-scoped service instance."""
        # For session scope, we would typically use a session ID
        # For now, we'll create a new instance per request
        return await self._create_request_instance(name, config, **kwargs)
    
    async async def _create_request_instance(self, name: str, config: ServiceConfig, **kwargs) -> Any:
        """Create request-scoped service instance."""
        start_time = time.time()
        try:
            instance = await self._create_instance(name, config, **kwargs)
            
            creation_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Created request service {name} in {creation_time_ms:.2f}ms")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create request service {name}: {e}")
            raise
    
    async def _create_instance(self, name: str, config: ServiceConfig, **kwargs) -> Any:
        """Create service instance."""
        # Resolve dependencies
        resolved_dependencies = {}
        for dep_name in config.dependencies:
            dep_instance = await self.get_service(dep_name)
            resolved_dependencies[dep_name] = dep_instance
        
        # Create instance using factory or constructor
        if config.factory_function:
            instance = config.factory_function(**resolved_dependencies, **kwargs)
        else:
            instance = config.service_class(**resolved_dependencies, **kwargs)
        
        # Initialize if it's a ServiceBase
        if isinstance(instance, ServiceBase):
            await instance.initialize()
        
        return instance
    
    async def initialize_all(self) -> None:
        """Initialize all services that have auto_initialize=True."""
        if self._initialized:
            return
        
        logger.info("Initializing all auto-initializable services...")
        
        for name, config in self.services.items():
            if config.auto_initialize and config.scope in [ServiceScope.SINGLETON, ServiceScope.APPLICATION]:
                try:
                    await self.get_service(name)
                except Exception as e:
                    logger.error(f"Failed to auto-initialize service {name}: {e}")
                    raise
        
        self._initialized = True
        logger.info("All services initialized successfully")
    
    async def cleanup_all(self) -> None:
        """Cleanup all services."""
        if not self._initialized:
            return
        
        logger.info("Cleaning up all services...")
        
        # Cleanup instances
        for name, service_instance in self.instances.items():
            try:
                if hasattr(service_instance.service, 'cleanup'):
                    await service_instance.service.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup service {name}: {e}")
        
        # Cleanup cache
        for name, instance in self._cache.items():
            try:
                if hasattr(instance, 'cleanup'):
                    await instance.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup cached service {name}: {e}")
        
        self.instances.clear()
        self._cache.clear()
        self._initialized = False
        logger.info("All services cleaned up successfully")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all services."""
        health_status = {
            "initialized": self._initialized,
            "total_services": len(self.services),
            "total_instances": len(self.instances),
            "total_cached": len(self._cache),
            "services": {}
        }
        
        # Add service configurations
        for name, config in self.services.items():
            health_status["services"][name] = {
                "name": config.name,
                "service_class": config.service_class.__name__,
                "scope": config.scope.value,
                "auto_initialize": config.auto_initialize,
                "dependencies": config.dependencies
            }
        
        # Add instance statistics
        for name, service_instance in self.instances.items():
            if name not in health_status["services"]:
                health_status["services"][name] = {}
            
            health_status["services"][name].update(service_instance.get_health_status())
        
        return health_status

# =============================================================================
# Service Decorators
# =============================================================================

def inject_service(service_name: str):
    """Decorator for injecting services into functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # This would be used with the service container
            # The actual injection would happen at the FastAPI level
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def singleton_service(cls: Type) -> Type:
    """Decorator for creating singleton services."""
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

class FastAPIServiceContainer:
    """FastAPI-specific service container."""
    
    def __init__(self, app) -> Any:
        self.app = app
        self.container = ServiceContainer()
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
    
    def register_service(
        self,
        name: str,
        service_class: Type,
        scope: ServiceScope = ServiceScope.REQUEST,
        dependencies: Optional[List[str]] = None,
        factory_function: Optional[Callable] = None,
        **kwargs
    ) -> None:
        """Register a service."""
        self.container.register_service(
            name=name,
            service_class=service_class,
            scope=scope,
            dependencies=dependencies,
            factory_function=factory_function,
            **kwargs
        )
    
    def get_service_function(self, name: str) -> Callable:
        """Get service function for FastAPI dependency injection."""
        async def service_dependency():
            
    """service_dependency function."""
return await self.container.get_service(name)
        
        return service_dependency
    
    def get_health_endpoint(self) -> Optional[Dict[str, Any]]:
        """Get health check endpoint for services."""
        async def health_check():
            
    """health_check function."""
return self.container.get_health_status()
        
        return health_check

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ServiceScope",
    "ServiceStatus",
    "ServiceConfig",
    "ServiceStats",
    "ServiceBase",
    "ServiceInstance",
    "ServiceContainer",
    "FastAPIServiceContainer",
    "inject_service",
    "singleton_service",
] 