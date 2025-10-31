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
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        import redis.asyncio as redis
        import redis.asyncio as redis
        import aiohttp
        import os
        import aiofiles
        import os
        import aiofiles
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Resource Manager for FastAPI Dependency Injection
Manages shared resources and their lifecycle.
"""


logger = structlog.get_logger()

# =============================================================================
# Resource Types
# =============================================================================

class ResourceType(Enum):
    """Resource type enumeration."""
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    STORAGE = "storage"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    MEMORY = "memory"
    NETWORK = "network"

class ResourceStatus(Enum):
    """Resource status enumeration."""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"

@dataclass
class ResourceConfig:
    """Resource configuration."""
    name: str
    resource_type: ResourceType
    max_connections: int = 10
    connection_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0
    auto_reconnect: bool = True
    enable_monitoring: bool = True
    enable_metrics: bool = True
    cleanup_on_shutdown: bool = True

@dataclass
class ResourceStats:
    """Resource statistics."""
    name: str
    resource_type: ResourceType
    status: ResourceStatus
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    error_count: int = 0
    total_connection_time_ms: float = 0.0
    average_connection_time_ms: float = 0.0
    active_connections: int = 0
    max_connections_reached: int = 0
    health_check_count: int = 0
    last_health_check: Optional[datetime] = None

# =============================================================================
# Base Resource Classes
# =============================================================================

class ResourceBase:
    """Base class for all resources."""
    
    def __init__(self, config: ResourceConfig):
        
    """__init__ function."""
self.config = config
        self.stats = ResourceStats(
            name=config.name,
            resource_type=config.resource_type,
            status=ResourceStatus.INITIALIZING,
            created_at=datetime.now(timezone.utc)
        )
        self._connections: List[Any] = []
        self._is_initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the resource."""
        if self._is_initialized:
            return
        
        self.stats.status = ResourceStatus.INITIALIZING
        start_time = time.time()
        
        try:
            await self._initialize_internal()
            self._is_initialized = True
            self.stats.status = ResourceStatus.READY
            
            connection_time_ms = (time.time() - start_time) * 1000
            self.stats.total_connection_time_ms += connection_time_ms
            self.stats.average_connection_time_ms = self.stats.total_connection_time_ms
            
            # Start health check if enabled
            if self.config.enable_monitoring:
                self._start_health_check()
            
            logger.info(
                f"Resource {self.config.name} initialized successfully",
                connection_time_ms=connection_time_ms
            )
            
        except Exception as e:
            self.stats.status = ResourceStatus.ERROR
            self.stats.error_count += 1
            logger.error(f"Failed to initialize resource {self.config.name}: {e}")
            raise
    
    async def _initialize_internal(self) -> None:
        """Internal initialization method to be implemented by subclasses."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup the resource."""
        if not self._is_initialized:
            return
        
        self.stats.status = ResourceStatus.SHUTTING_DOWN
        
        # Stop health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        try:
            await self._cleanup_internal()
            self._is_initialized = False
            self.stats.status = ResourceStatus.SHUTDOWN
            logger.info(f"Resource {self.config.name} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup resource {self.config.name}: {e}")
            raise
    
    async def _cleanup_internal(self) -> None:
        """Internal cleanup method to be implemented by subclasses."""
        pass
    
    @asynccontextmanager
    async def get_connection(self) -> Optional[Dict[str, Any]]:
        """Get a connection from the resource pool."""
        if not self._is_initialized:
            raise RuntimeError(f"Resource {self.config.name} not initialized")
        
        # Check connection limit
        if len(self._connections) >= self.config.max_connections:
            self.stats.max_connections_reached += 1
            raise RuntimeError(f"Maximum connections reached for resource {self.config.name}")
        
        connection = None
        start_time = time.time()
        
        try:
            # Get or create connection
            connection = await self._get_connection_internal()
            self._connections.append(connection)
            self.stats.active_connections = len(self._connections)
            
            self.stats.last_accessed = datetime.now(timezone.utc)
            self.stats.access_count += 1
            
            yield connection
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error(f"Error getting connection from resource {self.config.name}: {e}")
            raise
            
        finally:
            # Return connection to pool
            if connection and connection in self._connections:
                self._connections.remove(connection)
                self.stats.active_connections = len(self._connections)
            
            connection_time_ms = (time.time() - start_time) * 1000
            self.stats.total_connection_time_ms += connection_time_ms
            self.stats.average_connection_time_ms = self.stats.total_connection_time_ms / self.stats.access_count
    
    async def _get_connection_internal(self) -> Optional[Dict[str, Any]]:
        """Internal method to get connection - to be implemented by subclasses."""
        raise NotImplementedError
    
    def _start_health_check(self) -> None:
        """Start health check task."""
        if self._health_check_task:
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self) -> None:
        """Health check loop."""
        while self._is_initialized:
            try:
                await self._perform_health_check()
                self.stats.last_health_check = datetime.now(timezone.utc)
                self.stats.health_check_count += 1
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed for resource {self.config.name}: {e}")
                self.stats.status = ResourceStatus.DEGRADED
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_health_check(self) -> None:
        """Perform health check - to be implemented by subclasses."""
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get resource health status."""
        return {
            "name": self.stats.name,
            "type": self.stats.resource_type.value,
            "status": self.stats.status.value,
            "is_initialized": self._is_initialized,
            "access_count": self.stats.access_count,
            "error_count": self.stats.error_count,
            "active_connections": self.stats.active_connections,
            "max_connections": self.config.max_connections,
            "average_connection_time_ms": self.stats.average_connection_time_ms,
            "last_accessed": self.stats.last_accessed.isoformat() if self.stats.last_accessed else None,
            "last_health_check": self.stats.last_health_check.isoformat() if self.stats.last_health_check else None,
            "health_check_count": self.stats.health_check_count
        }

# =============================================================================
# Database Resource
# =============================================================================

class DatabaseResource(ResourceBase):
    """Database resource for managing database connections."""
    
    def __init__(self, config: ResourceConfig, database_url: str):
        
    """__init__ function."""
super().__init__(config)
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
    
    async def _initialize_internal(self) -> None:
        """Initialize database connection pool."""
        
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=self.config.max_connections,
            max_overflow=self.config.max_connections * 2,
            pool_timeout=self.config.connection_timeout
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False
        )
        
        # Test connection
        async with self.session_factory() as session:
            await session.execute("SELECT 1")
    
    async def _cleanup_internal(self) -> None:
        """Cleanup database connections."""
        if self.engine:
            await self.engine.dispose()
    
    async def _get_connection_internal(self) -> Optional[Dict[str, Any]]:
        """Get database session."""
        return self.session_factory()
    
    async def _perform_health_check(self) -> None:
        """Perform database health check."""
        async with self.session_factory() as session:
            await session.execute("SELECT 1")

# =============================================================================
# Redis Resource
# =============================================================================

class RedisResource(ResourceBase):
    """Redis resource for managing Redis connections."""
    
    def __init__(self, config: ResourceConfig, redis_url: str):
        
    """__init__ function."""
super().__init__(config)
        self.redis_url = redis_url
        self.pool = None
    
    async def _initialize_internal(self) -> None:
        """Initialize Redis connection pool."""
        
        self.pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=self.config.max_connections,
            socket_connect_timeout=self.config.connection_timeout,
            socket_timeout=self.config.connection_timeout,
            retry_on_timeout=True,
            health_check_interval=self.config.health_check_interval
        )
        
        # Test connection
        client = redis.Redis(connection_pool=self.pool)
        await client.ping()
        await client.close()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup Redis connections."""
        if self.pool:
            await self.pool.disconnect()
    
    async def _get_connection_internal(self) -> Optional[Dict[str, Any]]:
        """Get Redis client."""
        return redis.Redis(connection_pool=self.pool)
    
    async def _perform_health_check(self) -> None:
        """Perform Redis health check."""
        client = redis.Redis(connection_pool=self.pool)
        await client.ping()
        await client.close()

# =============================================================================
# External API Resource
# =============================================================================

class ExternalAPIResource(ResourceBase):
    """External API resource for managing API connections."""
    
    def __init__(self, config: ResourceConfig, base_url: str, headers: Optional[Dict[str, str]] = None):
        
    """__init__ function."""
super().__init__(config)
        self.base_url = base_url
        self.headers = headers or {}
        self.session = None
    
    async def _initialize_internal(self) -> None:
        """Initialize HTTP session."""
        
        timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
        self.session = aiohttp.ClientSession(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout
        )
        
        # Test connection
        async with self.session.get("/health") as response:
            if response.status >= 400:
                raise RuntimeError(f"External API health check failed: {response.status}")
    
    async def _cleanup_internal(self) -> None:
        """Cleanup HTTP session."""
        if self.session:
            await self.session.close()
    
    async def _get_connection_internal(self) -> Optional[Dict[str, Any]]:
        """Get HTTP session."""
        return self.session
    
    async def _perform_health_check(self) -> None:
        """Perform external API health check."""
        async with self.session.get("/health") as response:
            if response.status >= 400:
                raise RuntimeError(f"External API health check failed: {response.status}")

# =============================================================================
# File System Resource
# =============================================================================

class FileSystemResource(ResourceBase):
    """File system resource for managing file operations."""
    
    def __init__(self, config: ResourceConfig, base_path: str):
        
    """__init__ function."""
super().__init__(config)
        self.base_path = base_path
        self._semaphore = None
    
    async def _initialize_internal(self) -> None:
        """Initialize file system resource."""
        
        # Ensure base path exists
        os.makedirs(self.base_path, exist_ok=True)
        
        # Create semaphore for limiting concurrent operations
        self._semaphore = asyncio.Semaphore(self.config.max_connections)
        
        # Test write access
        test_file = os.path.join(self.base_path, ".test")
        async with aiofiles.open(test_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await f.write("test")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        os.remove(test_file)
    
    async def _cleanup_internal(self) -> None:
        """Cleanup file system resource."""
        # Nothing to cleanup for file system
        pass
    
    async def _get_connection_internal(self) -> Optional[Dict[str, Any]]:
        """Get file system connection (semaphore)."""
        return self._semaphore
    
    async def _perform_health_check(self) -> None:
        """Perform file system health check."""
        
        # Test write access
        test_file = os.path.join(self.base_path, f".health_check_{int(time.time())}")
        async with aiofiles.open(test_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await f.write("health_check")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        os.remove(test_file)

# =============================================================================
# Resource Manager
# =============================================================================

class ResourceManager:
    """Main resource manager for managing all resources."""
    
    def __init__(self) -> Any:
        self.resources: Dict[str, ResourceBase] = {}
        self._initialized = False
    
    def register_resource(self, name: str, resource: ResourceBase) -> None:
        """Register a resource."""
        self.resources[name] = resource
        logger.info(f"Registered resource: {name} ({resource.__class__.__name__})")
    
    def get_resource(self, name: str) -> Optional[ResourceBase]:
        """Get a resource by name."""
        return self.resources.get(name)
    
    async def initialize_all(self) -> None:
        """Initialize all resources."""
        if self._initialized:
            return
        
        logger.info("Initializing all resources...")
        
        for name, resource in self.resources.items():
            try:
                await resource.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize resource {name}: {e}")
                raise
        
        self._initialized = True
        logger.info("All resources initialized successfully")
    
    async def cleanup_all(self) -> None:
        """Cleanup all resources."""
        if not self._initialized:
            return
        
        logger.info("Cleaning up all resources...")
        
        for name, resource in self.resources.items():
            try:
                await resource.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup resource {name}: {e}")
        
        self._initialized = False
        logger.info("All resources cleaned up successfully")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all resources."""
        health_status = {
            "initialized": self._initialized,
            "total_resources": len(self.resources),
            "resources": {}
        }
        
        for name, resource in self.resources.items():
            health_status["resources"][name] = resource.get_health_status()
        
        return health_status
    
    @asynccontextmanager
    async def get_resource_connection(self, resource_name: str):
        """Get a connection from a specific resource."""
        resource = self.get_resource(resource_name)
        if not resource:
            raise RuntimeError(f"Resource {resource_name} not found")
        
        async with resource.get_connection() as connection:
            yield connection

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ResourceType",
    "ResourceStatus",
    "ResourceConfig",
    "ResourceStats",
    "ResourceBase",
    "DatabaseResource",
    "RedisResource",
    "ExternalAPIResource",
    "FileSystemResource",
    "ResourceManager",
] 