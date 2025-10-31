from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import weakref
import threading
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
from functools import wraps
import gc
import psutil
import os
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
    import numba
    from numba import jit, njit
from typing import Any, List, Dict, Optional
"""
Shared Resources for Instagram Captions API v14.0

Comprehensive shared resource management including:
- Connection pools (database, HTTP, Redis)
- Shared caches (memory, disk, distributed)
- AI model instances
- Configuration management
- Resource lifecycle management
- Thread-safe resource access
- Resource monitoring and metrics
"""


# Performance libraries
try:
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs) -> Any:
        def decorator(func) -> Any:
            return func
        return decorator
    njit = jit

logger = logging.getLogger(__name__)

# Type variables for generic operations
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class ResourceType(Enum):
    """Types of shared resources"""
    DATABASE = "database"
    HTTP_CLIENT = "http_client"
    REDIS = "redis"
    AI_MODEL = "ai_model"
    CACHE = "cache"
    FILE_STORAGE = "file_storage"
    CONFIG = "config"
    LOGGER = "logger"
    METRICS = "metrics"


class ResourceState(Enum):
    """Resource states"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    CLOSED = "closed"


@dataclass
class ResourceConfig:
    """Configuration for shared resources"""
    # Database
    database_url: str = "postgresql+asyncpg://user:pass@localhost/db"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    
    # HTTP Client
    http_timeout: int = 30
    http_max_connections: int = 100
    http_connection_limit: int = 50
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_pool_size: int = 20
    redis_max_connections: int = 50
    
    # AI Models
    model_cache_size: int = 5
    model_load_timeout: int = 60
    model_memory_limit_mb: int = 2048
    
    # Cache
    cache_size: int = 10000
    cache_ttl: int = 3600
    cache_cleanup_interval: int = 300
    
    # File Storage
    storage_path: str = "/tmp/shared_storage"
    storage_max_size_mb: int = 1024
    
    # Performance
    enable_monitoring: bool = True
    metrics_interval: int = 60
    health_check_interval: int = 30


@dataclass
class ResourceInfo:
    """Information about a shared resource"""
    name: str
    resource_type: ResourceType
    state: ResourceState = ResourceState.INITIALIZING
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    dependencies: Set[str] = field(default_factory=set)


class ConnectionPool:
    """Generic connection pool for various resource types"""
    
    def __init__(self, pool_size: int = 20, max_overflow: int = 30):
        
    """__init__ function."""
self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._pool: List[Any] = []
        self._in_use: Set[Any] = set()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(pool_size + max_overflow)
    
    async def get(self) -> Optional[Dict[str, Any]]:
        """Get a connection from the pool"""
        async with self._semaphore:
            async with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                    self._in_use.add(conn)
                    return conn
                else:
                    # Create new connection
                    conn = await self._create_connection()
                    self._in_use.add(conn)
                    return conn
    
    async def put(self, conn: Any):
        """Return a connection to the pool"""
        async with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                if len(self._pool) < self.pool_size:
                    self._pool.append(conn)
                else:
                    await self._close_connection(conn)
    
    async def _create_connection(self) -> Any:
        """Create a new connection - to be overridden"""
        raise NotImplementedError
    
    async def _close_connection(self, conn: Any):
        """Close a connection - to be overridden"""
        raise NotImplementedError
    
    async def close_all(self) -> Any:
        """Close all connections in the pool"""
        async with self._lock:
            for conn in self._pool:
                await self._close_connection(conn)
            for conn in self._in_use:
                await self._close_connection(conn)
            self._pool.clear()
            self._in_use.clear()


class DatabasePool(ConnectionPool):
    """Database connection pool"""
    
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 30):
        
    """__init__ function."""
super().__init__(pool_size, max_overflow)
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
    
    async def initialize(self) -> Any:
        """Initialize the database pool"""
        self.engine = create_async_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=30,
            echo=False
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def _create_connection(self) -> AsyncSession:
        """Create a new database session"""
        return self.session_factory()
    
    async def _close_connection(self, session: AsyncSession):
        """Close a database session"""
        await session.close()
    
    async def get_session(self) -> AsyncSession:
        """Get a database session"""
        return await self.get()
    
    async def return_session(self, session: AsyncSession):
        """Return a database session to the pool"""
        await self.put(session)


class HTTPClientPool(ConnectionPool):
    """HTTP client connection pool"""
    
    def __init__(self, timeout: int = 30, max_connections: int = 100):
        
    """__init__ function."""
super().__init__(max_connections, max_connections // 2)
        self.timeout = timeout
        self.session = None
    
    async def initialize(self) -> Any:
        """Initialize the HTTP client pool"""
        connector = aiohttp.TCPConnector(
            limit=self.pool_size,
            limit_per_host=self.pool_size // 4,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
    
    async def _create_connection(self) -> aiohttp.ClientSession:
        """Create a new HTTP session"""
        return self.session
    
    async def _close_connection(self, session: aiohttp.ClientSession):
        """Close an HTTP session"""
        # Don't close the main session, just return it
        pass
    
    async def get_client(self) -> aiohttp.ClientSession:
        """Get an HTTP client"""
        return await self.get()
    
    async def return_client(self, client: aiohttp.ClientSession):
        """Return an HTTP client to the pool"""
        await self.put(client)


class RedisPool(ConnectionPool):
    """Redis connection pool"""
    
    def __init__(self, redis_url: str, pool_size: int = 20):
        
    """__init__ function."""
super().__init__(pool_size, pool_size // 2)
        self.redis_url = redis_url
        self.redis_pool = None
    
    async def initialize(self) -> Any:
        """Initialize the Redis pool"""
        self.redis_pool = aioredis.from_url(
            self.redis_url,
            max_connections=self.pool_size,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def _create_connection(self) -> aioredis.Redis:
        """Create a new Redis connection"""
        return self.redis_pool
    
    async def _close_connection(self, conn: aioredis.Redis):
        """Close a Redis connection"""
        # Don't close the main pool, just return it
        pass
    
    async def get_redis(self) -> aioredis.Redis:
        """Get a Redis client"""
        return await self.get()
    
    async def return_redis(self, redis: aioredis.Redis):
        """Return a Redis client to the pool"""
        await self.put(redis)


class AIModelPool:
    """AI model instance pool"""
    
    def __init__(self, model_cache_size: int = 5, memory_limit_mb: int = 2048):
        
    """__init__ function."""
self.model_cache_size = model_cache_size
        self.memory_limit_mb = memory_limit_mb
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ResourceInfo] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(model_cache_size)
    
    async def get_model(self, model_name: str, loader_func: Callable[[], Any]) -> Optional[Dict[str, Any]]:
        """Get an AI model instance"""
        async with self._semaphore:
            async with self._lock:
                if model_name in self.models:
                    # Update access info
                    self.model_info[model_name].access_count += 1
                    self.model_info[model_name].last_accessed = time.time()
                    return self.models[model_name]
                
                # Load new model
                try:
                    model = await self._load_model(model_name, loader_func)
                    self.models[model_name] = model
                    self.model_info[model_name] = ResourceInfo(
                        name=model_name,
                        resource_type=ResourceType.AI_MODEL,
                        state=ResourceState.READY
                    )
                    return model
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    raise
    
    async def _load_model(self, model_name: str, loader_func: Callable[[], Any]) -> Any:
        """Load an AI model"""
        if asyncio.iscoroutinefunction(loader_func):
            return await loader_func()
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, loader_func)
    
    async def unload_model(self, model_name: str):
        """Unload an AI model"""
        async with self._lock:
            if model_name in self.models:
                del self.models[model_name]
                del self.model_info[model_name]
                # Force garbage collection
                gc.collect()
    
    async def get_model_info(self) -> Dict[str, ResourceInfo]:
        """Get information about loaded models"""
        return self.model_info.copy()


class SharedCache:
    """Shared cache with multiple backends"""
    
    def __init__(self, cache_size: int = 10000, ttl: int = 3600):
        
    """__init__ function."""
self.cache_size = cache_size
        self.ttl = ttl
        self.memory_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
    
    async def initialize(self) -> Any:
        """Initialize the cache"""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        async with self._lock:
            if key in self.memory_cache:
                # Check if expired
                if time.time() - self.cache_timestamps[key] > self.ttl:
                    del self.memory_cache[key]
                    del self.cache_timestamps[key]
                    return None
                
                return self.memory_cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in cache"""
        async with self._lock:
            # Check cache size
            if len(self.memory_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = min(self.cache_timestamps.keys(), 
                               key=lambda k: self.cache_timestamps[k])
                del self.memory_cache[oldest_key]
                del self.cache_timestamps[oldest_key]
            
            self.memory_cache[key] = value
            self.cache_timestamps[key] = time.time()
    
    async def delete(self, key: str):
        """Delete a value from cache"""
        async with self._lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
                del self.cache_timestamps[key]
    
    async def clear(self) -> Any:
        """Clear all cache"""
        async with self._lock:
            self.memory_cache.clear()
            self.cache_timestamps.clear()
    
    async def _cleanup_expired(self) -> Any:
        """Clean up expired cache entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = time.time()
                expired_keys = []
                
                async with self._lock:
                    for key, timestamp in self.cache_timestamps.items():
                        if current_time - timestamp > self.ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.memory_cache[key]
                        del self.cache_timestamps[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            return {
                "size": len(self.memory_cache),
                "max_size": self.cache_size,
                "ttl": self.ttl,
                "usage_percent": (len(self.memory_cache) / self.cache_size) * 100
            }


class SharedResources:
    """Main shared resources manager"""
    
    def __init__(self, config: ResourceConfig):
        
    """__init__ function."""
self.config = config
        self.resources: Dict[str, Any] = {}
        self.resource_info: Dict[str, ResourceInfo] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._shutdown = False
        
        # Initialize pools
        self.database_pool = DatabasePool(
            config.database_url,
            config.database_pool_size,
            config.database_max_overflow
        )
        
        self.http_pool = HTTPClientPool(
            config.http_timeout,
            config.http_max_connections
        )
        
        self.redis_pool = RedisPool(
            config.redis_url,
            config.redis_pool_size
        )
        
        self.ai_model_pool = AIModelPool(
            config.model_cache_size,
            config.model_memory_limit_mb
        )
        
        self.shared_cache = SharedCache(
            config.cache_size,
            config.cache_ttl
        )
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "database_connections": 0,
            "http_requests": 0,
            "redis_operations": 0,
            "model_loads": 0,
            "errors": 0
        }
    
    async def initialize(self) -> Any:
        """Initialize all shared resources"""
        if self._initialized:
            return
        
        logger.info("🚀 Initializing shared resources...")
        
        try:
            # Initialize database pool
            await self.database_pool.initialize()
            logger.info("✅ Database pool initialized")
            
            # Initialize HTTP client pool
            await self.http_pool.initialize()
            logger.info("✅ HTTP client pool initialized")
            
            # Initialize Redis pool
            await self.redis_pool.initialize()
            logger.info("✅ Redis pool initialized")
            
            # Initialize shared cache
            await self.shared_cache.initialize()
            logger.info("✅ Shared cache initialized")
            
            # Start background tasks
            if self.config.enable_monitoring:
                self._background_tasks.add(
                    asyncio.create_task(self._monitor_resources())
                )
                self._background_tasks.add(
                    asyncio.create_task(self._health_check())
                )
            
            self._initialized = True
            logger.info("✅ All shared resources initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize shared resources: {e}")
            raise
    
    async def shutdown(self) -> Any:
        """Shutdown all shared resources"""
        if self._shutdown:
            return
        
        logger.info("🛑 Shutting down shared resources...")
        self._shutdown = True
        
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close pools
            await self.database_pool.close_all()
            await self.http_pool.close_all()
            await self.redis_pool.close_all()
            await self.shared_cache.clear()
            
            logger.info("✅ All shared resources shut down successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    async def get_database_session(self) -> AsyncSession:
        """Get a database session"""
        self.stats["database_connections"] += 1
        return await self.database_pool.get_session()
    
    async def return_database_session(self, session: AsyncSession):
        """Return a database session"""
        await self.database_pool.return_session(session)
    
    async async def get_http_client(self) -> aiohttp.ClientSession:
        """Get an HTTP client"""
        self.stats["http_requests"] += 1
        return await self.http_pool.get_client()
    
    async def return_http_client(self, client: aiohttp.ClientSession):
        """Return an HTTP client"""
        await self.http_pool.return_client(client)
    
    async def get_redis_client(self) -> aioredis.Redis:
        """Get a Redis client"""
        self.stats["redis_operations"] += 1
        return await self.redis_pool.get_redis()
    
    async def return_redis_client(self, redis: aioredis.Redis):
        """Return a Redis client"""
        await self.redis_pool.return_redis(redis)
    
    async def get_ai_model(self, model_name: str, loader_func: Callable[[], Any]) -> Optional[Dict[str, Any]]:
        """Get an AI model"""
        self.stats["model_loads"] += 1
        return await self.ai_model_pool.get_model(model_name, loader_func)
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        value = await self.shared_cache.get(key)
        if value is not None:
            self.stats["cache_hits"] += 1
        return value
    
    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in cache"""
        await self.shared_cache.set(key, value, ttl)
    
    async def _monitor_resources(self) -> Any:
        """Monitor resource usage"""
        while not self._shutdown:
            try:
                # Get system metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                # Update resource info
                async with self._lock:
                    for resource_name, resource_info in self.resource_info.items():
                        resource_info.memory_usage_mb = memory_info.percent
                        resource_info.cpu_usage_percent = cpu_percent
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _health_check(self) -> Any:
        """Health check for resources"""
        while not self._shutdown:
            try:
                # Check database connectivity
                try:
                    session = await self.get_database_session()
                    await session.close()
                    await self.return_database_session(session)
                except Exception as e:
                    logger.warning(f"Database health check failed: {e}")
                
                # Check Redis connectivity
                try:
                    redis = await self.get_redis_client()
                    await redis.ping()
                    await self.return_redis_client(redis)
                except Exception as e:
                    logger.warning(f"Redis health check failed: {e}")
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        cache_stats = await self.shared_cache.get_stats()
        model_info = await self.ai_model_pool.get_model_info()
        
        return {
            **self.stats,
            "cache_stats": cache_stats,
            "model_info": model_info,
            "resource_info": self.resource_info.copy(),
            "initialized": self._initialized,
            "shutdown": self._shutdown
        }


# Global shared resources instance
shared_resources: Optional[SharedResources] = None


async def initialize_shared_resources(config: ResourceConfig) -> SharedResources:
    """Initialize global shared resources"""
    global shared_resources
    
    if shared_resources is None:
        shared_resources = SharedResources(config)
        await shared_resources.initialize()
    
    return shared_resources


async def get_shared_resources() -> SharedResources:
    """Get global shared resources instance"""
    if shared_resources is None:
        raise RuntimeError("Shared resources not initialized")
    return shared_resources


async def shutdown_shared_resources():
    """Shutdown global shared resources"""
    global shared_resources
    
    if shared_resources is not None:
        await shared_resources.shutdown()
        shared_resources = None


# Context managers for resource usage
@asynccontextmanager
async def database_session():
    """Context manager for database sessions"""
    resources = await get_shared_resources()
    session = await resources.get_database_session()
    try:
        yield session
    finally:
        await resources.return_database_session(session)


@asynccontextmanager
async def http_client():
    """Context manager for HTTP clients"""
    resources = await get_shared_resources()
    client = await resources.get_http_client()
    try:
        yield client
    finally:
        await resources.return_http_client(client)


@asynccontextmanager
async def redis_client():
    """Context manager for Redis clients"""
    resources = await get_shared_resources()
    client = await resources.get_redis_client()
    try:
        yield client
    finally:
        await resources.return_redis_client(client)


# Decorators for resource management
def with_cache(ttl: Optional[int] = None):
    """Decorator to cache function results"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            resources = await get_shared_resources()
            cached_result = await resources.get_cache(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await resources.set_cache(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def with_ai_model(model_name: str, loader_func: Callable[[], Any]):
    """Decorator to use AI models"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            resources = await get_shared_resources()
            model = await resources.get_ai_model(model_name, loader_func)
            
            # Add model to kwargs
            kwargs['model'] = model
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator 