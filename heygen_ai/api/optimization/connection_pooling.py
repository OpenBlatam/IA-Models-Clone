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
import weakref
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy import text
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
import httpx
import ssl
import certifi
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Connection Pooling and Async Optimization for HeyGen AI API
Efficient connection management for databases, HTTP clients, and external services.
"""


logger = structlog.get_logger()

# =============================================================================
# Connection Types
# =============================================================================

class ConnectionType(Enum):
    """Connection type enumeration."""
    DATABASE = "database"
    REDIS = "redis"
    HTTP = "http"
    MONGO = "mongo"
    EXTERNAL_API = "external_api"

class PoolStrategy(Enum):
    """Pool strategy enumeration."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"

@dataclass
class ConnectionConfig:
    """Connection configuration."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    retry_attempts: int = 3
    retry_delay: float = 1.0
    connection_timeout: int = 30
    read_timeout: int = 30
    keepalive_timeout: int = 60
    max_retries: int = 3
    backoff_factor: float = 0.3
    ssl_verify: bool = True
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    compression: bool = True
    gzip: bool = True
    brotli: bool = True

@dataclass
class PoolStats:
    """Connection pool statistics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    max_connections: int = 0
    connection_errors: int = 0
    timeout_errors: int = 0
    avg_connection_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# =============================================================================
# Base Connection Pool
# =============================================================================

class BaseConnectionPool:
    """Base connection pool class."""
    
    def __init__(
        self,
        config: ConnectionConfig,
        pool_strategy: PoolStrategy = PoolStrategy.STATIC
    ):
        
    """__init__ function."""
self.config = config
        self.pool_strategy = pool_strategy
        self.stats = PoolStats(max_connections=config.pool_size)
        self._connections: List[Any] = []
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Start cleanup task
        self._start_cleanup_task()
    
    async def get_connection(self) -> Optional[Dict[str, Any]]:
        """Get connection from pool."""
        async with self._lock:
            if self._connections:
                connection = self._connections.pop()
                self.stats.active_connections += 1
                self.stats.last_used = datetime.now(timezone.utc)
                return connection
            
            # Create new connection if pool not full
            if self.stats.total_connections < self.config.pool_size:
                connection = await self._create_connection()
                self.stats.total_connections += 1
                self.stats.active_connections += 1
                self.stats.last_used = datetime.now(timezone.utc)
                return connection
            
            # Wait for connection to become available
            return await self._wait_for_connection()
    
    async def return_connection(self, connection: Any):
        """Return connection to pool."""
        async with self._lock:
            if self.stats.active_connections > 0:
                self.stats.active_connections -= 1
            
            # Check if connection is still valid
            if await self._is_connection_valid(connection):
                self._connections.append(connection)
                self.stats.idle_connections = len(self._connections)
            else:
                # Remove invalid connection
                await self._close_connection(connection)
                self.stats.total_connections -= 1
    
    async def close_all(self) -> Any:
        """Close all connections in pool."""
        async with self._lock:
            for connection in self._connections:
                await self._close_connection(connection)
            
            self._connections.clear()
            self.stats.total_connections = 0
            self.stats.active_connections = 0
            self.stats.idle_connections = 0
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
    
    async def _create_connection(self) -> Any:
        """Create new connection (to be implemented by subclasses)."""
        raise NotImplementedError
    
    async def _close_connection(self, connection: Any):
        """Close connection (to be implemented by subclasses)."""
        raise NotImplementedError
    
    async def _is_connection_valid(self, connection: Any) -> bool:
        """Check if connection is valid (to be implemented by subclasses)."""
        raise NotImplementedError
    
    async def _wait_for_connection(self) -> Any:
        """Wait for connection to become available."""
        start_time = time.time()
        
        while time.time() - start_time < self.config.pool_timeout:
            await asyncio.sleep(0.1)
            
            if self._connections:
                connection = self._connections.pop()
                self.stats.active_connections += 1
                self.stats.last_used = datetime.now(timezone.utc)
                return connection
        
        raise TimeoutError("Connection pool timeout")
    
    def _start_cleanup_task(self) -> Any:
        """Start periodic cleanup task."""
        async def cleanup():
            
    """cleanup function."""
while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                    await self._cleanup_idle_connections()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup())
    
    async def _cleanup_idle_connections(self) -> Any:
        """Cleanup idle connections."""
        async with self._lock:
            current_time = datetime.now(timezone.utc)
            
            # Remove connections that have been idle too long
            connections_to_remove = []
            for connection in self._connections:
                if (current_time - self.stats.last_used).total_seconds() > self.config.pool_recycle:
                    connections_to_remove.append(connection)
            
            for connection in connections_to_remove:
                self._connections.remove(connection)
                await self._close_connection(connection)
                self.stats.total_connections -= 1
            
            self.stats.idle_connections = len(self._connections)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_connections": self.stats.total_connections,
            "active_connections": self.stats.active_connections,
            "idle_connections": self.stats.idle_connections,
            "max_connections": self.stats.max_connections,
            "connection_errors": self.stats.connection_errors,
            "timeout_errors": self.stats.timeout_errors,
            "avg_connection_time_ms": self.stats.avg_connection_time_ms,
            "avg_query_time_ms": self.stats.avg_query_time_ms,
            "last_used": self.stats.last_used.isoformat() if self.stats.last_used else None,
            "created_at": self.stats.created_at.isoformat(),
            "pool_strategy": self.pool_strategy.value
        }

# =============================================================================
# Database Connection Pool
# =============================================================================

class DatabaseConnectionPool(BaseConnectionPool):
    """Database connection pool using SQLAlchemy."""
    
    def __init__(self, config: ConnectionConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Create SQLAlchemy engine
        self.engine = create_async_engine(
            config.url,
            poolclass=QueuePool,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
            pool_pre_ping=True,
            pool_reset_on_return='commit',
            echo=False,
            connect_args={
                "server_settings": {
                    "jit": "off",
                    "statement_timeout": f"{config.connection_timeout * 1000}",
                    "lock_timeout": "10000",
                }
            }
        )
        
        # Session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    @asynccontextmanager
    async def get_session(self) -> Optional[Dict[str, Any]]:
        """Get database session with automatic cleanup."""
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute query with connection pooling."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                # Set query timeout
                if timeout:
                    await session.execute(text(f"SET statement_timeout = {timeout * 1000}"))
                
                # Execute query
                result = await session.execute(text(query), params or {})
                
                # Convert to list of dictionaries
                rows = []
                for row in result:
                    rows.append(dict(row._mapping))
                
                # Update statistics
                duration_ms = (time.time() - start_time) * 1000
                self._update_query_stats(duration_ms)
                
                return rows
                
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error(f"Database query error: {e}")
            raise
    
    async def execute_batch(
        self,
        queries: List[str],
        params: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 1000
    ) -> List[List[Dict[str, Any]]]:
        """Execute batch of queries efficiently."""
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_params = params[i:i + batch_size] if params else [{}] * len(batch_queries)
            
            # Execute batch in parallel
            tasks = [
                self.execute_query(query, param)
                for query, param in zip(batch_queries, batch_params)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    async def _create_connection(self) -> AsyncSession:
        """Create new database connection."""
        return self.session_factory()
    
    async def _close_connection(self, connection: AsyncSession):
        """Close database connection."""
        await connection.close()
    
    async def _is_connection_valid(self, connection: AsyncSession) -> bool:
        """Check if database connection is valid."""
        try:
            await connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    def _update_query_stats(self, duration_ms: float):
        """Update query statistics."""
        if self.stats.avg_query_time_ms == 0:
            self.stats.avg_query_time_ms = duration_ms
        else:
            self.stats.avg_query_time_ms = (
                (self.stats.avg_query_time_ms + duration_ms) / 2
            )

# =============================================================================
# Redis Connection Pool
# =============================================================================

class RedisConnectionPool(BaseConnectionPool):
    """Redis connection pool."""
    
    def __init__(self, config: ConnectionConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Create Redis connection pool
        self.redis_pool = redis.ConnectionPool.from_url(
            config.url,
            max_connections=config.pool_size,
            retry_on_timeout=True,
            socket_connect_timeout=config.connection_timeout,
            socket_timeout=config.read_timeout,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
            health_check_interval=30
        )
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client from pool."""
        return redis.Redis(connection_pool=self.redis_pool)
    
    async def execute_command(
        self,
        command: str,
        *args,
        timeout: Optional[int] = None
    ) -> Any:
        """Execute Redis command with connection pooling."""
        start_time = time.time()
        
        try:
            client = await self.get_redis_client()
            
            # Execute command
            if timeout:
                result = await asyncio.wait_for(
                    client.execute_command(command, *args),
                    timeout=timeout
                )
            else:
                result = await client.execute_command(command, *args)
            
            # Update statistics
            duration_ms = (time.time() - start_time) * 1000
            self._update_query_stats(duration_ms)
            
            return result
            
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error(f"Redis command error: {e}")
            raise
    
    async def _create_connection(self) -> redis.Redis:
        """Create new Redis connection."""
        return redis.Redis(connection_pool=self.redis_pool)
    
    async def _close_connection(self, connection: redis.Redis):
        """Close Redis connection."""
        await connection.close()
    
    async def _is_connection_valid(self, connection: redis.Redis) -> bool:
        """Check if Redis connection is valid."""
        try:
            await connection.ping()
            return True
        except Exception:
            return False
    
    def _update_query_stats(self, duration_ms: float):
        """Update query statistics."""
        if self.stats.avg_query_time_ms == 0:
            self.stats.avg_query_time_ms = duration_ms
        else:
            self.stats.avg_query_time_ms = (
                (self.stats.avg_query_time_ms + duration_ms) / 2
            )

# =============================================================================
# HTTP Connection Pool
# =============================================================================

class HTTPConnectionPool(BaseConnectionPool):
    """HTTP connection pool using aiohttp."""
    
    def __init__(self, config: ConnectionConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # SSL context
        ssl_context = None
        if config.ssl_verify:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            if config.ssl_cert and config.ssl_key:
                ssl_context.load_cert_chain(config.ssl_cert, config.ssl_key)
        
        # Create connector
        self.connector = aiohttp.TCPConnector(
            limit=config.pool_size,
            limit_per_host=config.pool_size // 2,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=ssl_context,
            keepalive_timeout=config.keepalive_timeout,
            enable_cleanup_closed=True
        )
        
        # Create session
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(
                total=config.connection_timeout,
                connect=config.connection_timeout,
                sock_read=config.read_timeout
            ),
            headers={
                "Accept-Encoding": "gzip, deflate, br" if config.compression else "identity"
            }
        )
    
    async async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make HTTP request with connection pooling."""
        start_time = time.time()
        
        try:
            # Add retry logic
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.session.request(method, url, **kwargs)
                    
                    # Update statistics
                    duration_ms = (time.time() - start_time) * 1000
                    self._update_query_stats(duration_ms)
                    
                    return response
                    
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    
                    # Exponential backoff
                    delay = self.config.retry_delay * (self.config.backoff_factor ** attempt)
                    await asyncio.sleep(delay)
            
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error(f"HTTP request error: {e}")
            raise
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make GET request."""
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make POST request."""
        return await self.request("POST", url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make PUT request."""
        return await self.request("PUT", url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make DELETE request."""
        return await self.request("DELETE", url, **kwargs)
    
    async def _create_connection(self) -> aiohttp.ClientSession:
        """Create new HTTP connection."""
        return self.session
    
    async def _close_connection(self, connection: aiohttp.ClientSession):
        """Close HTTP connection."""
        await connection.close()
    
    async def _is_connection_valid(self, connection: aiohttp.ClientSession) -> bool:
        """Check if HTTP connection is valid."""
        return not connection.closed
    
    def _update_query_stats(self, duration_ms: float):
        """Update query statistics."""
        if self.stats.avg_query_time_ms == 0:
            self.stats.avg_query_time_ms = duration_ms
        else:
            self.stats.avg_query_time_ms = (
                (self.stats.avg_query_time_ms + duration_ms) / 2
            )
    
    async def close_all(self) -> Any:
        """Close all HTTP connections."""
        await super().close_all()
        await self.session.close()

# =============================================================================
# MongoDB Connection Pool
# =============================================================================

class MongoDBConnectionPool(BaseConnectionPool):
    """MongoDB connection pool using motor."""
    
    def __init__(self, config: ConnectionConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Create MongoDB client
        self.client = AsyncIOMotorClient(
            config.url,
            maxPoolSize=config.pool_size,
            minPoolSize=1,
            maxIdleTimeMS=config.pool_recycle * 1000,
            serverSelectionTimeoutMS=config.connection_timeout * 1000,
            socketTimeoutMS=config.read_timeout * 1000,
            connectTimeoutMS=config.connection_timeout * 1000,
            retryWrites=True,
            retryReads=True
        )
        
        # Get database
        self.db = self.client.get_default_database()
    
    async def execute_command(
        self,
        collection: str,
        command: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute MongoDB command with connection pooling."""
        start_time = time.time()
        
        try:
            # Get collection
            coll = self.db[collection]
            
            # Execute command
            if command == "find":
                result = await coll.find(*args, **kwargs).to_list(length=None)
            elif command == "find_one":
                result = await coll.find_one(*args, **kwargs)
            elif command == "insert_one":
                result = await coll.insert_one(*args, **kwargs)
            elif command == "insert_many":
                result = await coll.insert_many(*args, **kwargs)
            elif command == "update_one":
                result = await coll.update_one(*args, **kwargs)
            elif command == "update_many":
                result = await coll.update_many(*args, **kwargs)
            elif command == "delete_one":
                result = await coll.delete_one(*args, **kwargs)
            elif command == "delete_many":
                result = await coll.delete_many(*args, **kwargs)
            elif command == "aggregate":
                result = await coll.aggregate(*args, **kwargs).to_list(length=None)
            else:
                raise ValueError(f"Unknown command: {command}")
            
            # Update statistics
            duration_ms = (time.time() - start_time) * 1000
            self._update_query_stats(duration_ms)
            
            return result
            
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error(f"MongoDB command error: {e}")
            raise
    
    async def _create_connection(self) -> Any:
        """Create new MongoDB connection."""
        return self.client
    
    async def _close_connection(self, connection: Any):
        """Close MongoDB connection."""
        await connection.close()
    
    async def _is_connection_valid(self, connection: Any) -> bool:
        """Check if MongoDB connection is valid."""
        try:
            await connection.admin.command('ping')
            return True
        except Exception:
            return False
    
    def _update_query_stats(self, duration_ms: float):
        """Update query statistics."""
        if self.stats.avg_query_time_ms == 0:
            self.stats.avg_query_time_ms = duration_ms
        else:
            self.stats.avg_query_time_ms = (
                (self.stats.avg_query_time_ms + duration_ms) / 2
            )

# =============================================================================
# Connection Pool Manager
# =============================================================================

class ConnectionPoolManager:
    """Manages multiple connection pools."""
    
    def __init__(self) -> Any:
        self.pools: Dict[str, BaseConnectionPool] = {}
        self._lock = asyncio.Lock()
    
    async def add_pool(
        self,
        name: str,
        pool_type: ConnectionType,
        config: ConnectionConfig
    ) -> BaseConnectionPool:
        """Add connection pool."""
        async with self._lock:
            if name in self.pools:
                raise ValueError(f"Pool '{name}' already exists")
            
            if pool_type == ConnectionType.DATABASE:
                pool = DatabaseConnectionPool(config)
            elif pool_type == ConnectionType.REDIS:
                pool = RedisConnectionPool(config)
            elif pool_type == ConnectionType.HTTP:
                pool = HTTPConnectionPool(config)
            elif pool_type == ConnectionType.MONGO:
                pool = MongoDBConnectionPool(config)
            else:
                raise ValueError(f"Unsupported pool type: {pool_type}")
            
            self.pools[name] = pool
            return pool
    
    async def get_pool(self, name: str) -> Optional[BaseConnectionPool]:
        """Get connection pool by name."""
        return self.pools.get(name)
    
    async def remove_pool(self, name: str):
        """Remove connection pool."""
        async with self._lock:
            if name in self.pools:
                pool = self.pools[name]
                await pool.close_all()
                del self.pools[name]
    
    async def close_all(self) -> Any:
        """Close all connection pools."""
        async with self._lock:
            for pool in self.pools.values():
                await pool.close_all()
            self.pools.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        return {
            name: pool.get_stats()
            for name, pool in self.pools.items()
        }
    
    async def health_check(self) -> Dict[str, str]:
        """Perform health check of all pools."""
        health_status = {}
        
        for name, pool in self.pools.items():
            try:
                # Try to get a connection
                if isinstance(pool, DatabaseConnectionPool):
                    async with pool.get_session() as session:
                        await session.execute(text("SELECT 1"))
                elif isinstance(pool, RedisConnectionPool):
                    client = await pool.get_redis_client()
                    await client.ping()
                elif isinstance(pool, HTTPConnectionPool):
                    # HTTP pools don't need explicit health check
                    pass
                elif isinstance(pool, MongoDBConnectionPool):
                    await pool.client.admin.command('ping')
                
                health_status[name] = "healthy"
                
            except Exception as e:
                health_status[name] = f"unhealthy: {str(e)}"
        
        return health_status

# =============================================================================
# Async Optimization Utilities
# =============================================================================

class AsyncOptimizer:
    """Async optimization utilities."""
    
    def __init__(self, max_concurrent_tasks: int = 100):
        
    """__init__ function."""
self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.task_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "active_tasks": 0
        }
    
    async def run_concurrent(
        self,
        tasks: List[Callable],
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Run tasks concurrently with controlled concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent or self.max_concurrent_tasks)
        
        async def run_task(task) -> Any:
            async with semaphore:
                self.task_stats["total_tasks"] += 1
                self.task_stats["active_tasks"] += 1
                
                try:
                    result = await task()
                    self.task_stats["completed_tasks"] += 1
                    return result
                except Exception as e:
                    self.task_stats["failed_tasks"] += 1
                    logger.error(f"Task failed: {e}")
                    raise
                finally:
                    self.task_stats["active_tasks"] -= 1
        
        return await asyncio.gather(*[run_task(task) for task in tasks], return_exceptions=True)
    
    async def run_batch(
        self,
        items: List[Any],
        processor: Callable,
        batch_size: int = 100,
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Process items in batches with controlled concurrency."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Create tasks for batch
            tasks = [processor(item) for item in batch]
            
            # Run batch concurrently
            batch_results = await self.run_concurrent(tasks, max_concurrent)
            results.extend(batch_results)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        return {
            **self.task_stats,
            "success_rate": (
                self.task_stats["completed_tasks"] / self.task_stats["total_tasks"]
                if self.task_stats["total_tasks"] > 0 else 0
            )
        }

# =============================================================================
# FastAPI Integration
# =============================================================================

def get_connection_pool_manager() -> ConnectionPoolManager:
    """Dependency to get connection pool manager."""
    # This would be configured in your FastAPI app
    return ConnectionPoolManager()

def get_async_optimizer() -> AsyncOptimizer:
    """Dependency to get async optimizer."""
    return AsyncOptimizer()

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ConnectionType",
    "PoolStrategy",
    "ConnectionConfig",
    "PoolStats",
    "BaseConnectionPool",
    "DatabaseConnectionPool",
    "RedisConnectionPool",
    "HTTPConnectionPool",
    "MongoDBConnectionPool",
    "ConnectionPoolManager",
    "AsyncOptimizer",
    "get_connection_pool_manager",
    "get_async_optimizer",
] 