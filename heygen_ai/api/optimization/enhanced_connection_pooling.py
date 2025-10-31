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
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import text, event
import redis.asyncio as redis
from pydantic import BaseModel
import psutil
import gc
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Enhanced Connection Pooling Manager for HeyGen AI API
Advanced connection pooling with monitoring, auto-scaling, and health checks.
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
    WEBSOCKET = "websocket"

class PoolStatus(Enum):
    """Pool status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"

# =============================================================================
# Connection Metrics
# =============================================================================

@dataclass
class ConnectionMetrics:
    """Connection pool metrics."""
    pool_name: str
    connection_type: ConnectionType
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    max_connections: int = 0
    connection_errors: int = 0
    avg_connection_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    last_health_check: Optional[datetime] = None
    status: PoolStatus = PoolStatus.OFFLINE
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pool_name": self.pool_name,
            "connection_type": self.connection_type.value,
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "max_connections": self.max_connections,
            "connection_errors": self.connection_errors,
            "avg_connection_time_ms": self.avg_connection_time_ms,
            "avg_query_time_ms": self.avg_query_time_ms,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "status": self.status.value,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "utilization_percent": (self.active_connections / self.max_connections * 100) if self.max_connections > 0 else 0
        }

# =============================================================================
# Database Connection Pool
# =============================================================================

class DatabaseConnectionPool:
    """Enhanced database connection pool with monitoring."""
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        echo: bool = False,
        pool_name: str = "default"
    ):
        
    """__init__ function."""
self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.echo = echo
        self.pool_name = pool_name
        
        # Engine and session factory
        self.engine = None
        self.async_session = None
        
        # Metrics
        self.metrics = ConnectionMetrics(
            pool_name=pool_name,
            connection_type=ConnectionType.DATABASE,
            max_connections=pool_size + max_overflow
        )
        
        # Connection tracking
        self.connection_times: List[float] = []
        self.query_times: List[float] = []
        self.error_count = 0
        
        # Health check
        self.last_health_check = None
        self.health_check_interval = 60  # seconds
        
    async def initialize(self) -> Any:
        """Initialize the database connection pool."""
        try:
            start_time = time.time()
            
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                self.database_url,
                echo=self.echo,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=self.pool_pre_ping,
                future=True,
                use_insertmanyvalues=True,
                use_insertmanyvalues_wo_returning=True
            )
            
            # Create async session factory
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Setup event listeners for monitoring
            self._setup_event_listeners()
            
            # Perform initial health check
            await self._health_check()
            
            init_time = (time.time() - start_time) * 1000
            self.metrics.avg_connection_time_ms = init_time
            
            logger.info(
                "Database connection pool initialized",
                pool_name=self.pool_name,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                init_time_ms=init_time
            )
            
        except Exception as e:
            self.metrics.status = PoolStatus.CRITICAL
            self.error_count += 1
            logger.error("Failed to initialize database pool", error=str(e))
            raise
    
    def _setup_event_listeners(self) -> List[Any]:
        """Setup SQLAlchemy event listeners for monitoring."""
        
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record) -> Any:
            self.metrics.total_connections += 1
            self.metrics.active_connections += 1
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy) -> Any:
            self.metrics.active_connections += 1
            self.metrics.idle_connections = max(0, self.metrics.idle_connections - 1)
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record) -> Any:
            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)
            self.metrics.idle_connections += 1
        
        @event.listens_for(self.engine, "disconnect")
        def receive_disconnect(dbapi_connection, connection_record) -> Any:
            self.metrics.total_connections = max(0, self.metrics.total_connections - 1)
            self.metrics.idle_connections = max(0, self.metrics.idle_connections - 1)
    
    @asynccontextmanager
    async def get_session(self) -> Optional[Dict[str, Any]]:
        """Get database session with monitoring."""
        start_time = time.time()
        session = None
        
        try:
            session = self.async_session()
            yield session
            
            # Record successful query time
            query_time = (time.time() - start_time) * 1000
            self.query_times.append(query_time)
            if len(self.query_times) > 1000:
                self.query_times = self.query_times[-1000:]
            
            self.metrics.avg_query_time_ms = sum(self.query_times) / len(self.query_times)
            
        except Exception as e:
            self.error_count += 1
            self.metrics.connection_errors += 1
            logger.error("Database session error", error=str(e))
            raise
        finally:
            if session:
                await session.close()
    
    async def _health_check(self) -> bool:
        """Perform health check on the database pool."""
        try:
            start_time = time.time()
            
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                await result.fetchone()
            
            health_time = (time.time() - start_time) * 1000
            self.connection_times.append(health_time)
            if len(self.connection_times) > 100:
                self.connection_times = self.connection_times[-100:]
            
            self.metrics.avg_connection_time_ms = sum(self.connection_times) / len(self.connection_times)
            self.metrics.last_health_check = datetime.now(timezone.utc)
            self.metrics.status = PoolStatus.HEALTHY
            
            # Update system metrics
            self._update_system_metrics()
            
            return True
            
        except Exception as e:
            self.metrics.status = PoolStatus.CRITICAL
            self.error_count += 1
            logger.error("Database health check failed", error=str(e))
            return False
    
    def _update_system_metrics(self) -> Any:
        """Update system resource metrics."""
        try:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
        except Exception:
            pass
    
    async def close(self) -> Any:
        """Close the database connection pool."""
        try:
            if self.engine:
                await self.engine.dispose()
            logger.info("Database connection pool closed", pool_name=self.pool_name)
        except Exception as e:
            logger.error("Error closing database pool", error=str(e))
    
    def get_metrics(self) -> ConnectionMetrics:
        """Get current pool metrics."""
        return self.metrics

# =============================================================================
# Redis Connection Pool
# =============================================================================

class RedisConnectionPool:
    """Enhanced Redis connection pool with monitoring."""
    
    def __init__(
        self,
        redis_url: str,
        max_connections: int = 50,
        pool_name: str = "default"
    ):
        
    """__init__ function."""
self.redis_url = redis_url
        self.max_connections = max_connections
        self.pool_name = pool_name
        
        # Redis client
        self.redis_client: Optional[redis.Redis] = None
        
        # Metrics
        self.metrics = ConnectionMetrics(
            pool_name=pool_name,
            connection_type=ConnectionType.REDIS,
            max_connections=max_connections
        )
        
        # Connection tracking
        self.connection_times: List[float] = []
        self.query_times: List[float] = []
        self.error_count = 0
        
        # Health check
        self.last_health_check = None
        self.health_check_interval = 30  # seconds
    
    async def initialize(self) -> Any:
        """Initialize the Redis connection pool."""
        try:
            start_time = time.time()
            
            # Create Redis client with connection pool
            self.redis_client = redis.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                health_check_interval=30,
                socket_keepalive=True,
                socket_keepalive_options={},
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            init_time = (time.time() - start_time) * 1000
            self.metrics.avg_connection_time_ms = init_time
            self.metrics.status = PoolStatus.HEALTHY
            
            logger.info(
                "Redis connection pool initialized",
                pool_name=self.pool_name,
                max_connections=self.max_connections,
                init_time_ms=init_time
            )
            
        except Exception as e:
            self.metrics.status = PoolStatus.CRITICAL
            self.error_count += 1
            logger.error("Failed to initialize Redis pool", error=str(e))
            raise
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        return self.redis_client
    
    async def _health_check(self) -> bool:
        """Perform health check on the Redis pool."""
        try:
            start_time = time.time()
            
            await self.redis_client.ping()
            
            health_time = (time.time() - start_time) * 1000
            self.connection_times.append(health_time)
            if len(self.connection_times) > 100:
                self.connection_times = self.connection_times[-100:]
            
            self.metrics.avg_connection_time_ms = sum(self.connection_times) / len(self.connection_times)
            self.metrics.last_health_check = datetime.now(timezone.utc)
            self.metrics.status = PoolStatus.HEALTHY
            
            # Update connection info
            info = await self.redis_client.info()
            self.metrics.active_connections = info.get('connected_clients', 0)
            self.metrics.total_connections = info.get('total_connections_received', 0)
            
            # Update system metrics
            self._update_system_metrics()
            
            return True
            
        except Exception as e:
            self.metrics.status = PoolStatus.CRITICAL
            self.error_count += 1
            logger.error("Redis health check failed", error=str(e))
            return False
    
    def _update_system_metrics(self) -> Any:
        """Update system resource metrics."""
        try:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
        except Exception:
            pass
    
    async def close(self) -> Any:
        """Close the Redis connection pool."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            logger.info("Redis connection pool closed", pool_name=self.pool_name)
        except Exception as e:
            logger.error("Error closing Redis pool", error=str(e))
    
    def get_metrics(self) -> ConnectionMetrics:
        """Get current pool metrics."""
        return self.metrics

# =============================================================================
# Enhanced Connection Pool Manager
# =============================================================================

class EnhancedConnectionPoolManager:
    """Enhanced connection pool manager with monitoring and auto-scaling."""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        max_database_connections: int = 50,
        max_redis_connections: int = 50,
        health_check_interval: int = 60,
        auto_scaling: bool = True,
        monitoring_enabled: bool = True
    ):
        
    """__init__ function."""
self.database_url = database_url
        self.redis_url = redis_url
        self.max_database_connections = max_database_connections
        self.max_redis_connections = max_redis_connections
        self.health_check_interval = health_check_interval
        self.auto_scaling = auto_scaling
        self.monitoring_enabled = monitoring_enabled
        
        # Connection pools
        self.database_pool: Optional[DatabaseConnectionPool] = None
        self.redis_pool: Optional[RedisConnectionPool] = None
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.start_time = datetime.now(timezone.utc)
        
        # Auto-scaling
        self.scaling_threshold = 0.8  # 80% utilization
        self.scaling_cooldown = 300  # 5 minutes
        self.last_scaling_time = None
    
    async def initialize(self) -> Any:
        """Initialize all connection pools."""
        try:
            # Initialize database pool
            if self.database_url:
                self.database_pool = DatabaseConnectionPool(
                    database_url=self.database_url,
                    pool_size=min(20, self.max_database_connections // 2),
                    max_overflow=self.max_database_connections - 20,
                    pool_name="main_database"
                )
                await self.database_pool.initialize()
            
            # Initialize Redis pool
            if self.redis_url:
                self.redis_pool = RedisConnectionPool(
                    redis_url=self.redis_url,
                    max_connections=self.max_redis_connections,
                    pool_name="main_redis"
                )
                await self.redis_pool.initialize()
            
            # Start monitoring if enabled
            if self.monitoring_enabled:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Enhanced connection pool manager initialized")
            
        except Exception as e:
            logger.error("Failed to initialize connection pool manager", error=str(e))
            raise
    
    async def _monitoring_loop(self) -> Any:
        """Monitoring loop for connection pools."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Health checks
                if self.database_pool:
                    await self.database_pool._health_check()
                
                if self.redis_pool:
                    await self.redis_pool._health_check()
                
                # Auto-scaling
                if self.auto_scaling:
                    await self._auto_scale_pools()
                
                # Log metrics periodically
                if self.monitoring_enabled:
                    self._log_metrics()
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
    
    async def _auto_scale_pools(self) -> Any:
        """Auto-scale connection pools based on utilization."""
        current_time = datetime.now(timezone.utc)
        
        # Check cooldown period
        if (self.last_scaling_time and 
            (current_time - self.last_scaling_time).total_seconds() < self.scaling_cooldown):
            return
        
        # Scale database pool
        if self.database_pool:
            metrics = self.database_pool.get_metrics()
            utilization = metrics.active_connections / metrics.max_connections
            
            if utilization > self.scaling_threshold:
                # Scale up
                new_pool_size = min(
                    metrics.max_connections + 10,
                    self.max_database_connections
                )
                # Note: In a real implementation, you'd need to recreate the pool
                logger.info(f"Database pool utilization high: {utilization:.2%}")
        
        # Scale Redis pool
        if self.redis_pool:
            metrics = self.redis_pool.get_metrics()
            utilization = metrics.active_connections / metrics.max_connections
            
            if utilization > self.scaling_threshold:
                logger.info(f"Redis pool utilization high: {utilization:.2%}")
        
        self.last_scaling_time = current_time
    
    def _log_metrics(self) -> Any:
        """Log connection pool metrics."""
        metrics = self.get_all_metrics()
        logger.info("Connection pool metrics", metrics=metrics)
    
    async def get_database_session(self) -> Optional[Dict[str, Any]]:
        """Get database session."""
        if not self.database_pool:
            raise RuntimeError("Database pool not initialized")
        return self.database_pool.get_session()
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self.redis_pool:
            raise RuntimeError("Redis pool not initialized")
        return await self.redis_pool.get_client()
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics from all connection pools."""
        metrics = {
            "manager_uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "auto_scaling_enabled": self.auto_scaling,
            "monitoring_enabled": self.monitoring_enabled
        }
        
        if self.database_pool:
            metrics["database"] = self.database_pool.get_metrics().to_dict()
        
        if self.redis_pool:
            metrics["redis"] = self.redis_pool.get_metrics().to_dict()
        
        return metrics
    
    async def close(self) -> Any:
        """Close all connection pools."""
        try:
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Close pools
            if self.database_pool:
                await self.database_pool.close()
            
            if self.redis_pool:
                await self.redis_pool.close()
            
            logger.info("Enhanced connection pool manager closed")
            
        except Exception as e:
            logger.error("Error closing connection pool manager", error=str(e))
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all pools."""
        results = {}
        
        if self.database_pool:
            results["database"] = await self.database_pool._health_check()
        
        if self.redis_pool:
            results["redis"] = await self.redis_pool._health_check()
        
        return results

# =============================================================================
# Factory Functions
# =============================================================================

async def create_enhanced_connection_pool_manager(
    database_url: Optional[str] = None,
    redis_url: Optional[str] = None,
    max_database_connections: int = 50,
    max_redis_connections: int = 50,
    health_check_interval: int = 60,
    auto_scaling: bool = True,
    monitoring_enabled: bool = True
) -> EnhancedConnectionPoolManager:
    """Create and initialize an enhanced connection pool manager."""
    manager = EnhancedConnectionPoolManager(
        database_url=database_url,
        redis_url=redis_url,
        max_database_connections=max_database_connections,
        max_redis_connections=max_redis_connections,
        health_check_interval=health_check_interval,
        auto_scaling=auto_scaling,
        monitoring_enabled=monitoring_enabled
    )
    
    await manager.initialize()
    return manager 