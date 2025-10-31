"""
🔄 CONNECTION POOL MANAGER v6.0.0 - ULTRA-FAST CONNECTIONS
==========================================================

High-performance connection pooling for the Blatam AI system:
- ⚡ Ultra-fast connection acquisition and release
- 🔄 Smart connection reuse and load balancing
- 📊 Connection health monitoring and auto-healing
- 🧵 Thread-safe connection management
- 💾 Memory-efficient connection pooling
- 🎯 Adaptive connection sizing
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic, Tuple
import uuid
import threading
from contextlib import asynccontextmanager
import aiohttp
import aioredis
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# 🎯 CONNECTION TYPES AND STATUS
# =============================================================================

class ConnectionType(Enum):
    """Types of connections supported."""
    HTTP = "http"
    DATABASE = "database"
    REDIS = "redis"
    WEBSOCKET = "websocket"
    CUSTOM = "custom"

class ConnectionStatus(Enum):
    """Connection lifecycle status."""
    IDLE = "idle"
    IN_USE = "in_use"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    HEALTH_CHECK = "health_check"

# =============================================================================
# 🎯 CONNECTION CONFIGURATION
# =============================================================================

@dataclass
class ConnectionConfig:
    """Configuration for connection pools."""
    connection_type: ConnectionType
    max_connections: int = 20
    min_connections: int = 5
    connection_timeout: float = 30.0
    pool_timeout: float = 10.0
    max_lifetime: float = 3600.0  # 1 hour
    health_check_interval: float = 60.0
    retry_attempts: int = 3
    enable_auto_scaling: bool = True
    enable_health_monitoring: bool = True
    enable_connection_reuse: bool = True
    connection_string: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = None
    extra_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'connection_type': self.connection_type.value,
            'max_connections': self.max_connections,
            'min_connections': self.min_connections,
            'connection_timeout': self.connection_timeout,
            'pool_timeout': self.pool_timeout,
            'max_lifetime': self.max_lifetime,
            'health_check_interval': self.health_check_interval,
            'retry_attempts': self.retry_attempts,
            'enable_auto_scaling': self.enable_auto_scaling,
            'enable_health_monitoring': self.enable_health_monitoring,
            'enable_connection_reuse': self.enable_connection_reuse,
            'connection_string': self.connection_string,
            'credentials': self.credentials,
            'extra_params': self.extra_params
        }

# =============================================================================
# 🎯 BASE CONNECTION INTERFACES
# =============================================================================

T = TypeVar('T')

class Connection(ABC, Generic[T]):
    """Abstract connection interface."""
    
    def __init__(self, connection_id: str, config: ConnectionConfig):
        self.connection_id = connection_id
        self.config = config
        self.status = ConnectionStatus.IDLE
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.error_count = 0
        self.last_error = None
        self._connection_obj: Optional[T] = None
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check connection health."""
        pass
    
    @abstractmethod
    async def execute(self, operation: str, *args, **kwargs) -> Any:
        """Execute operation using connection."""
        pass
    
    def is_expired(self) -> bool:
        """Check if connection has expired."""
        return time.time() - self.created_at > self.config.max_lifetime
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return (
            self.status == ConnectionStatus.CONNECTED and
            self.error_count < self.config.retry_attempts and
            not self.is_expired()
        )
    
    def mark_used(self) -> None:
        """Mark connection as used."""
        self.last_used = time.time()
        self.use_count += 1
        self.status = ConnectionStatus.IN_USE
    
    def mark_idle(self) -> None:
        """Mark connection as idle."""
        self.status = ConnectionStatus.IDLE
    
    def mark_error(self, error: Exception) -> None:
        """Mark connection as having an error."""
        self.error_count += 1
        self.last_error = error
        self.status = ConnectionStatus.ERROR

# =============================================================================
# 🎯 CONNECTION POOL
# =============================================================================

class ConnectionPool(Generic[T]):
    """High-performance connection pool."""
    
    def __init__(self, name: str, config: ConnectionConfig, connection_factory: Callable[[str, ConnectionConfig], Connection[T]]):
        self.name = name
        self.config = config
        self.connection_factory = connection_factory
        self.pool_id = str(uuid.uuid4())
        
        # Connection storage
        self.available_connections: List[Connection[T]] = []
        self.in_use_connections: List[Connection[T]] = []
        self.connecting_connections: List[Connection[T]] = []
        
        # Pool management
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_connections_created = 0
        self.total_connections_acquired = 0
        self.total_connections_released = 0
        self.total_connection_errors = 0
        
        # Initialize pool
        self._initialize_pool()
        
        logger.info(f"🔄 Connection Pool '{name}' initialized with ID: {self.pool_id}")
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        # Start background tasks
        if self.config.enable_health_monitoring:
            self._start_health_monitoring()
        
        if self.config.enable_auto_scaling:
            self._start_auto_scaling()
        
        # Start cleanup task
        self._start_cleanup_task()
    
    async def get_connection(self, timeout: Optional[float] = None) -> Connection[T]:
        """Get a connection from the pool."""
        timeout = timeout or self.config.pool_timeout
        start_time = time.time()
        
        async with self._lock:
            # Try to get an available connection
            if self.available_connections:
                connection = self.available_connections.pop()
                if connection.is_healthy():
                    connection.mark_used()
                    self.in_use_connections.append(connection)
                    self.total_connections_acquired += 1
                    return connection
                else:
                    # Remove unhealthy connection
                    await self._remove_connection(connection)
            
            # Create new connection if possible
            if len(self.in_use_connections) + len(self.connecting_connections) < self.config.max_connections:
                connection = await self._create_connection()
                if connection:
                    connection.mark_used()
                    self.in_use_connections.append(connection)
                    self.total_connections_acquired += 1
                    return connection
            
            # Wait for available connection
            while time.time() - start_time < timeout:
                await asyncio.sleep(0.001)  # Small delay
                
                # Check for newly available connections
                if self.available_connections:
                    connection = self.available_connections.pop()
                    if connection.is_healthy():
                        connection.mark_used()
                        self.in_use_connections.append(connection)
                        self.total_connections_acquired += 1
                        return connection
                
                # Check if we can create new connections
                if len(self.in_use_connections) + len(self.connecting_connections) < self.config.max_connections:
                    connection = await self._create_connection()
                    if connection:
                        connection.mark_used()
                        self.in_use_connections.append(connection)
                        self.total_connections_acquired += 1
                        return connection
            
            raise TimeoutError(f"Timeout waiting for connection in pool '{self.name}'")
    
    async def release_connection(self, connection: Connection[T]) -> None:
        """Release a connection back to the pool."""
        async with self._lock:
            if connection in self.in_use_connections:
                self.in_use_connections.remove(connection)
                
                if connection.is_healthy() and self.config.enable_connection_reuse:
                    connection.mark_idle()
                    self.available_connections.append(connection)
                    self.total_connections_released += 1
                else:
                    # Remove unhealthy connection
                    await self._remove_connection(connection)
    
    async def _create_connection(self) -> Optional[Connection[T]]:
        """Create a new connection."""
        try:
            connection_id = f"{self.name}_conn_{self.total_connections_created}"
            connection = self.connection_factory(connection_id, self.config)
            
            # Add to connecting list
            self.connecting_connections.append(connection)
            
            # Connect
            success = await connection.connect()
            if success:
                self.connecting_connections.remove(connection)
                self.total_connections_created += 1
                logger.debug(f"🔄 Created new connection: {connection_id}")
                return connection
            else:
                self.connecting_connections.remove(connection)
                self.total_connection_errors += 1
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to create connection: {e}")
            self.total_connection_errors += 1
            return None
    
    async def _remove_connection(self, connection: Connection[T]) -> None:
        """Remove a connection from the pool."""
        try:
            await connection.disconnect()
        except Exception as e:
            logger.warning(f"⚠️ Error disconnecting connection: {e}")
        
        # Remove from all lists
        if connection in self.available_connections:
            self.available_connections.remove(connection)
        if connection in self.in_use_connections:
            self.in_use_connections.remove(connection)
        if connection in self.connecting_connections:
            self.connecting_connections.remove(connection)
    
    def _start_health_monitoring(self) -> None:
        """Start periodic health monitoring."""
        async def health_monitor():
            while True:
                try:
                    await asyncio.sleep(self.config.health_check_interval)
                    await self._health_check_all_connections()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Health monitoring error: {e}")
        
        self._health_check_task = asyncio.create_task(health_monitor())
    
    async def _health_check_all_connections(self) -> None:
        """Check health of all connections."""
        async with self._lock:
            # Check available connections
            for connection in self.available_connections[:]:
                if not connection.is_healthy():
                    await self._remove_connection(connection)
            
            # Check in-use connections
            for connection in self.in_use_connections[:]:
                if not connection.is_healthy():
                    await self._remove_connection(connection)
    
    def _start_auto_scaling(self) -> None:
        """Start auto-scaling task."""
        async def auto_scaler():
            while True:
                try:
                    await asyncio.sleep(30)  # Check every 30 seconds
                    await self._adjust_pool_size()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Auto-scaling error: {e}")
        
        self._scaling_task = asyncio.create_task(auto_scaler())
    
    async def _adjust_pool_size(self) -> None:
        """Adjust pool size based on usage patterns."""
        total_connections = len(self.available_connections) + len(self.in_use_connections)
        usage_rate = len(self.in_use_connections) / max(1, total_connections)
        
        if usage_rate > 0.8 and total_connections < self.config.max_connections:
            # High usage - create more connections
            await self._create_connection()
        elif usage_rate < 0.2 and total_connections > self.config.min_connections:
            # Low usage - remove some connections
            if self.available_connections:
                connection = self.available_connections.pop()
                await self._remove_connection(connection)
    
    def _start_cleanup_task(self) -> None:
        """Start periodic cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Cleanup every minute
                    await self._cleanup_expired_connections()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Cleanup error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_expired_connections(self) -> None:
        """Cleanup expired connections."""
        async with self._lock:
            # Remove expired available connections
            for connection in self.available_connections[:]:
                if connection.is_expired():
                    await self._remove_connection(connection)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_id': self.pool_id,
            'name': self.name,
            'available_connections': len(self.available_connections),
            'in_use_connections': len(self.in_use_connections),
            'connecting_connections': len(self.connecting_connections),
            'total_connections': (
                len(self.available_connections) + 
                len(self.in_use_connections) + 
                len(self.connecting_connections)
            ),
            'total_connections_created': self.total_connections_created,
            'total_connections_acquired': self.total_connections_acquired,
            'total_connections_released': self.total_connections_released,
            'total_connection_errors': self.total_connection_errors,
            'max_connections': self.config.max_connections,
            'min_connections': self.config.min_connections
        }
    
    async def shutdown(self) -> None:
        """Shutdown the connection pool."""
        logger.info(f"🔄 Shutting down Connection Pool '{self.name}'...")
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._scaling_task:
            self._scaling_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Wait for tasks to complete
        tasks = []
        if self._health_check_task:
            tasks.append(self._health_check_task)
        if self._scaling_task:
            tasks.append(self._scaling_task)
        if self._cleanup_task:
            tasks.append(self._cleanup_task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Close all connections
        async with self._lock:
            for connection in self.available_connections + self.in_use_connections + self.connecting_connections:
                await self._remove_connection(connection)
        
        logger.info(f"✅ Connection Pool '{self.name}' shutdown complete")

# =============================================================================
# 🎯 CONNECTION POOL MANAGER
# =============================================================================

class ConnectionPoolManager:
    """Manages multiple connection pools."""
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self.pool_configs: Dict[str, ConnectionConfig] = {}
        self._manager_lock = asyncio.Lock()
    
    async def create_pool(
        self, 
        name: str, 
        config: ConnectionConfig, 
        connection_factory: Callable[[str, ConnectionConfig], Connection]
    ) -> ConnectionPool:
        """Create a new connection pool."""
        async with self._manager_lock:
            if name in self.pools:
                logger.warning(f"⚠️ Pool '{name}' already exists, replacing")
                await self.remove_pool(name)
            
            pool = ConnectionPool(name, config, connection_factory)
            self.pools[name] = pool
            self.pool_configs[name] = config
            
            logger.info(f"🔄 Created connection pool: {name}")
            return pool
    
    async def remove_pool(self, name: str) -> bool:
        """Remove a connection pool."""
        async with self._manager_lock:
            if name not in self.pools:
                return False
            
            pool = self.pools[name]
            await pool.shutdown()
            
            del self.pools[name]
            del self.pool_configs[name]
            
            logger.info(f"🗑️ Removed connection pool: {name}")
            return True
    
    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name."""
        return self.pools.get(name)
    
    def list_pools(self) -> List[str]:
        """List all pool names."""
        return list(self.pools.keys())
    
    def get_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        return {
            name: pool.get_pool_stats()
            for name, pool in self.pools.items()
        }
    
    async def shutdown_all(self) -> None:
        """Shutdown all connection pools."""
        logger.info("🔄 Shutting down all connection pools...")
        
        async with self._manager_lock:
            for name, pool in self.pools.items():
                try:
                    await pool.shutdown()
                    logger.info(f"✅ Pool '{name}' shutdown complete")
                except Exception as e:
                    logger.error(f"❌ Error shutting down pool '{name}': {e}")
        
        self.pools.clear()
        self.pool_configs.clear()
        logger.info("✅ All connection pools shutdown complete")

# =============================================================================
# 🚀 FACTORY FUNCTIONS
# =============================================================================

def create_connection_pool_manager() -> ConnectionPoolManager:
    """Create a new connection pool manager."""
    return ConnectionPoolManager()

def create_optimized_connection_config(
    connection_type: ConnectionType,
    **kwargs
) -> ConnectionConfig:
    """Create an optimized connection configuration."""
    config = ConnectionConfig(connection_type=connection_type)
    
    # Apply optimizations based on connection type
    if connection_type == ConnectionType.HTTP:
        config.max_connections = 100
        config.connection_timeout = 10.0
        config.pool_timeout = 5.0
    elif connection_type == ConnectionType.DATABASE:
        config.max_connections = 50
        config.connection_timeout = 30.0
        config.pool_timeout = 10.0
    elif connection_type == ConnectionType.REDIS:
        config.max_connections = 30
        config.connection_timeout = 5.0
        config.pool_timeout = 2.0
    
    # Apply custom settings
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

# =============================================================================
# 🌟 EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ConnectionType",
    "ConnectionStatus",
    
    # Configuration
    "ConnectionConfig",
    
    # Interfaces
    "Connection",
    
    # Pool management
    "ConnectionPool",
    "ConnectionPoolManager",
    
    # Factory functions
    "create_connection_pool_manager",
    "create_optimized_connection_config"
]


