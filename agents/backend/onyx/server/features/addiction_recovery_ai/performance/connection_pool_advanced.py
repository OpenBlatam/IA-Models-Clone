"""
Advanced Connection Pool Manager
Ultra-fast connection pooling with health checks and adaptive sizing
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, Callable
from contextlib import asynccontextmanager
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state"""
    IDLE = "idle"
    IN_USE = "in_use"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass
class Connection:
    """Connection object"""
    id: str
    created_at: float
    last_used: float
    state: ConnectionState
    obj: Any
    health_check_count: int = 0
    error_count: int = 0


class AdvancedConnectionPool:
    """
    Advanced connection pool manager
    
    Features:
    - Health checks
    - Adaptive pool sizing
    - Connection lifecycle management
    - Automatic recovery
    - Load-based scaling
    - Connection statistics
    """
    
    def __init__(
        self,
        pool_name: str,
        factory: Callable,
        min_size: int = 5,
        max_size: int = 50,
        max_idle_time: int = 300,
        health_check_interval: int = 60,
        health_check_func: Optional[Callable] = None
    ):
        self.pool_name = pool_name
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.health_check_func = health_check_func
        
        self._pool: deque = deque()
        self._connections: Dict[str, Connection] = {}
        self._in_use: Dict[str, Connection] = {}
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        
        self._stats = {
            "created": 0,
            "reused": 0,
            "closed": 0,
            "health_checks": 0,
            "errors": 0
        }
        
        logger.info(f"✅ Advanced connection pool '{pool_name}' initialized (min: {min_size}, max: {max_size})")
    
    async def initialize(self):
        """Initialize pool with minimum connections"""
        async with self._lock:
            for _ in range(self.min_size):
                await self._create_connection()
            
            # Start health check task
            if self.health_check_func:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _create_connection(self) -> Connection:
        """Create new connection"""
        conn_id = f"{self.pool_name}_{int(time.time() * 1000)}"
        
        try:
            if asyncio.iscoroutinefunction(self.factory):
                conn_obj = await self.factory()
            else:
                conn_obj = self.factory()
            
            connection = Connection(
                id=conn_id,
                created_at=time.time(),
                last_used=time.time(),
                state=ConnectionState.IDLE,
                obj=conn_obj
            )
            
            self._connections[conn_id] = connection
            self._pool.append(connection)
            self._stats["created"] += 1
            
            logger.debug(f"Created connection: {conn_id}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire connection from pool
        
        Args:
            timeout: Maximum time to wait for connection
            
        Yields:
            Connection object
        """
        start_time = time.time()
        
        while True:
            async with self._lock:
                # Try to get from pool
                if self._pool:
                    connection = self._pool.popleft()
                    connection.state = ConnectionState.IN_USE
                    connection.last_used = time.time()
                    self._in_use[connection.id] = connection
                    self._stats["reused"] += 1
                    break
                
                # Try to create new connection
                if len(self._connections) < self.max_size:
                    try:
                        connection = await self._create_connection()
                        connection.state = ConnectionState.IN_USE
                        connection.last_used = time.time()
                        self._in_use[connection.id] = connection
                        break
                    except Exception as e:
                        logger.error(f"Failed to create connection: {e}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Failed to acquire connection within timeout")
            
            # Wait a bit before retrying
            await asyncio.sleep(0.01)
        
        try:
            yield connection.obj
        finally:
            # Return connection to pool
            async with self._lock:
                if connection.id in self._in_use:
                    del self._in_use[connection.id]
                
                # Check if connection is still healthy
                if connection.state == ConnectionState.UNHEALTHY:
                    await self._close_connection(connection)
                else:
                    connection.state = ConnectionState.IDLE
                    connection.last_used = time.time()
                    self._pool.append(connection)
    
    async def _health_check_loop(self):
        """Periodic health check loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all connections"""
        async with self._lock:
            connections_to_check = list(self._pool) + list(self._in_use.values())
        
        for connection in connections_to_check:
            try:
                if self.health_check_func:
                    if asyncio.iscoroutinefunction(self.health_check_func):
                        is_healthy = await self.health_check_func(connection.obj)
                    else:
                        is_healthy = self.health_check_func(connection.obj)
                    
                    if not is_healthy:
                        connection.state = ConnectionState.UNHEALTHY
                        connection.error_count += 1
                        self._stats["errors"] += 1
                    else:
                        connection.health_check_count += 1
                        if connection.state == ConnectionState.UNHEALTHY:
                            connection.state = ConnectionState.IDLE
                
                self._stats["health_checks"] += 1
                
            except Exception as e:
                logger.error(f"Health check failed for {connection.id}: {e}")
                connection.state = ConnectionState.UNHEALTHY
                connection.error_count += 1
    
    async def _close_connection(self, connection: Connection):
        """Close and remove connection"""
        try:
            if hasattr(connection.obj, 'close'):
                if asyncio.iscoroutinefunction(connection.obj.close):
                    await connection.obj.close()
                else:
                    connection.obj.close()
        except Exception as e:
            logger.error(f"Error closing connection {connection.id}: {e}")
        
        connection.state = ConnectionState.CLOSED
        if connection.id in self._connections:
            del self._connections[connection.id]
        if connection.id in self._in_use:
            del self._in_use[connection.id]
        
        self._stats["closed"] += 1
    
    async def cleanup_idle_connections(self):
        """Remove idle connections that exceed max_idle_time"""
        current_time = time.time()
        async with self._lock:
            connections_to_remove = []
            
            for connection in list(self._pool):
                idle_time = current_time - connection.last_used
                if idle_time > self.max_idle_time and len(self._connections) > self.min_size:
                    connections_to_remove.append(connection)
            
            for connection in connections_to_remove:
                self._pool.remove(connection)
                await self._close_connection(connection)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            **self._stats,
            "pool_size": len(self._pool),
            "in_use": len(self._in_use),
            "total": len(self._connections),
            "available": len(self._pool)
        }
    
    async def shutdown(self):
        """Shutdown pool and close all connections"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            all_connections = list(self._connections.values())
            for connection in all_connections:
                await self._close_connection(connection)


# Global pools registry
_pools: Dict[str, AdvancedConnectionPool] = {}


def create_connection_pool(
    pool_name: str,
    factory: Callable,
    **kwargs
) -> AdvancedConnectionPool:
    """Create and register connection pool"""
    pool = AdvancedConnectionPool(pool_name, factory, **kwargs)
    _pools[pool_name] = pool
    return pool


def get_connection_pool(pool_name: str) -> Optional[AdvancedConnectionPool]:
    """Get connection pool by name"""
    return _pools.get(pool_name)















