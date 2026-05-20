"""
Connection Pool Manager
=======================

Advanced connection pooling for databases, Redis, and external services.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
import weakref

logger = logging.getLogger(__name__)

class ConnectionPool:
    """Generic async connection pool."""
    
    def __init__(
        self,
        factory: Callable,
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time: float = 300.0,
        max_lifetime: float = 3600.0
    ):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.max_lifetime = max_lifetime
        
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.active_connections = 0
        self.total_created = 0
        self.total_closed = 0
        self.connection_times: Dict[Any, datetime] = {}
        self.last_used: Dict[Any, datetime] = {}
        
        self.cleanup_task = None
        self.is_running = False
    
    async def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire connection from pool."""
        connection = None
        
        try:
            # Try to get from pool
            connection = await asyncio.wait_for(
                self.pool.get(),
                timeout=0.1
            )
            
            # Check if connection is still valid
            if await self._is_valid(connection):
                self.active_connections += 1
                self.last_used[connection] = datetime.now()
                return connection
            else:
                # Connection invalid, create new one
                await self._close_connection(connection)
                connection = None
        
        except asyncio.TimeoutError:
            pass
        
        # Create new connection if needed
        if connection is None and self.active_connections < self.max_size:
            connection = await self._create_connection()
            self.active_connections += 1
            self.last_used[connection] = datetime.now()
            return connection
        
        # Wait for available connection
        if timeout:
            connection = await asyncio.wait_for(
                self.pool.get(),
                timeout=timeout
            )
            if await self._is_valid(connection):
                self.active_connections += 1
                self.last_used[connection] = datetime.now()
                return connection
        
        raise TimeoutError("Could not acquire connection from pool")
    
    async def release(self, connection: Any):
        """Release connection back to pool."""
        if connection is None:
            return
        
        self.active_connections -= 1
        self.last_used[connection] = datetime.now()
        
        # Check if should keep in pool
        if self.pool.qsize() < self.max_size:
            await self.pool.put(connection)
        else:
            await self._close_connection(connection)
    
    async def _create_connection(self) -> Any:
        """Create new connection."""
        try:
            if asyncio.iscoroutinefunction(self.factory):
                connection = await self.factory()
            else:
                connection = self.factory()
            
            self.total_created += 1
            self.connection_times[connection] = datetime.now()
            self.last_used[connection] = datetime.now()
            
            logger.debug(f"Created new connection (total: {self.total_created})")
            return connection
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise
    
    async def _close_connection(self, connection: Any):
        """Close connection."""
        try:
            if hasattr(connection, 'close'):
                if asyncio.iscoroutinefunction(connection.close):
                    await connection.close()
                else:
                    connection.close()
            
            if connection in self.connection_times:
                del self.connection_times[connection]
            if connection in self.last_used:
                del self.last_used[connection]
            
            self.total_closed += 1
            logger.debug(f"Closed connection (total: {self.total_closed})")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    async def _is_valid(self, connection: Any) -> bool:
        """Check if connection is valid."""
        try:
            # Check lifetime
            if connection in self.connection_times:
                age = (datetime.now() - self.connection_times[connection]).total_seconds()
                if age > self.max_lifetime:
                    return False
            
            # Check if connection has is_valid method
            if hasattr(connection, 'is_valid'):
                if asyncio.iscoroutinefunction(connection.is_valid):
                    return await connection.is_valid()
                else:
                    return connection.is_valid()
            
            # Default: assume valid
            return True
        except:
            return False
    
    async def _ensure_min_pool_size(self):
        """Ensure minimum pool size."""
        while self.pool.qsize() < self.min_size and self.active_connections < self.max_size:
            try:
                connection = await self._create_connection()
                await self.pool.put(connection)
            except Exception as e:
                logger.error(f"Failed to create connection for pool: {e}")
                break
    
    async def _cleanup_idle_connections(self):
        """Cleanup idle connections."""
        now = datetime.now()
        connections_to_remove = []
        
        # Check connections in pool
        temp_connections = []
        while not self.pool.empty():
            conn = await self.pool.get()
            if conn in self.last_used:
                idle_time = (now - self.last_used[conn]).total_seconds()
                if idle_time > self.max_idle_time:
                    connections_to_remove.append(conn)
                else:
                    temp_connections.append(conn)
            else:
                temp_connections.append(conn)
        
        # Put back valid connections
        for conn in temp_connections:
            await self.pool.put(conn)
        
        # Close idle connections
        for conn in connections_to_remove:
            await self._close_connection(conn)
            logger.debug(f"Removed idle connection")
    
    async def start_maintenance(self, interval: float = 60.0):
        """Start maintenance tasks."""
        if self.is_running:
            return
        
        self.is_running = True
        
        async def maintenance_loop():
            while self.is_running:
                try:
                    await self._ensure_min_pool_size()
                    await self._cleanup_idle_connections()
                    await asyncio.sleep(interval)
                except Exception as e:
                    logger.error(f"Maintenance error: {e}")
                    await asyncio.sleep(interval)
        
        self.cleanup_task = asyncio.create_task(maintenance_loop())
        logger.info("Connection pool maintenance started")
    
    async def stop_maintenance(self):
        """Stop maintenance tasks."""
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        while not self.pool.empty():
            conn = await self.pool.get()
            await self._close_connection(conn)
        
        logger.info("Connection pool maintenance stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool.qsize(),
            "active_connections": self.active_connections,
            "total_created": self.total_created,
            "total_closed": self.total_closed,
            "max_size": self.max_size,
            "min_size": self.min_size,
            "utilization_percent": round((self.active_connections / self.max_size * 100) if self.max_size > 0 else 0, 2)
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return await self.acquire()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Note: This would need connection to be tracked
        pass
































