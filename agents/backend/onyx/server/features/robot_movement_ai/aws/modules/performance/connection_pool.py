"""
Connection Pool Manager
=======================

Advanced connection pooling for database and external services.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from contextlib import asynccontextmanager
import time

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Connection pool for managing connections."""
    
    def __init__(
        self,
        factory: Callable,
        min_size: int = 2,
        max_size: int = 10,
        max_idle: int = 300,
        max_lifetime: int = 3600
    ):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle = max_idle
        self.max_lifetime = max_lifetime
        self._pool: list = []
        self._in_use: set = set()
        self._lock = asyncio.Lock()
        self._created_at: Dict[Any, float] = {}
        self._last_used: Dict[Any, float] = {}
    
    async def acquire(self):
        """Acquire connection from pool."""
        async with self._lock:
            # Try to get existing connection
            while self._pool:
                conn = self._pool.pop()
                if self._is_valid(conn):
                    self._in_use.add(conn)
                    self._last_used[conn] = time.time()
                    return conn
                else:
                    # Connection invalid, remove it
                    await self._close_connection(conn)
            
            # Create new connection if under max size
            if len(self._in_use) < self.max_size:
                conn = await self._create_connection()
                self._in_use.add(conn)
                self._created_at[conn] = time.time()
                self._last_used[conn] = time.time()
                return conn
            
            # Pool exhausted, wait for connection
            logger.warning("Connection pool exhausted, waiting...")
            await asyncio.sleep(0.1)
            return await self.acquire()
    
    async def release(self, conn):
        """Release connection back to pool."""
        async with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                if self._is_valid(conn):
                    self._pool.append(conn)
                    self._last_used[conn] = time.time()
                else:
                    await self._close_connection(conn)
    
    @asynccontextmanager
    async def connection(self):
        """Context manager for connection."""
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)
    
    def _is_valid(self, conn) -> bool:
        """Check if connection is valid."""
        if conn not in self._created_at:
            return False
        
        # Check lifetime
        age = time.time() - self._created_at[conn]
        if age > self.max_lifetime:
            return False
        
        # Check idle time
        if conn in self._last_used:
            idle = time.time() - self._last_used[conn]
            if idle > self.max_idle:
                return False
        
        return True
    
    async def _create_connection(self):
        """Create new connection."""
        if asyncio.iscoroutinefunction(self.factory):
            return await self.factory()
        return self.factory()
    
    async def _close_connection(self, conn):
        """Close connection."""
        try:
            if hasattr(conn, 'close'):
                if asyncio.iscoroutinefunction(conn.close):
                    await conn.close()
                else:
                    conn.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    async def close_all(self):
        """Close all connections."""
        async with self._lock:
            for conn in list(self._pool) + list(self._in_use):
                await self._close_connection(conn)
            self._pool.clear()
            self._in_use.clear()
            self._created_at.clear()
            self._last_used.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": len(self._pool),
            "in_use": len(self._in_use),
            "total": len(self._pool) + len(self._in_use),
            "max_size": self.max_size,
            "min_size": self.min_size
        }


class ConnectionPoolManager:
    """Manager for multiple connection pools."""
    
    def __init__(self):
        self._pools: Dict[str, ConnectionPool] = {}
    
    def register_pool(
        self,
        name: str,
        factory: Callable,
        min_size: int = 2,
        max_size: int = 10,
        max_idle: int = 300,
        max_lifetime: int = 3600
    ):
        """Register a connection pool."""
        self._pools[name] = ConnectionPool(
            factory=factory,
            min_size=min_size,
            max_size=max_size,
            max_idle=max_idle,
            max_lifetime=max_lifetime
        )
        logger.info(f"Registered connection pool: {name}")
    
    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get connection pool by name."""
        return self._pools.get(name)
    
    async def close_all(self):
        """Close all pools."""
        for pool in self._pools.values():
            await pool.close_all()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        return {name: pool.get_stats() for name, pool in self._pools.items()}















