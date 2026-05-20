"""
Connection Pool - High-performance connection pooling
"""

import asyncio
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """Connection wrapper"""
    connection_id: str
    connection: Any
    created_at: float
    last_used: float
    in_use: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_used is None:
            self.last_used = time.time()


class ConnectionPool:
    """
    High-performance connection pool:
    - Async connection management
    - Auto-scaling
    - Connection reuse
    - Health checks
    """
    
    def __init__(
        self,
        factory: Callable,
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time: float = 300.0,
        health_check_interval: float = 60.0
    ):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        
        self.pool: deque = deque()
        self.in_use: Dict[str, Connection] = {}
        self.lock = asyncio.Lock()
        
        self.total_created = 0
        self.total_reused = 0
        self.total_closed = 0
    
    async def acquire(self) -> Connection:
        """Acquire a connection from pool"""
        async with self.lock:
            # Try to get from pool
            while self.pool:
                conn = self.pool.popleft()
                
                # Check if connection is still valid
                if time.time() - conn.last_used > self.max_idle_time:
                    await self._close_connection(conn)
                    continue
                
                # Check health
                if await self._health_check(conn):
                    conn.in_use = True
                    conn.last_used = time.time()
                    self.in_use[conn.connection_id] = conn
                    self.total_reused += 1
                    return conn
                else:
                    await self._close_connection(conn)
            
            # Create new connection if under max
            if len(self.in_use) < self.max_size:
                conn = await self._create_connection()
                conn.in_use = True
                self.in_use[conn.connection_id] = conn
                self.total_created += 1
                return conn
            
            # Wait for available connection
            while len(self.in_use) >= self.max_size:
                await asyncio.sleep(0.01)
                # Try again
                return await self.acquire()
    
    async def release(self, conn: Connection):
        """Release connection back to pool"""
        async with self.lock:
            if conn.connection_id in self.in_use:
                del self.in_use[conn.connection_id]
            
            conn.in_use = False
            conn.last_used = time.time()
            
            # Add back to pool if not too old
            if time.time() - conn.created_at < self.max_idle_time * 2:
                self.pool.append(conn)
            else:
                await self._close_connection(conn)
    
    async def _create_connection(self) -> Connection:
        """Create a new connection"""
        connection_id = f"conn_{int(time.time() * 1000000)}"
        connection = await self.factory()
        
        return Connection(
            connection_id=connection_id,
            connection=connection,
            created_at=time.time(),
            last_used=time.time()
        )
    
    async def _close_connection(self, conn: Connection):
        """Close a connection"""
        try:
            if hasattr(conn.connection, "close"):
                if asyncio.iscoroutinefunction(conn.connection.close):
                    await conn.connection.close()
                else:
                    conn.connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {str(e)}")
        
        self.total_closed += 1
    
    async def _health_check(self, conn: Connection) -> bool:
        """Check connection health"""
        try:
            if hasattr(conn.connection, "ping"):
                if asyncio.iscoroutinefunction(conn.connection.ping):
                    await conn.connection.ping()
                else:
                    conn.connection.ping()
            return True
        except Exception:
            return False
    
    async def initialize(self):
        """Initialize pool with minimum connections"""
        async with self.lock:
            for _ in range(self.min_size):
                conn = await self._create_connection()
                self.pool.append(conn)
                self.total_created += 1
    
    async def cleanup(self):
        """Cleanup idle connections"""
        async with self.lock:
            now = time.time()
            to_remove = []
            
            for conn in self.pool:
                if now - conn.last_used > self.max_idle_time:
                    to_remove.append(conn)
            
            for conn in to_remove:
                self.pool.remove(conn)
                await self._close_connection(conn)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "pool_size": len(self.pool),
            "in_use": len(self.in_use),
            "total_created": self.total_created,
            "total_reused": self.total_reused,
            "total_closed": self.total_closed,
            "reuse_rate": (
                self.total_reused / (self.total_created + self.total_reused)
                if (self.total_created + self.total_reused) > 0
                else 0.0
            )
        }


