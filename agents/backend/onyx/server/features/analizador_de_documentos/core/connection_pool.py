"""
Connection Pool for Document Analyzer
======================================

Advanced connection pooling for efficient resource management.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, TypeVar
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class Connection:
    """Connection wrapper"""
    connection_id: str
    connection: Any
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    is_available: bool = True

class ConnectionPool:
    """Advanced connection pool"""
    
    def __init__(
        self,
        pool_name: str,
        factory: Callable,
        min_size: int = 2,
        max_size: int = 10,
        timeout: float = 30.0
    ):
        self.pool_name = pool_name
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        
        self.pool: deque = deque()
        self.active_connections: Dict[str, Connection] = {}
        self.lock = asyncio.Lock()
        
        logger.info(f"ConnectionPool '{pool_name}' initialized. Min: {min_size}, Max: {max_size}")
    
    async def acquire(self) -> Any:
        """Acquire a connection from pool"""
        async with self.lock:
            # Try to get available connection
            while self.pool:
                conn_wrapper = self.pool.popleft()
                if conn_wrapper.is_available:
                    conn_wrapper.is_available = False
                    conn_wrapper.last_used = datetime.now()
                    conn_wrapper.use_count += 1
                    self.active_connections[conn_wrapper.connection_id] = conn_wrapper
                    return conn_wrapper.connection
                else:
                    # Connection was already taken, continue
                    continue
            
            # Create new connection if under max
            if len(self.active_connections) < self.max_size:
                connection_id = f"{self.pool_name}_{int(time.time())}"
                
                if asyncio.iscoroutinefunction(self.factory):
                    connection = await self.factory()
                else:
                    connection = self.factory()
                
                conn_wrapper = Connection(
                    connection_id=connection_id,
                    connection=connection,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    is_available=False
                )
                
                self.active_connections[connection_id] = conn_wrapper
                logger.debug(f"Created new connection: {connection_id}")
                return connection
            
            # Wait for available connection
            raise RuntimeError("Connection pool exhausted")
    
    async def release(self, connection: Any):
        """Release connection back to pool"""
        async with self.lock:
            # Find connection
            for conn_id, conn_wrapper in list(self.active_connections.items()):
                if conn_wrapper.connection == connection:
                    conn_wrapper.is_available = True
                    conn_wrapper.last_used = datetime.now()
                    del self.active_connections[conn_id]
                    self.pool.append(conn_wrapper)
                    return
            
            logger.warning("Connection not found in pool")
    
    async def close_all(self):
        """Close all connections"""
        async with self.lock:
            for conn_wrapper in list(self.pool) + list(self.active_connections.values()):
                try:
                    if hasattr(conn_wrapper.connection, 'close'):
                        if asyncio.iscoroutinefunction(conn_wrapper.connection.close):
                            await conn_wrapper.connection.close()
                        else:
                            conn_wrapper.connection.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self.pool.clear()
            self.active_connections.clear()
            logger.info(f"Closed all connections in pool '{self.pool_name}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "pool_name": self.pool_name,
            "pool_size": len(self.pool),
            "active_connections": len(self.active_connections),
            "total_connections": len(self.pool) + len(self.active_connections),
            "min_size": self.min_size,
            "max_size": self.max_size
        }
















