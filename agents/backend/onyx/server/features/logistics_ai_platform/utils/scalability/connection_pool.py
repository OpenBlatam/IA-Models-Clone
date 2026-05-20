"""
Connection pooling utilities

This module provides connection pooling for better resource management and scalability.
"""

import asyncio
from typing import Optional, Generic, TypeVar, Callable, Awaitable
from collections import deque
from contextlib import asynccontextmanager

from utils.logger import logger

T = TypeVar('T')


class ConnectionPool(Generic[T]):
    """
    Generic connection pool for managing reusable connections
    
    Provides connection pooling to reduce overhead and improve scalability.
    """
    
    def __init__(
        self,
        factory: Callable[[], Awaitable[T]],
        max_size: int = 10,
        min_size: int = 2,
        timeout: float = 5.0
    ):
        """
        Initialize connection pool
        
        Args:
            factory: Async function to create new connections
            max_size: Maximum number of connections in pool
            min_size: Minimum number of connections to maintain
            timeout: Timeout for acquiring connection
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.timeout = timeout
        self.pool: deque = deque()
        self.active: int = 0
        self._lock = asyncio.Lock()
        self._closed = False
    
    async def acquire(self) -> T:
        """
        Acquire a connection from the pool
        
        Returns:
            Connection from pool or newly created connection
            
        Raises:
            TimeoutError: If unable to acquire connection within timeout
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        async with self._lock:
            if self.pool:
                self.active += 1
                return self.pool.popleft()
            
            if self.active < self.max_size:
                self.active += 1
                try:
                    connection = await self.factory()
                    return connection
                except Exception as e:
                    self.active -= 1
                    logger.error(f"Failed to create connection: {e}")
                    raise
        
        try:
            await asyncio.wait_for(self._wait_for_connection(), timeout=self.timeout)
            async with self._lock:
                if self.pool:
                    self.active += 1
                    return self.pool.popleft()
        except asyncio.TimeoutError:
            raise TimeoutError(f"Unable to acquire connection within {self.timeout}s")
    
    async def release(self, connection: T) -> None:
        """
        Release a connection back to the pool
        
        Args:
            connection: Connection to release
        """
        if self._closed:
            return
        
        async with self._lock:
            self.active -= 1
            if len(self.pool) < self.max_size:
                self.pool.append(connection)
    
    @asynccontextmanager
    async def connection(self):
        """
        Context manager for acquiring and releasing connections
        
        Usage:
            async with pool.connection() as conn:
                # use conn
        """
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)
    
    async def _wait_for_connection(self) -> None:
        """Wait for a connection to become available"""
        while True:
            async with self._lock:
                if self.pool or self.active < self.max_size:
                    return
            await asyncio.sleep(0.1)
    
    async def close(self) -> None:
        """Close the connection pool and cleanup"""
        self._closed = True
        async with self._lock:
            while self.pool:
                conn = self.pool.popleft()
                if hasattr(conn, 'close'):
                    try:
                        if asyncio.iscoroutinefunction(conn.close):
                            await conn.close()
                        else:
                            conn.close()
                    except Exception as e:
                        logger.warning(f"Error closing connection: {e}")


_pools: dict = {}


def get_connection_pool(
    name: str,
    factory: Callable[[], Awaitable[T]],
    max_size: int = 10,
    min_size: int = 2
) -> ConnectionPool[T]:
    """
    Get or create a named connection pool
    
    Args:
        name: Pool name
        factory: Connection factory function
        max_size: Maximum pool size
        min_size: Minimum pool size
        
    Returns:
        Connection pool instance
    """
    if name not in _pools:
        _pools[name] = ConnectionPool(factory, max_size, min_size)
    return _pools[name]







