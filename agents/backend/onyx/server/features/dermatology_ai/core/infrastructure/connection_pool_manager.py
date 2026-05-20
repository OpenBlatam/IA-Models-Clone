"""
Connection Pool Manager
Manages database connection pools for optimal performance
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Connection pool configuration"""
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # Recycle connections after 1 hour
    pool_pre_ping: bool = True  # Verify connections before use


class ConnectionPoolManager:
    """Manages connection pools for different database adapters"""
    
    def __init__(self):
        self.pools: Dict[str, Any] = {}
        self.configs: Dict[str, PoolConfig] = {}
        self._lock = asyncio.Lock()
    
    def register_pool(
        self,
        pool_name: str,
        pool: Any,
        config: Optional[PoolConfig] = None
    ):
        """
        Register a connection pool
        
        Args:
            pool_name: Name identifier for the pool
            pool: Pool instance
            config: Pool configuration
        """
        self.pools[pool_name] = pool
        self.configs[pool_name] = config or PoolConfig()
        logger.info(f"Registered connection pool: {pool_name}")
    
    async def get_connection(self, pool_name: str):
        """
        Get connection from pool
        
        Args:
            pool_name: Pool name
            
        Returns:
            Connection from pool
        """
        if pool_name not in self.pools:
            raise ValueError(f"Pool {pool_name} not found")
        
        pool = self.pools[pool_name]
        
        # Try to get connection with timeout
        try:
            if hasattr(pool, "acquire"):
                return await asyncio.wait_for(
                    pool.acquire(),
                    timeout=self.configs[pool_name].pool_timeout
                )
            elif hasattr(pool, "get_connection"):
                return await asyncio.wait_for(
                    pool.get_connection(),
                    timeout=self.configs[pool_name].pool_timeout
                )
            else:
                # Fallback: assume pool is connection itself
                return pool
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting connection from pool {pool_name}")
            raise
    
    async def release_connection(self, pool_name: str, connection: Any):
        """
        Release connection back to pool
        
        Args:
            pool_name: Pool name
            connection: Connection to release
        """
        if pool_name not in self.pools:
            return
        
        pool = self.pools[pool_name]
        
        try:
            if hasattr(pool, "release"):
                await pool.release(connection)
            elif hasattr(pool, "return_connection"):
                await pool.return_connection(connection)
            elif hasattr(connection, "close"):
                await connection.close()
        except Exception as e:
            logger.warning(f"Error releasing connection: {e}")
    
    @asynccontextmanager
    async def get_connection_context(self, pool_name: str):
        """
        Context manager for getting and releasing connections
        
        Args:
            pool_name: Pool name
            
        Yields:
            Connection from pool
        """
        connection = None
        try:
            connection = await self.get_connection(pool_name)
            yield connection
        finally:
            if connection:
                await self.release_connection(pool_name, connection)
    
    async def health_check(self, pool_name: str) -> bool:
        """
        Check pool health
        
        Args:
            pool_name: Pool name
            
        Returns:
            True if pool is healthy
        """
        if pool_name not in self.pools:
            return False
        
        try:
            async with self.get_connection_context(pool_name) as conn:
                # Simple health check - try to use connection
                if hasattr(conn, "ping"):
                    await conn.ping()
                return True
        except Exception as e:
            logger.warning(f"Pool {pool_name} health check failed: {e}")
            return False
    
    def get_pool_stats(self, pool_name: str) -> Dict[str, Any]:
        """Get pool statistics"""
        if pool_name not in self.pools:
            return {}
        
        pool = self.pools[pool_name]
        config = self.configs[pool_name]
        
        stats = {
            "min_size": config.min_size,
            "max_size": config.max_size,
            "max_overflow": config.max_overflow,
        }
        
        # Try to get actual pool stats
        if hasattr(pool, "size"):
            stats["current_size"] = pool.size()
        if hasattr(pool, "checked_in"):
            stats["checked_in"] = pool.checked_in()
        if hasattr(pool, "checked_out"):
            stats["checked_out"] = pool.checked_out()
        
        return stats


# Global pool manager
_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> ConnectionPoolManager:
    """Get global connection pool manager"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager















