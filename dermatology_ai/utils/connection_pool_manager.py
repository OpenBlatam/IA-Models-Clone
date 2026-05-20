"""
Advanced Connection Pool Manager
Manages pools for multiple connection types
"""

from typing import Dict, Optional, Callable, Any
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class PoolType(str, Enum):
    """Connection pool types"""
    DATABASE = "database"
    HTTP = "http"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"


class ConnectionPoolManager:
    """
    Manages multiple connection pools.
    Provides unified interface for different connection types.
    """
    
    def __init__(self):
        self.pools: Dict[str, Any] = {}
        self.pool_factories: Dict[PoolType, Callable] = {}
    
    def register_pool_factory(
        self,
        pool_type: PoolType,
        factory: Callable
    ):
        """Register factory function for pool type"""
        self.pool_factories[pool_type] = factory
        logger.debug(f"Registered pool factory for {pool_type.value}")
    
    async def get_pool(
        self,
        pool_type: PoolType,
        pool_name: str = "default",
        **config
    ) -> Any:
        """
        Get or create connection pool
        
        Args:
            pool_type: Type of pool
            pool_name: Name of pool instance
            **config: Pool configuration
            
        Returns:
            Connection pool instance
        """
        pool_key = f"{pool_type.value}:{pool_name}"
        
        if pool_key in self.pools:
            return self.pools[pool_key]
        
        # Create new pool
        factory = self.pool_factories.get(pool_type)
        if not factory:
            raise ValueError(f"No factory registered for {pool_type.value}")
        
        pool = await factory(**config)
        self.pools[pool_key] = pool
        
        logger.info(f"Created connection pool: {pool_key}")
        return pool
    
    async def close_pool(self, pool_type: PoolType, pool_name: str = "default"):
        """Close connection pool"""
        pool_key = f"{pool_type.value}:{pool_name}"
        
        if pool_key in self.pools:
            pool = self.pools[pool_key]
            if hasattr(pool, "close"):
                await pool.close()
            del self.pools[pool_key]
            logger.info(f"Closed connection pool: {pool_key}")
    
    async def close_all(self):
        """Close all connection pools"""
        for pool_key, pool in list(self.pools.items()):
            if hasattr(pool, "close"):
                await pool.close()
            del self.pools[pool_key]
        
        logger.info("Closed all connection pools")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        stats = {}
        
        for pool_key, pool in self.pools.items():
            if hasattr(pool, "get_stats"):
                stats[pool_key] = pool.get_stats()
            else:
                stats[pool_key] = {"status": "active"}
        
        return stats


# Global pool manager
_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> ConnectionPoolManager:
    """Get or create global pool manager"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager















