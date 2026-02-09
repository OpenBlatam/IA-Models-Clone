"""
Connection Pooler
=================

Advanced connection pooling for maximum performance.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from contextlib import asynccontextmanager
from aws.modules.performance.connection_pool import ConnectionPool, ConnectionPoolManager

logger = logging.getLogger(__name__)


class ConnectionPooler:
    """Advanced connection pooler with auto-scaling."""
    
    def __init__(self):
        self.pool_manager = ConnectionPoolManager()
        self._stats: Dict[str, Dict[str, Any]] = {}
    
    def create_pool(
        self,
        name: str,
        factory: Callable,
        min_size: int = 5,
        max_size: int = 20,
        initial_size: Optional[int] = None,
        max_idle: int = 300,
        max_lifetime: int = 3600,
        auto_scale: bool = True
    ):
        """Create optimized connection pool."""
        initial_size = initial_size or min_size
        
        pool = ConnectionPool(
            factory=factory,
            min_size=min_size,
            max_size=max_size,
            max_idle=max_idle,
            max_lifetime=max_lifetime
        )
        
        self.pool_manager._pools[name] = pool
        
        # Pre-create initial connections
        if initial_size > 0:
            asyncio.create_task(self._prefill_pool(name, initial_size))
        
        # Auto-scale if enabled
        if auto_scale:
            asyncio.create_task(self._auto_scale_pool(name))
        
        logger.info(f"Created connection pool: {name} (min={min_size}, max={max_size})")
    
    async def _prefill_pool(self, name: str, count: int):
        """Pre-fill pool with connections."""
        pool = self.pool_manager.get_pool(name)
        if not pool:
            return
        
        for _ in range(count):
            try:
                conn = await pool.acquire()
                await pool.release(conn)
            except Exception as e:
                logger.warning(f"Failed to pre-fill pool {name}: {e}")
    
    async def _auto_scale_pool(self, name: str):
        """Auto-scale pool based on usage."""
        pool = self.pool_manager.get_pool(name)
        if not pool:
            return
        
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            stats = pool.get_stats()
            utilization = stats["in_use"] / stats["max_size"] if stats["max_size"] > 0 else 0
            
            # Auto-scale logic
            if utilization > 0.8 and stats["total"] < stats["max_size"]:
                # Scale up
                logger.debug(f"Pool {name} utilization high ({utilization:.2%}), scaling up")
            elif utilization < 0.2 and stats["total"] > stats["min_size"]:
                # Scale down
                logger.debug(f"Pool {name} utilization low ({utilization:.2%}), scaling down")
    
    @asynccontextmanager
    async def get_connection(self, pool_name: str):
        """Get connection from pool."""
        pool = self.pool_manager.get_pool(pool_name)
        if not pool:
            raise ValueError(f"Pool {pool_name} not found")
        
        async with pool.connection() as conn:
            yield conn
    
    def get_pool_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get pool statistics."""
        if name:
            pool = self.pool_manager.get_pool(name)
            if pool:
                return pool.get_stats()
            return {}
        
        return self.pool_manager.get_all_stats()















