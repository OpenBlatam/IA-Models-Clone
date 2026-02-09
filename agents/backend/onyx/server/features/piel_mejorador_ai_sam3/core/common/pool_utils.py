"""
Pool Utilities for Piel Mejorador AI SAM3
========================================

Unified resource pool pattern utilities.
"""

import asyncio
import logging
from typing import TypeVar, Generic, Callable, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PooledResource:
    """Pooled resource."""
    resource: Any
    acquired_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


class ResourcePool(Generic[T]):
    """Generic resource pool."""
    
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 0,
        name: str = "pool"
    ):
        """
        Initialize resource pool.
        
        Args:
            factory: Factory function to create resources
            max_size: Maximum pool size
            min_size: Minimum pool size
            name: Pool name
        """
        self._factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.name = name
        self._pool: List[PooledResource] = []
        self._lock = asyncio.Lock()
        self._acquired_count = 0
    
    async def acquire(self) -> T:
        """
        Acquire resource from pool.
        
        Returns:
            Resource
        """
        async with self._lock:
            # Try to get from pool
            if self._pool:
                pooled = self._pool.pop()
                pooled.acquired_at = datetime.now()
                self._acquired_count += 1
                logger.debug(f"Acquired resource from {self.name}")
                return pooled.resource
            
            # Create new if under max size
            if self._acquired_count + len(self._pool) < self.max_size:
                resource = self._factory()
                self._acquired_count += 1
                logger.debug(f"Created new resource for {self.name}")
                return resource
            
            # Wait for resource to be available
            # Simple implementation: create anyway if max_size allows
            if self._acquired_count < self.max_size:
                resource = self._factory()
                self._acquired_count += 1
                return resource
            
            raise RuntimeError(f"Pool {self.name} exhausted (max_size={self.max_size})")
    
    async def release(self, resource: T):
        """
        Release resource back to pool.
        
        Args:
            resource: Resource to release
        """
        async with self._lock:
            pooled = PooledResource(resource=resource)
            self._pool.append(pooled)
            self._acquired_count -= 1
            logger.debug(f"Released resource to {self.name}")
    
    @asynccontextmanager
    async def get(self):
        """
        Get resource as context manager.
        
        Yields:
            Resource
        """
        resource = await self.acquire()
        try:
            yield resource
        finally:
            await self.release(resource)
    
    async def size(self) -> int:
        """Get current pool size."""
        async with self._lock:
            return len(self._pool)
    
    async def acquired_count(self) -> int:
        """Get number of acquired resources."""
        async with self._lock:
            return self._acquired_count
    
    async def available_count(self) -> int:
        """Get number of available resources."""
        async with self._lock:
            return len(self._pool)
    
    async def clear(self):
        """Clear pool."""
        async with self._lock:
            self._pool.clear()
            logger.debug(f"Cleared pool {self.name}")


class PoolUtils:
    """Unified pool utilities."""
    
    @staticmethod
    def create_pool(
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 0,
        name: str = "pool"
    ) -> ResourcePool:
        """
        Create resource pool.
        
        Args:
            factory: Factory function
            max_size: Maximum pool size
            min_size: Minimum pool size
            name: Pool name
            
        Returns:
            ResourcePool
        """
        return ResourcePool(factory, max_size, min_size, name)
    
    @staticmethod
    async def initialize_pool(
        pool: ResourcePool,
        initial_size: Optional[int] = None
    ):
        """
        Initialize pool with resources.
        
        Args:
            pool: Pool to initialize
            initial_size: Number of resources to pre-create (defaults to min_size)
        """
        if initial_size is None:
            initial_size = pool.min_size
        
        for _ in range(initial_size):
            resource = await pool.acquire()
            await pool.release(resource)


# Convenience functions
def create_pool(factory: Callable[[], T], **kwargs) -> ResourcePool:
    """Create pool."""
    return PoolUtils.create_pool(factory, **kwargs)




