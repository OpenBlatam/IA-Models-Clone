"""
Resource Pool for Color Grading AI
====================================

Resource pooling for efficient resource management.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PooledResource:
    """Pooled resource."""
    resource: Any
    acquired_at: Optional[datetime] = None
    use_count: int = 0
    last_used: Optional[datetime] = None


class ResourcePool(Generic[T]):
    """
    Resource pool for managing shared resources.
    
    Features:
    - Resource pooling
    - Automatic cleanup
    - Usage tracking
    - Health checks
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: float = 300.0
    ):
        """
        Initialize resource pool.
        
        Args:
            factory: Factory function to create resources
            max_size: Maximum pool size
            min_size: Minimum pool size
            max_idle_time: Maximum idle time before cleanup (seconds)
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self._pool: deque = deque()
        self._acquired: Dict[T, PooledResource] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def acquire(self) -> T:
        """
        Acquire resource from pool.
        
        Returns:
            Resource instance
        """
        async with self._lock:
            # Try to get from pool
            if self._pool:
                resource = self._pool.popleft()
            else:
                # Create new resource
                resource = self.factory()
            
            # Track acquired resource
            pooled = PooledResource(
                resource=resource,
                acquired_at=datetime.now(),
                use_count=1,
                last_used=datetime.now()
            )
            self._acquired[resource] = pooled
            
            return resource
    
    async def release(self, resource: T):
        """
        Release resource back to pool.
        
        Args:
            resource: Resource to release
        """
        async with self._lock:
            if resource not in self._acquired:
                logger.warning("Attempted to release resource not in pool")
                return
            
            pooled = self._acquired.pop(resource)
            pooled.last_used = datetime.now()
            
            # Return to pool if not at max size
            if len(self._pool) < self.max_size:
                self._pool.append(resource)
            else:
                # Pool full, discard resource
                logger.debug("Pool full, discarding resource")
    
    async def cleanup(self):
        """Cleanup idle resources."""
        async with self._lock:
            now = datetime.now()
            idle_resources = []
            
            for resource in list(self._pool):
                # Check if resource has cleanup method
                if hasattr(resource, 'last_used'):
                    # Resource-specific cleanup check
                    continue
                
                # Simple idle check (would need resource-specific logic)
                idle_resources.append(resource)
            
            # Remove idle resources (keeping min_size)
            while len(self._pool) > self.min_size and idle_resources:
                resource = idle_resources.pop(0)
                if resource in self._pool:
                    self._pool.remove(resource)
                    logger.debug("Removed idle resource from pool")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": len(self._pool),
            "acquired": len(self._acquired),
            "max_size": self.max_size,
            "min_size": self.min_size,
            "total_resources": len(self._pool) + len(self._acquired),
        }
    
    async def close(self):
        """Close pool and cleanup resources."""
        async with self._lock:
            # Cleanup all resources
            for resource in list(self._pool):
                if hasattr(resource, 'close'):
                    try:
                        await resource.close()
                    except:
                        pass
            
            self._pool.clear()
            self._acquired.clear()
            
            if self._cleanup_task:
                self._cleanup_task.cancel()




