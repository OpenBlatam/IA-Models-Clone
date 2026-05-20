"""
Pool utilities
Resource pooling patterns
"""

from typing import TypeVar, Callable, Optional, List, Any
from asyncio import Semaphore, Queue
import asyncio

T = TypeVar('T')


class ResourcePool:
    """
    Resource pool for managing limited resources
    """
    
    def __init__(self, resources: List[T], max_size: Optional[int] = None):
        self.resources = resources.copy()
        self.max_size = max_size or len(resources)
        self.available = Queue(maxsize=self.max_size)
        self.in_use: set[T] = set()
        
        # Initialize available queue
        for resource in self.resources[:self.max_size]:
            self.available.put_nowait(resource)
    
    async def acquire(self) -> T:
        """Acquire resource from pool"""
        resource = await self.available.get()
        self.in_use.add(resource)
        return resource
    
    async def release(self, resource: T) -> None:
        """Release resource back to pool"""
        if resource in self.in_use:
            self.in_use.remove(resource)
            await self.available.put(resource)
    
    def available_count(self) -> int:
        """Get count of available resources"""
        return self.available.qsize()
    
    def in_use_count(self) -> int:
        """Get count of resources in use"""
        return len(self.in_use)
    
    async def __aenter__(self):
        return await self.acquire()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Resource should be released by caller
        pass


def create_resource_pool(
    resources: List[T],
    max_size: Optional[int] = None
) -> ResourcePool:
    """Create new resource pool"""
    return ResourcePool(resources, max_size)


async def with_pool(
    pool: ResourcePool,
    func: Callable[[T], Any]
) -> Any:
    """
    Execute function with resource from pool
    
    Args:
        pool: Resource pool
        func: Function to execute with resource
    
    Returns:
        Function result
    """
    resource = await pool.acquire()
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(resource)
        return func(resource)
    finally:
        await pool.release(resource)

