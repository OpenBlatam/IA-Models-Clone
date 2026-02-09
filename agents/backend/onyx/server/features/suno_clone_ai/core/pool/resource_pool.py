"""
Resource Pool

Utilities for resource pooling.
"""

import logging
import queue
import threading
from typing import Any, Callable, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ResourcePool:
    """Pool for managing resources."""
    
    def __init__(
        self,
        factory: Callable,
        max_size: int = 10,
        min_size: int = 2
    ):
        """
        Initialize resource pool.
        
        Args:
            factory: Factory function to create resources
            max_size: Maximum pool size
            min_size: Minimum pool size
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.pool = queue.Queue(maxsize=max_size)
        self.created = 0
        self.lock = threading.Lock()
        
        # Initialize pool with minimum resources
        for _ in range(min_size):
            self._create_resource()
    
    def _create_resource(self) -> Any:
        """Create new resource."""
        with self.lock:
            if self.created < self.max_size:
                resource = self.factory()
                self.created += 1
                return resource
        return None
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """
        Get resource from pool.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Resource instance
        """
        try:
            resource = self.pool.get(timeout=timeout)
            return resource
        except queue.Empty:
            # Try to create new resource
            resource = self._create_resource()
            if resource:
                return resource
            raise TimeoutError("No resources available in pool")
    
    def put(self, resource: Any) -> None:
        """
        Return resource to pool.
        
        Args:
            resource: Resource instance
        """
        try:
            self.pool.put_nowait(resource)
        except queue.Full:
            # Pool is full, discard resource
            logger.warning("Pool is full, discarding resource")
    
    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """
        Acquire resource context manager.
        
        Args:
            timeout: Timeout in seconds
            
        Yields:
            Resource instance
        """
        resource = self.get(timeout)
        try:
            yield resource
        finally:
            self.put(resource)
    
    def size(self) -> int:
        """Get current pool size."""
        return self.pool.qsize()
    
    def clear(self) -> None:
        """Clear pool."""
        while not self.pool.empty():
            try:
                self.pool.get_nowait()
            except queue.Empty:
                break


def create_pool(
    factory: Callable,
    **kwargs
) -> ResourcePool:
    """Create resource pool."""
    return ResourcePool(factory, **kwargs)


def get_from_pool(
    pool: ResourcePool,
    timeout: Optional[float] = None
) -> Any:
    """Get resource from pool."""
    return pool.get(timeout)


def return_to_pool(
    pool: ResourcePool,
    resource: Any
) -> None:
    """Return resource to pool."""
    pool.put(resource)



