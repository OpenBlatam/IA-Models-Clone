"""
Resource Pool
=============

System for managing resource pools (connections, workers, etc.).
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, TypeVar, Generic, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PoolState(Enum):
    """Pool state."""
    IDLE = "idle"
    BUSY = "busy"
    FULL = "full"
    EMPTY = "empty"


@dataclass
class PoolConfig:
    """Pool configuration."""
    min_size: int = 1
    max_size: int = 10
    timeout: float = 30.0  # Timeout for acquiring resource
    idle_timeout: float = 300.0  # Timeout before closing idle resources


class ResourcePool(Generic[T]):
    """
    Generic resource pool for managing reusable resources.
    
    Features:
    - Configurable min/max size
    - Timeout support
    - Context manager support
    - Cleanup functions
    - Statistics tracking
    """
    
    def __init__(
        self,
        name: str,
        factory: Callable[[], Awaitable[T]],
        cleanup: Optional[Callable[[T], Awaitable[None]]] = None,
        config: Optional[PoolConfig] = None
    ):
        """
        Initialize resource pool.
        
        Args:
            name: Pool name
            factory: Factory function to create resources
            cleanup: Optional cleanup function
            config: Pool configuration
        """
        self.name = name
        self.factory = factory
        self.cleanup = cleanup
        self.config = config or PoolConfig()
        
        self.available: List[T] = []
        self.in_use: Dict[str, T] = {}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._total_created = 0
    
    async def acquire(self, timeout: Optional[float] = None) -> T:
        """
        Acquire a resource from pool.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Resource instance
        """
        timeout = timeout or self.config.timeout
        
        async with self._condition:
            # Try to get from available
            if self.available:
                resource = self.available.pop()
                resource_id = id(resource)
                self.in_use[resource_id] = resource
                return resource
            
            # Create new if under max size
            if len(self.in_use) < self.config.max_size:
                resource = await self.factory()
                self._total_created += 1
                resource_id = id(resource)
                self.in_use[resource_id] = resource
                logger.debug(f"Created new resource in pool {self.name}")
                return resource
            
            # Wait for available resource
            try:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                if self.available:
                    resource = self.available.pop()
                    resource_id = id(resource)
                    self.in_use[resource_id] = resource
                    return resource
                raise TimeoutError(f"Timeout waiting for resource in pool {self.name}")
            except asyncio.TimeoutError:
                raise TimeoutError(f"Timeout waiting for resource in pool {self.name}")
    
    async def release(self, resource: T):
        """
        Release a resource back to pool.
        
        Args:
            resource: Resource to release
        """
        async with self._lock:
            resource_id = id(resource)
            
            if resource_id in self.in_use:
                del self.in_use[resource_id]
                self.available.append(resource)
                self._condition.notify_all()
                logger.debug(f"Released resource in pool {self.name}")
    
    @asynccontextmanager
    async def get(self, timeout: Optional[float] = None):
        """
        Context manager for acquiring and releasing resources.
        
        Args:
            timeout: Optional timeout
            
        Usage:
            async with pool.get() as resource:
                # use resource
        """
        resource = await self.acquire(timeout)
        try:
            yield resource
        finally:
            await self.release(resource)
    
    async def close(self):
        """Close pool and cleanup all resources."""
        async with self._lock:
            # Cleanup available resources
            if self.cleanup:
                for resource in self.available:
                    try:
                        await self.cleanup(resource)
                    except Exception as e:
                        logger.error(f"Error cleaning up resource: {e}")
            
            # Cleanup in-use resources
            if self.cleanup:
                for resource in self.in_use.values():
                    try:
                        await self.cleanup(resource)
                    except Exception as e:
                        logger.error(f"Error cleaning up resource: {e}")
            
            self.available.clear()
            self.in_use.clear()
            logger.info(f"Closed pool {self.name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "name": self.name,
            "available": len(self.available),
            "in_use": len(self.in_use),
            "total_created": self._total_created,
            "max_size": self.config.max_size,
            "min_size": self.config.min_size
        }
    
    def get_state(self) -> PoolState:
        """Get current pool state."""
        total = len(self.available) + len(self.in_use)
        
        if total == 0:
            return PoolState.EMPTY
        elif len(self.in_use) >= self.config.max_size:
            return PoolState.FULL
        elif len(self.in_use) > 0:
            return PoolState.BUSY
        else:
            return PoolState.IDLE

