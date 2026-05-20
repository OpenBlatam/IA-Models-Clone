"""
Resource Manager
Advanced resource management and concurrency control
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Resource manager for concurrency control
    
    Features:
    - Semaphore-based concurrency control
    - Resource pooling
    - Resource limits
    - Resource monitoring
    """
    
    def __init__(self):
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._resource_pools: Dict[str, list] = {}
        self._resource_limits: Dict[str, int] = {}
        self._active_resources: Dict[str, int] = defaultdict(int)
    
    def create_semaphore(self, resource_name: str, limit: int) -> asyncio.Semaphore:
        """Create semaphore for resource"""
        if resource_name not in self._semaphores:
            self._semaphores[resource_name] = asyncio.Semaphore(limit)
            self._resource_limits[resource_name] = limit
            logger.info(f"Created semaphore for {resource_name}: limit={limit}")
        
        return self._semaphores[resource_name]
    
    @asynccontextmanager
    async def acquire_resource(self, resource_name: str):
        """Acquire resource with context manager"""
        semaphore = self._semaphores.get(resource_name)
        
        if not semaphore:
            # Create default semaphore
            semaphore = self.create_semaphore(resource_name, 10)
        
        async with semaphore:
            self._active_resources[resource_name] += 1
            try:
                yield
            finally:
                self._active_resources[resource_name] = max(
                    0,
                    self._active_resources[resource_name] - 1
                )
    
    def get_resource_usage(self, resource_name: str) -> Dict[str, Any]:
        """Get resource usage statistics"""
        limit = self._resource_limits.get(resource_name, 0)
        active = self._active_resources.get(resource_name, 0)
        
        return {
            "resource": resource_name,
            "limit": limit,
            "active": active,
            "available": limit - active,
            "utilization": active / limit if limit > 0 else 0
        }
    
    def create_resource_pool(
        self,
        pool_name: str,
        factory: Callable,
        size: int = 10
    ) -> None:
        """Create resource pool"""
        self._resource_pools[pool_name] = [factory() for _ in range(size)]
        logger.info(f"Created resource pool: {pool_name} (size: {size})")
    
    def get_from_pool(self, pool_name: str):
        """Get resource from pool"""
        if pool_name in self._resource_pools and self._resource_pools[pool_name]:
            return self._resource_pools[pool_name].pop()
        return None
    
    def return_to_pool(self, pool_name: str, resource: Any) -> None:
        """Return resource to pool"""
        if pool_name in self._resource_pools:
            self._resource_pools[pool_name].append(resource)


# Global resource manager
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager















