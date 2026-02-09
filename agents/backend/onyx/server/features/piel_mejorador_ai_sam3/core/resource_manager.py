"""
Resource Manager for Piel Mejorador AI SAM3
===========================================

Centralized resource management.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class Resource:
    """Resource definition."""
    name: str
    resource: Any
    cleanup_func: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)


class ResourceManager:
    """
    Centralized resource management.
    
    Features:
    - Resource registration
    - Automatic cleanup
    - Resource tracking
    - Lifecycle management
    """
    
    def __init__(self):
        """Initialize resource manager."""
        self._resources: Dict[str, Resource] = {}
        self._cleanup_order: List[str] = []
    
    def register(
        self,
        name: str,
        resource: Any,
        cleanup_func: Optional[Callable] = None
    ):
        """
        Register a resource.
        
        Args:
            name: Resource name
            resource: Resource object
            cleanup_func: Optional cleanup function
        """
        self._resources[name] = Resource(
            name=name,
            resource=resource,
            cleanup_func=cleanup_func
        )
        self._cleanup_order.append(name)
        logger.debug(f"Registered resource: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a resource.
        
        Args:
            name: Resource name
            
        Returns:
            Resource or None
        """
        resource_def = self._resources.get(name)
        return resource_def.resource if resource_def else None
    
    async def cleanup(self, name: Optional[str] = None):
        """
        Cleanup resources.
        
        Args:
            name: Optional resource name (all if None)
        """
        if name:
            await self._cleanup_resource(name)
        else:
            # Cleanup in reverse order
            for resource_name in reversed(self._cleanup_order):
                if resource_name in self._resources:
                    await self._cleanup_resource(resource_name)
    
    async def _cleanup_resource(self, name: str):
        """Cleanup a single resource."""
        resource_def = self._resources.get(name)
        if not resource_def:
            return
        
        try:
            if resource_def.cleanup_func:
                if asyncio.iscoroutinefunction(resource_def.cleanup_func):
                    await resource_def.cleanup_func(resource_def.resource)
                else:
                    resource_def.cleanup_func(resource_def.resource)
            elif hasattr(resource_def.resource, 'close'):
                close_method = resource_def.resource.close
                if asyncio.iscoroutinefunction(close_method):
                    await close_method()
                else:
                    close_method()
            elif hasattr(resource_def.resource, 'cleanup'):
                cleanup_method = resource_def.resource.cleanup
                if asyncio.iscoroutinefunction(cleanup_method):
                    await cleanup_method()
                else:
                    cleanup_method()
            
            del self._resources[name]
            logger.debug(f"Cleaned up resource: {name}")
        except Exception as e:
            logger.error(f"Error cleaning up resource {name}: {e}")
    
    def list_resources(self) -> List[str]:
        """List all registered resources."""
        return list(self._resources.keys())
    
    @asynccontextmanager
    async def managed_resource(self, name: str, resource: Any, cleanup_func: Optional[Callable] = None):
        """
        Context manager for resource.
        
        Args:
            name: Resource name
            resource: Resource object
            cleanup_func: Optional cleanup function
        """
        self.register(name, resource, cleanup_func)
        try:
            yield resource
        finally:
            await self.cleanup(name)




