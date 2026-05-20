"""
Resource Pool Manager
=====================

Advanced resource pool management with automatic scaling and health monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ResourceState(str, Enum):
    """Resource states."""
    IDLE = "idle"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"
    RESERVED = "reserved"

@dataclass
class Resource:
    """Resource definition."""
    resource_id: str
    resource_type: str
    state: ResourceState
    metadata: Dict[str, Any] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    health_score: float = 1.0

class ResourcePoolManager:
    """Advanced resource pool manager."""
    
    def __init__(
        self,
        pool_name: str,
        min_size: int = 2,
        max_size: int = 10,
        factory: Optional[Callable] = None
    ):
        self.pool_name = pool_name
        self.min_size = min_size
        self.max_size = max_size
        self.factory = factory
        self.resources: Dict[str, Resource] = {}
        self.available_resources: List[str] = []
        self.maintenance_task = None
        self.is_running = False
    
    def add_resource(
        self,
        resource_id: str,
        resource_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Resource:
        """Add a resource to the pool."""
        resource = Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            state=ResourceState.IDLE,
            metadata=metadata or {}
        )
        
        self.resources[resource_id] = resource
        self.available_resources.append(resource_id)
        
        logger.info(f"Resource added to pool {self.pool_name}: {resource_id}")
        
        return resource
    
    async def acquire_resource(self, timeout: float = 30.0) -> Optional[Resource]:
        """Acquire a resource from the pool."""
        start_time = datetime.now()
        
        while True:
            # Check for available resource
            if self.available_resources:
                resource_id = self.available_resources.pop(0)
                resource = self.resources[resource_id]
                resource.state = ResourceState.BUSY
                resource.last_used = datetime.now()
                resource.usage_count += 1
                
                logger.debug(f"Resource acquired: {resource_id}")
                return resource
            
            # Check if we can create new resource
            if len(self.resources) < self.max_size and self.factory:
                try:
                    new_resource = await self._create_resource()
                    if new_resource:
                        new_resource.state = ResourceState.BUSY
                        new_resource.last_used = datetime.now()
                        return new_resource
                except Exception as e:
                    logger.error(f"Failed to create resource: {e}")
            
            # Check timeout
            if (datetime.now() - start_time).total_seconds() > timeout:
                logger.warning(f"Timeout acquiring resource from pool {self.pool_name}")
                return None
            
            # Wait before retry
            await asyncio.sleep(0.1)
    
    async def release_resource(self, resource_id: str):
        """Release a resource back to the pool."""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            resource.state = ResourceState.IDLE
            
            if resource_id not in self.available_resources:
                self.available_resources.append(resource_id)
            
            logger.debug(f"Resource released: {resource_id}")
    
    async def _create_resource(self) -> Optional[Resource]:
        """Create a new resource using factory."""
        if not self.factory:
            return None
        
        try:
            import uuid
            resource_id = str(uuid.uuid4())
            
            if asyncio.iscoroutinefunction(self.factory):
                resource_data = await self.factory()
            else:
                resource_data = self.factory()
            
            resource = Resource(
                resource_id=resource_id,
                resource_type="dynamic",
                state=ResourceState.IDLE,
                metadata={"factory_created": True, "data": resource_data}
            )
            
            self.resources[resource_id] = resource
            self.available_resources.append(resource_id)
            
            logger.info(f"Resource created: {resource_id}")
            return resource
            
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            return None
    
    async def _maintenance_loop(self):
        """Background maintenance loop."""
        while self.is_running:
            try:
                # Ensure minimum pool size
                while len(self.resources) < self.min_size and self.factory:
                    await self._create_resource()
                
                # Remove unhealthy resources
                unhealthy = [
                    rid for rid, resource in self.resources.items()
                    if resource.state == ResourceState.UNHEALTHY
                ]
                
                for resource_id in unhealthy:
                    await self._remove_resource(resource_id)
                
                await asyncio.sleep(60.0)
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(60.0)
    
    async def _remove_resource(self, resource_id: str):
        """Remove a resource from the pool."""
        if resource_id in self.resources:
            del self.resources[resource_id]
            if resource_id in self.available_resources:
                self.available_resources.remove(resource_id)
            logger.info(f"Resource removed: {resource_id}")
    
    def set_resource_health(self, resource_id: str, health_score: float):
        """Set health score for a resource."""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            resource.health_score = health_score
            
            if health_score < 0.5:
                resource.state = ResourceState.UNHEALTHY
                if resource_id in self.available_resources:
                    self.available_resources.remove(resource_id)
    
    async def start_maintenance(self):
        """Start maintenance task."""
        if self.is_running:
            return
        
        self.is_running = True
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info(f"Resource pool maintenance started: {self.pool_name}")
    
    async def stop_maintenance(self):
        """Stop maintenance task."""
        self.is_running = False
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Resource pool maintenance stopped: {self.pool_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        idle = sum(1 for r in self.resources.values() if r.state == ResourceState.IDLE)
        busy = sum(1 for r in self.resources.values() if r.state == ResourceState.BUSY)
        unhealthy = sum(1 for r in self.resources.values() if r.state == ResourceState.UNHEALTHY)
        
        return {
            "pool_name": self.pool_name,
            "total_resources": len(self.resources),
            "idle": idle,
            "busy": busy,
            "unhealthy": unhealthy,
            "available": len(self.available_resources),
            "min_size": self.min_size,
            "max_size": self.max_size,
            "utilization": round((busy / len(self.resources) * 100) if self.resources else 0, 2)
        }

# Global instances
resource_pools: Dict[str, ResourcePoolManager] = {}
















