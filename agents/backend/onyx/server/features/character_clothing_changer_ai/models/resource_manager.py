"""
Resource Manager for Flux2 Clothing Changer
===========================================

Advanced resource management and allocation.
"""

import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class Resource:
    """Resource information."""
    resource_id: str
    resource_type: ResourceType
    capacity: float
    allocated: float = 0.0
    reserved: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def available(self) -> float:
        """Get available capacity."""
        return self.capacity - self.allocated - self.reserved
    
    @property
    def utilization(self) -> float:
        """Get utilization percentage."""
        if self.capacity == 0:
            return 0.0
        return (self.allocated / self.capacity) * 100.0


class ResourceManager:
    """Advanced resource management system."""
    
    def __init__(self):
        """Initialize resource manager."""
        self.resources: Dict[str, Resource] = {}
        self.allocations: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def register_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        capacity: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Resource:
        """
        Register resource.
        
        Args:
            resource_id: Resource identifier
            resource_type: Resource type
            capacity: Resource capacity
            metadata: Optional metadata
            
        Returns:
            Created resource
        """
        resource = Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            capacity=capacity,
            metadata=metadata or {},
        )
        
        self.resources[resource_id] = resource
        logger.info(f"Registered resource: {resource_id} ({resource_type.value})")
        return resource
    
    def allocate(
        self,
        resource_id: str,
        amount: float,
        allocation_id: Optional[str] = None,
    ) -> bool:
        """
        Allocate resource.
        
        Args:
            resource_id: Resource identifier
            amount: Amount to allocate
            allocation_id: Optional allocation identifier
            
        Returns:
            True if allocated
        """
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.available < amount:
            logger.warning(f"Insufficient {resource_id}: available {resource.available}, requested {amount}")
            return False
        
        resource.allocated += amount
        
        if allocation_id:
            self.allocations[allocation_id][resource_id] = amount
        
        logger.debug(f"Allocated {amount} of {resource_id}")
        return True
    
    def deallocate(
        self,
        resource_id: str,
        amount: float,
        allocation_id: Optional[str] = None,
    ) -> bool:
        """
        Deallocate resource.
        
        Args:
            resource_id: Resource identifier
            amount: Amount to deallocate
            allocation_id: Optional allocation identifier
            
        Returns:
            True if deallocated
        """
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.allocated < amount:
            logger.warning(f"Cannot deallocate {amount} from {resource_id}: only {resource.allocated} allocated")
            return False
        
        resource.allocated -= amount
        
        if allocation_id and allocation_id in self.allocations:
            if resource_id in self.allocations[allocation_id]:
                del self.allocations[allocation_id][resource_id]
        
        logger.debug(f"Deallocated {amount} of {resource_id}")
        return True
    
    def reserve(
        self,
        resource_id: str,
        amount: float,
    ) -> bool:
        """
        Reserve resource.
        
        Args:
            resource_id: Resource identifier
            amount: Amount to reserve
            
        Returns:
            True if reserved
        """
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.available < amount:
            return False
        
        resource.reserved += amount
        logger.debug(f"Reserved {amount} of {resource_id}")
        return True
    
    def release_reservation(
        self,
        resource_id: str,
        amount: float,
    ) -> bool:
        """
        Release reservation.
        
        Args:
            resource_id: Resource identifier
            amount: Amount to release
            
        Returns:
            True if released
        """
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.reserved < amount:
            return False
        
        resource.reserved -= amount
        logger.debug(f"Released reservation of {amount} from {resource_id}")
        return True
    
    def get_resource_status(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get resource status."""
        if resource_id not in self.resources:
            return None
        
        resource = self.resources[resource_id]
        
        return {
            "resource_id": resource_id,
            "type": resource.resource_type.value,
            "capacity": resource.capacity,
            "allocated": resource.allocated,
            "reserved": resource.reserved,
            "available": resource.available,
            "utilization": resource.utilization,
        }
    
    def get_all_resources_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all resources."""
        return {
            resource_id: self.get_resource_status(resource_id)
            for resource_id in self.resources.keys()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        return {
            "total_resources": len(self.resources),
            "resources_by_type": {
                rtype.value: len([r for r in self.resources.values() if r.resource_type == rtype])
                for rtype in ResourceType
            },
            "total_allocations": len(self.allocations),
        }


