"""
Shared Resources for Flux2 Clothing Changer
===========================================

Advanced shared resource management system.
"""

import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceState(Enum):
    """Resource state."""
    AVAILABLE = "available"
    LOCKED = "locked"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"


@dataclass
class SharedResource:
    """Shared resource information."""
    resource_id: str
    resource_type: str
    capacity: int
    current_usage: int = 0
    state: ResourceState = ResourceState.AVAILABLE
    locked_by: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def available(self) -> int:
        """Get available capacity."""
        return self.capacity - self.current_usage
    
    @property
    def utilization(self) -> float:
        """Get utilization percentage."""
        if self.capacity == 0:
            return 0.0
        return (self.current_usage / self.capacity) * 100.0


class SharedResources:
    """Advanced shared resource management system."""
    
    def __init__(self):
        """Initialize shared resources."""
        self.resources: Dict[str, SharedResource] = {}
        self.resource_locks: Dict[str, Dict[str, Any]] = {}
        self.usage_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def register_resource(
        self,
        resource_id: str,
        resource_type: str,
        capacity: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SharedResource:
        """
        Register shared resource.
        
        Args:
            resource_id: Resource identifier
            resource_type: Resource type
            capacity: Resource capacity
            metadata: Optional metadata
            
        Returns:
            Created resource
        """
        resource = SharedResource(
            resource_id=resource_id,
            resource_type=resource_type,
            capacity=capacity,
            metadata=metadata or {},
        )
        
        self.resources[resource_id] = resource
        logger.info(f"Registered resource: {resource_id}")
        return resource
    
    def acquire(
        self,
        resource_id: str,
        amount: int = 1,
        requester_id: Optional[str] = None,
    ) -> bool:
        """
        Acquire resource.
        
        Args:
            resource_id: Resource identifier
            amount: Amount to acquire
            requester_id: Optional requester identifier
            
        Returns:
            True if acquired
        """
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.state != ResourceState.AVAILABLE:
            return False
        
        if resource.available < amount:
            return False
        
        resource.current_usage += amount
        resource.state = ResourceState.IN_USE
        
        if requester_id:
            resource.locked_by = requester_id
        
        # Record usage
        self.usage_history[resource_id].append({
            "action": "acquire",
            "amount": amount,
            "requester": requester_id,
            "timestamp": time.time(),
        })
        
        logger.debug(f"Acquired {amount} of {resource_id}")
        return True
    
    def release(
        self,
        resource_id: str,
        amount: int = 1,
        requester_id: Optional[str] = None,
    ) -> bool:
        """
        Release resource.
        
        Args:
            resource_id: Resource identifier
            amount: Amount to release
            requester_id: Optional requester identifier
            
        Returns:
            True if released
        """
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.current_usage < amount:
            return False
        
        resource.current_usage -= amount
        
        if resource.current_usage == 0:
            resource.state = ResourceState.AVAILABLE
            resource.locked_by = None
        
        # Record usage
        self.usage_history[resource_id].append({
            "action": "release",
            "amount": amount,
            "requester": requester_id,
            "timestamp": time.time(),
        })
        
        logger.debug(f"Released {amount} of {resource_id}")
        return True
    
    def lock_resource(
        self,
        resource_id: str,
        locker_id: str,
    ) -> bool:
        """
        Lock resource.
        
        Args:
            resource_id: Resource identifier
            locker_id: Locker identifier
            
        Returns:
            True if locked
        """
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.state != ResourceState.AVAILABLE:
            return False
        
        resource.state = ResourceState.LOCKED
        resource.locked_by = locker_id
        
        self.resource_locks[resource_id] = {
            "locked_by": locker_id,
            "locked_at": time.time(),
        }
        
        logger.info(f"Locked resource: {resource_id} by {locker_id}")
        return True
    
    def unlock_resource(self, resource_id: str) -> bool:
        """
        Unlock resource.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            True if unlocked
        """
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.state != ResourceState.LOCKED:
            return False
        
        resource.state = ResourceState.AVAILABLE
        resource.locked_by = None
        
        if resource_id in self.resource_locks:
            del self.resource_locks[resource_id]
        
        logger.info(f"Unlocked resource: {resource_id}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get shared resources statistics."""
        return {
            "total_resources": len(self.resources),
            "available_resources": len([
                r for r in self.resources.values()
                if r.state == ResourceState.AVAILABLE
            ]),
            "locked_resources": len([
                r for r in self.resources.values()
                if r.state == ResourceState.LOCKED
            ]),
            "in_use_resources": len([
                r for r in self.resources.values()
                if r.state == ResourceState.IN_USE
            ]),
        }


