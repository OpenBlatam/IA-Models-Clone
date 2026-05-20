"""
Resource Abstraction for Reinforcement Learning
================================================

Paper: "Resource Abstraction for Reinforcement Learning"

Key concepts:
- Resource abstraction in RL
- Hierarchical resource management
- Resource allocation strategies
- Multi-resource optimization
- Abstraction levels
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class ResourceType(Enum):
    """Types of resources."""
    COMPUTATIONAL = "computational"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    ENERGY = "energy"
    TIME = "time"


class AbstractionLevel(Enum):
    """Levels of abstraction."""
    LOW_LEVEL = "low_level"
    MID_LEVEL = "mid_level"
    HIGH_LEVEL = "high_level"
    ABSTRACT = "abstract"


@dataclass
class Resource:
    """A resource."""
    resource_id: str
    resource_type: ResourceType
    capacity: float
    current_usage: float = 0.0
    abstraction_level: AbstractionLevel = AbstractionLevel.MID_LEVEL
    
    @property
    def available(self) -> float:
        """Available capacity."""
        return max(0.0, self.capacity - self.current_usage)
    
    @property
    def utilization(self) -> float:
        """Utilization percentage."""
        return self.current_usage / self.capacity if self.capacity > 0 else 0.0


@dataclass
class ResourceAllocation:
    """Resource allocation."""
    allocation_id: str
    resource_id: str
    amount: float
    priority: int
    timestamp: datetime = field(default_factory=datetime.now)


class ResourceAbstractionManager:
    """
    Resource abstraction manager for RL agents.
    
    Manages resources at different abstraction levels.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize resource manager.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.resources: Dict[str, Resource] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_history: List[ResourceAllocation] = []
        
        # Initialize default resources
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize default resources."""
        default_resources = {
            "cpu": Resource(
                resource_id="cpu",
                resource_type=ResourceType.COMPUTATIONAL,
                capacity=100.0,
                abstraction_level=AbstractionLevel.LOW_LEVEL
            ),
            "memory": Resource(
                resource_id="memory",
                resource_type=ResourceType.MEMORY,
                capacity=1000.0,  # MB
                abstraction_level=AbstractionLevel.MID_LEVEL
            ),
            "network": Resource(
                resource_id="network",
                resource_type=ResourceType.NETWORK,
                capacity=100.0,  # Mbps
                abstraction_level=AbstractionLevel.MID_LEVEL
            ),
            "energy": Resource(
                resource_id="energy",
                resource_type=ResourceType.ENERGY,
                capacity=100.0,  # Percentage
                abstraction_level=AbstractionLevel.HIGH_LEVEL
            )
        }
        
        self.resources.update(default_resources)
    
    def allocate_resource(
        self,
        resource_id: str,
        amount: float,
        priority: int = 1
    ) -> Optional[ResourceAllocation]:
        """
        Allocate a resource.
        
        Args:
            resource_id: Resource identifier
            amount: Amount to allocate
            priority: Allocation priority
            
        Returns:
            Resource allocation or None if not available
        """
        if resource_id not in self.resources:
            return None
        
        resource = self.resources[resource_id]
        
        if resource.available < amount:
            return None
        
        allocation = ResourceAllocation(
            allocation_id=f"alloc_{datetime.now().timestamp()}",
            resource_id=resource_id,
            amount=amount,
            priority=priority
        )
        
        # Update resource usage
        resource.current_usage += amount
        
        self.allocations[allocation.allocation_id] = allocation
        self.allocation_history.append(allocation)
        
        return allocation
    
    def deallocate_resource(self, allocation_id: str) -> bool:
        """
        Deallocate a resource.
        
        Args:
            allocation_id: Allocation identifier
            
        Returns:
            True if deallocated successfully
        """
        if allocation_id not in self.allocations:
            return False
        
        allocation = self.allocations[allocation_id]
        resource = self.resources.get(allocation.resource_id)
        
        if resource:
            resource.current_usage = max(0.0, resource.current_usage - allocation.amount)
        
        del self.allocations[allocation_id]
        return True
    
    def get_resource_status(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a resource.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            Resource status
        """
        resource = self.resources.get(resource_id)
        if not resource:
            return None
        
        return {
            "resource_id": resource.resource_id,
            "resource_type": resource.resource_type.value,
            "capacity": resource.capacity,
            "current_usage": resource.current_usage,
            "available": resource.available,
            "utilization": resource.utilization,
            "abstraction_level": resource.abstraction_level.value
        }
    
    def optimize_allocation(
        self,
        requirements: Dict[str, float],
        priorities: Optional[Dict[str, int]] = None
    ) -> Dict[str, ResourceAllocation]:
        """
        Optimize resource allocation for requirements.
        
        Args:
            requirements: Dictionary of resource_id -> amount needed
            priorities: Optional priorities for each resource
            
        Returns:
            Dictionary of allocations
        """
        allocations = {}
        priorities = priorities or {}
        
        # Sort by priority
        sorted_requirements = sorted(
            requirements.items(),
            key=lambda x: priorities.get(x[0], 1),
            reverse=True
        )
        
        for resource_id, amount in sorted_requirements:
            priority = priorities.get(resource_id, 1)
            allocation = self.allocate_resource(resource_id, amount, priority)
            if allocation:
                allocations[resource_id] = allocation
        
        return allocations
    
    def get_abstraction_level(
        self,
        resource_id: str
    ) -> Optional[AbstractionLevel]:
        """Get abstraction level of a resource."""
        resource = self.resources.get(resource_id)
        return resource.abstraction_level if resource else None
    
    def set_abstraction_level(
        self,
        resource_id: str,
        level: AbstractionLevel
    ) -> bool:
        """
        Set abstraction level of a resource.
        
        Args:
            resource_id: Resource identifier
            level: Abstraction level
            
        Returns:
            True if set successfully
        """
        resource = self.resources.get(resource_id)
        if not resource:
            return False
        
        resource.abstraction_level = level
        return True
    
    def get_all_resources_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all resources."""
        return {
            resource_id: self.get_resource_status(resource_id)
            for resource_id in self.resources.keys()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource management statistics."""
        if not self.resources:
            return {}
        
        total_capacity = sum(r.capacity for r in self.resources.values())
        total_usage = sum(r.current_usage for r in self.resources.values())
        avg_utilization = sum(r.utilization for r in self.resources.values()) / len(self.resources) if self.resources else 0.0
        
        return {
            "total_resources": len(self.resources),
            "total_capacity": total_capacity,
            "total_usage": total_usage,
            "total_available": total_capacity - total_usage,
            "average_utilization": avg_utilization,
            "active_allocations": len(self.allocations),
            "allocation_history_count": len(self.allocation_history)
        }



