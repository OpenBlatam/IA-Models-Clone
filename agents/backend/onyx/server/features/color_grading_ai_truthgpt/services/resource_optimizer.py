"""
Resource Optimizer for Color Grading AI
========================================

Intelligent resource optimization with dynamic allocation and monitoring.
"""

import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


@dataclass
class ResourceUsage:
    """Resource usage information."""
    resource_type: ResourceType
    current: float
    max: float
    percentage: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Resource allocation."""
    resource_type: ResourceType
    allocated: float
    reserved: float
    available: float
    timestamp: datetime = field(default_factory=datetime.now)


class ResourceOptimizer:
    """
    Resource optimizer with intelligent allocation.
    
    Features:
    - Resource monitoring
    - Dynamic allocation
    - Load balancing
    - Auto-scaling
    - Resource limits
    - Optimization recommendations
    """
    
    def __init__(self):
        """Initialize resource optimizer."""
        self._allocations: Dict[ResourceType, ResourceAllocation] = {}
        self._usage_history: Dict[ResourceType, List[ResourceUsage]] = {}
        self._max_history = 1000
        self._limits: Dict[ResourceType, float] = {
            ResourceType.CPU: 80.0,  # 80% CPU limit
            ResourceType.MEMORY: 85.0,  # 85% memory limit
            ResourceType.DISK: 90.0,  # 90% disk limit
        }
        self._reservations: Dict[ResourceType, float] = {}
    
    def get_resource_usage(self, resource_type: ResourceType) -> ResourceUsage:
        """
        Get current resource usage.
        
        Args:
            resource_type: Resource type
            
        Returns:
            Resource usage
        """
        if resource_type == ResourceType.CPU:
            current = psutil.cpu_percent(interval=0.1)
            max_val = 100.0
        elif resource_type == ResourceType.MEMORY:
            mem = psutil.virtual_memory()
            current = mem.used / (1024 ** 3)  # GB
            max_val = mem.total / (1024 ** 3)  # GB
        elif resource_type == ResourceType.DISK:
            disk = psutil.disk_usage('/')
            current = disk.used / (1024 ** 3)  # GB
            max_val = disk.total / (1024 ** 3)  # GB
        else:
            current = 0.0
            max_val = 100.0
        
        percentage = (current / max_val * 100) if max_val > 0 else 0.0
        
        usage = ResourceUsage(
            resource_type=resource_type,
            current=current,
            max=max_val,
            percentage=percentage
        )
        
        # Store in history
        if resource_type not in self._usage_history:
            self._usage_history[resource_type] = []
        
        self._usage_history[resource_type].append(usage)
        if len(self._usage_history[resource_type]) > self._max_history:
            self._usage_history[resource_type] = self._usage_history[resource_type][-self._max_history:]
        
        return usage
    
    def get_all_resource_usage(self) -> Dict[ResourceType, ResourceUsage]:
        """Get usage for all resources."""
        return {
            resource_type: self.get_resource_usage(resource_type)
            for resource_type in ResourceType
        }
    
    def allocate_resource(
        self,
        resource_type: ResourceType,
        amount: float
    ) -> bool:
        """
        Allocate resource.
        
        Args:
            resource_type: Resource type
            amount: Amount to allocate
            
        Returns:
            True if allocated successfully
        """
        usage = self.get_resource_usage(resource_type)
        limit = self._limits.get(resource_type, 100.0)
        reserved = self._reservations.get(resource_type, 0.0)
        
        # Check if allocation is possible
        available = (limit - usage.percentage) - reserved
        
        if amount > available:
            logger.warning(
                f"Cannot allocate {amount}% of {resource_type.value}: "
                f"only {available:.2f}% available"
            )
            return False
        
        # Allocate
        if resource_type not in self._allocations:
            self._allocations[resource_type] = ResourceAllocation(
                resource_type=resource_type,
                allocated=0.0,
                reserved=reserved,
                available=available
            )
        
        allocation = self._allocations[resource_type]
        allocation.allocated += amount
        allocation.available = available - allocation.allocated
        
        logger.info(f"Allocated {amount}% of {resource_type.value}")
        
        return True
    
    def release_resource(self, resource_type: ResourceType, amount: float):
        """Release allocated resource."""
        if resource_type in self._allocations:
            allocation = self._allocations[resource_type]
            allocation.allocated = max(0.0, allocation.allocated - amount)
            usage = self.get_resource_usage(resource_type)
            limit = self._limits.get(resource_type, 100.0)
            reserved = self._reservations.get(resource_type, 0.0)
            allocation.available = (limit - usage.percentage) - reserved - allocation.allocated
            
            logger.info(f"Released {amount}% of {resource_type.value}")
    
    def set_resource_limit(self, resource_type: ResourceType, limit: float):
        """Set resource limit."""
        self._limits[resource_type] = limit
        logger.info(f"Set {resource_type.value} limit to {limit}%")
    
    def reserve_resource(self, resource_type: ResourceType, amount: float):
        """Reserve resource."""
        self._reservations[resource_type] = self._reservations.get(resource_type, 0.0) + amount
        logger.info(f"Reserved {amount}% of {resource_type.value}")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations."""
        recommendations = []
        
        for resource_type in ResourceType:
            usage = self.get_resource_usage(resource_type)
            limit = self._limits.get(resource_type, 100.0)
            
            if usage.percentage > limit * 0.9:
                recommendations.append({
                    "resource": resource_type.value,
                    "issue": "high_usage",
                    "current": usage.percentage,
                    "limit": limit,
                    "recommendation": f"Reduce {resource_type.value} usage or increase limit"
                })
            
            # Check for trends
            history = self._usage_history.get(resource_type, [])
            if len(history) >= 10:
                recent = history[-10:]
                avg_usage = sum(u.percentage for u in recent) / len(recent)
                
                if avg_usage > limit * 0.8:
                    recommendations.append({
                        "resource": resource_type.value,
                        "issue": "sustained_high_usage",
                        "average": avg_usage,
                        "recommendation": f"Consider scaling or optimization for {resource_type.value}"
                    })
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource optimizer statistics."""
        all_usage = self.get_all_resource_usage()
        
        return {
            "current_usage": {
                rt.value: {
                    "current": u.current,
                    "max": u.max,
                    "percentage": u.percentage
                }
                for rt, u in all_usage.items()
            },
            "allocations": {
                rt.value: {
                    "allocated": a.allocated,
                    "reserved": a.reserved,
                    "available": a.available
                }
                for rt, a in self._allocations.items()
            },
            "limits": {rt.value: limit for rt, limit in self._limits.items()},
            "recommendations_count": len(self.get_optimization_recommendations()),
        }




