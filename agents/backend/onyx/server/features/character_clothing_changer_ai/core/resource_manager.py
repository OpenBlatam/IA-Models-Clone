"""
Resource Manager
================

System for managing system resources and limits.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, resource management features limited")


class ResourceType(Enum):
    """Resource type."""
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    FILE_DESCRIPTORS = "file_descriptors"
    THREADS = "threads"


@dataclass
class ResourceLimit:
    """Resource limit."""
    resource_type: ResourceType
    limit: float
    unit: str = "percent"  # percent, bytes, count
    action: str = "warn"  # warn, block, throttle


@dataclass
class ResourceUsage:
    """Resource usage."""
    resource_type: ResourceType
    current: float
    limit: Optional[float] = None
    percentage: float = 0.0
    unit: str = "percent"
    timestamp: datetime = field(default_factory=datetime.now)


class ResourceManager:
    """Resource manager."""
    
    def __init__(self):
        """Initialize resource manager."""
        self.limits: Dict[ResourceType, ResourceLimit] = {}
        self.usage_history: List[ResourceUsage] = []
        self.max_history = 10000
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
            logger.warning("psutil not available, resource monitoring limited")
    
    def set_limit(
        self,
        resource_type: ResourceType,
        limit: float,
        unit: str = "percent",
        action: str = "warn"
    ):
        """
        Set resource limit.
        
        Args:
            resource_type: Resource type
            limit: Limit value
            unit: Unit (percent, bytes, count)
            action: Action on limit (warn, block, throttle)
        """
        resource_limit = ResourceLimit(
            resource_type=resource_type,
            limit=limit,
            unit=unit,
            action=action
        )
        self.limits[resource_type] = resource_limit
        logger.info(f"Set limit for {resource_type.value}: {limit} {unit}")
    
    def get_usage(self, resource_type: ResourceType) -> ResourceUsage:
        """
        Get current resource usage.
        
        Args:
            resource_type: Resource type
            
        Returns:
            Resource usage
        """
        if not PSUTIL_AVAILABLE or not self.process:
            return ResourceUsage(
                resource_type=resource_type,
                current=0.0,
                percentage=0.0
            )
        
        if resource_type == ResourceType.MEMORY:
            memory_info = self.process.memory_info()
            current = memory_info.rss
            system_memory = psutil.virtual_memory()
            percentage = (current / system_memory.total) * 100
            limit = self.limits.get(resource_type)
            
            return ResourceUsage(
                resource_type=resource_type,
                current=current,
                limit=limit.limit if limit else None,
                percentage=percentage,
                unit="bytes"
            )
        
        elif resource_type == ResourceType.CPU:
            current = self.process.cpu_percent(interval=0.1)
            limit = self.limits.get(resource_type)
            percentage = (current / limit.limit * 100) if limit else current
            
            return ResourceUsage(
                resource_type=resource_type,
                current=current,
                limit=limit.limit if limit else None,
                percentage=percentage,
                unit="percent"
            )
        
        elif resource_type == ResourceType.DISK:
            disk = psutil.disk_usage('/')
            current = disk.used
            percentage = (disk.used / disk.total) * 100
            limit = self.limits.get(resource_type)
            
            return ResourceUsage(
                resource_type=resource_type,
                current=current,
                limit=limit.limit if limit else None,
                percentage=percentage,
                unit="bytes"
            )
        
        elif resource_type == ResourceType.THREADS:
            current = self.process.num_threads()
            limit = self.limits.get(resource_type)
            percentage = (current / limit.limit * 100) if limit else 0.0
            
            return ResourceUsage(
                resource_type=resource_type,
                current=current,
                limit=limit.limit if limit else None,
                percentage=percentage,
                unit="count"
            )
        
        else:
            return ResourceUsage(
                resource_type=resource_type,
                current=0.0,
                percentage=0.0
            )
    
    def check_limits(self) -> Dict[ResourceType, bool]:
        """
        Check all resource limits.
        
        Returns:
            Dictionary of limit violations
        """
        violations = {}
        
        for resource_type in ResourceType:
            usage = self.get_usage(resource_type)
            limit = self.limits.get(resource_type)
            
            if limit:
                if limit.unit == "percent":
                    exceeds = usage.percentage > limit.limit
                else:
                    exceeds = usage.current > limit.limit
                
                violations[resource_type] = exceeds
                
                if exceeds:
                    if limit.action == "warn":
                        logger.warning(f"Resource limit exceeded: {resource_type.value} ({usage.percentage:.1f}%)")
                    elif limit.action == "block":
                        logger.error(f"Resource limit blocked: {resource_type.value}")
                    elif limit.action == "throttle":
                        logger.warning(f"Resource limit throttling: {resource_type.value}")
            else:
                violations[resource_type] = False
        
        return violations
    
    def get_all_usage(self) -> Dict[ResourceType, ResourceUsage]:
        """Get usage for all resources."""
        return {
            resource_type: self.get_usage(resource_type)
            for resource_type in ResourceType
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        all_usage = self.get_all_usage()
        violations = self.check_limits()
        
        return {
            "limits": {
                rt.value: {
                    "limit": limit.limit,
                    "unit": limit.unit,
                    "action": limit.action
                }
                for rt, limit in self.limits.items()
            },
            "usage": {
                rt.value: {
                    "current": usage.current,
                    "percentage": usage.percentage,
                    "unit": usage.unit
                }
                for rt, usage in all_usage.items()
            },
            "violations": {
                rt.value: violated
                for rt, violated in violations.items()
            }
        }

