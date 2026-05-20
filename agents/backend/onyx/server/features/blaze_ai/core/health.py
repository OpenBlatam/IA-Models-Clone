"""
Health System Components
========================

This module contains the health monitoring components for the Blaze AI system.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums
# =============================================================================

class HealthStatus(Enum):
    """System health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Component type enumeration."""
    CORE = "core"
    ENGINE = "engine"
    MANAGER = "manager"
    SERVICE = "service"
    UTILITY = "utility"
    INTERFACE = "interface"
    CACHE = "cache"
    MONITOR = "monitor"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ComponentHealth:
    """Health information for a single component."""
    
    component_id: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details
        }


@dataclass
class SystemHealth:
    """Overall system health information."""
    
    system_name: str = "Blaze AI"
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    last_updated: float = field(default_factory=time.time)
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    
    def add_component(self, component_id: str, component_type: ComponentType, 
                     status: HealthStatus, message: str, details: Optional[Dict[str, Any]] = None):
        """Add or update component health information."""
        self.components[component_id] = ComponentHealth(
            component_id=component_id,
            component_type=component_type,
            status=status,
            message=message,
            details=details or {}
        )
        self.last_updated = time.time()
        self._update_overall_status()
    
    def update_component(self, component_id: str, status: HealthStatus, 
                        message: str, details: Optional[Dict[str, Any]] = None):
        """Update existing component health information."""
        if component_id in self.components:
            component = self.components[component_id]
            component.status = status
            component.message = message
            component.timestamp = time.time()
            if details:
                component.details.update(details)
            
            self.last_updated = time.time()
            self._update_overall_status()
    
    def remove_component(self, component_id: str):
        """Remove component health information."""
        if component_id in self.components:
            del self.components[component_id]
            self.last_updated = time.time()
            self._update_overall_status()
    
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health information for a specific component."""
        return self.components.get(component_id)
    
    def get_healthy_components(self) -> List[ComponentHealth]:
        """Get list of healthy components."""
        return [comp for comp in self.components.values() if comp.is_healthy()]
    
    def get_unhealthy_components(self) -> List[ComponentHealth]:
        """Get list of unhealthy components."""
        return [comp for comp in self.components.values() if not comp.is_healthy()]
    
    def _update_overall_status(self):
        """Update overall system health status."""
        if not self.components:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        healthy_count = len(self.get_healthy_components())
        total_count = len(self.components)
        
        if healthy_count == total_count:
            self.overall_status = HealthStatus.HEALTHY
        elif healthy_count >= total_count * 0.8:  # 80% threshold
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.UNHEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary information."""
        total_components = len(self.components)
        healthy_components = len(self.get_healthy_components())
        unhealthy_components = len(self.get_unhealthy_components())
        
        return {
            "system_name": self.system_name,
            "overall_status": self.overall_status.value,
            "last_updated": self.last_updated,
            "total_components": total_components,
            "healthy_components": healthy_components,
            "unhealthy_components": unhealthy_components,
            "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0.0,
            "components": {
                comp_id: comp.to_dict() for comp_id, comp in self.components.items()
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.get_health_summary()


# =============================================================================
# Abstract Base Classes
# =============================================================================

class HealthCheckable(ABC):
    """Abstract base class for components that can report health."""
    
    @abstractmethod
    def get_health_status(self) -> ComponentHealth:
        """Get the health status of this component."""
        pass
    
    @abstractmethod
    async def perform_health_check(self) -> bool:
        """Perform a health check and return success status."""
        pass
