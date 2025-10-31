from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Health Status Entity
===================

Domain entity for representing system health status.
"""



class HealthState(Enum):
    """Health check states."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    
    name: str
    state: HealthState
    message: str = ""
    last_check: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.state == HealthState.HEALTHY


@dataclass
class HealthStatus:
    """Overall system health status."""
    
    overall_state: HealthState
    timestamp: datetime
    version: str
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    
    @classmethod
    def create_healthy(cls, version: str) -> "HealthStatus":
        """Create a healthy status."""
        return cls(
            overall_state=HealthState.HEALTHY,
            timestamp=datetime.utcnow(),
            version=version
        )
    
    def add_component_check(self, component: ComponentHealth):
        """Add a component health check."""
        self.components[component.name] = component
        self._update_overall_state()
    
    def _update_overall_state(self) -> Any:
        """Update overall health state based on components."""
        if not self.components:
            self.overall_state = HealthState.UNKNOWN
            return
        
        healthy_count = sum(1 for comp in self.components.values() if comp.is_healthy())
        total_count = len(self.components)
        
        if healthy_count == total_count:
            self.overall_state = HealthState.HEALTHY
        elif healthy_count == 0:
            self.overall_state = HealthState.UNHEALTHY
        else:
            self.overall_state = HealthState.DEGRADED
    
    def is_ready(self) -> bool:
        """Check if system is ready to serve requests."""
        return self.overall_state in [HealthState.HEALTHY, HealthState.DEGRADED]
    
    def is_alive(self) -> bool:
        """Check if system is alive (basic liveness)."""
        return self.overall_state != HealthState.UNHEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.overall_state.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "checks": {
                name: {
                    "status": comp.state.value,
                    "message": comp.message,
                    "last_check": comp.last_check.isoformat(),
                    "details": comp.details
                }
                for name, comp in self.components.items()
            }
        } 