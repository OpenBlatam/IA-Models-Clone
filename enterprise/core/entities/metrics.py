from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Metrics Data Entity
==================

Domain entity for representing application metrics.
"""



@dataclass
class MetricsData:
    """Application metrics data."""
    
    timestamp: datetime
    request_count: int = 0
    error_count: int = 0
    average_response_time: float = 0.0
    cache_hit_ratio: float = 0.0
    active_connections: int = 0
    circuit_breaker_states: Dict[str, str] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_empty(cls) -> "MetricsData":
        """Create empty metrics data."""
        return cls(timestamp=datetime.utcnow())
    
    def add_request(self, response_time: float, is_error: bool = False):
        """Add a request to metrics."""
        self.request_count += 1
        if is_error:
            self.error_count += 1
        
        # Update average response time
        if self.request_count == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.request_count - 1) + response_time) 
                / self.request_count
            )
    
    def update_cache_hit_ratio(self, hit_ratio: float):
        """Update cache hit ratio."""
        self.cache_hit_ratio = max(0.0, min(1.0, hit_ratio))
    
    def update_circuit_breaker_state(self, service: str, state: str):
        """Update circuit breaker state for a service."""
        self.circuit_breaker_states[service] = state
    
    def get_error_rate(self) -> float:
        """Calculate error rate."""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.get_error_rate(),
            "average_response_time": self.average_response_time,
            "cache_hit_ratio": self.cache_hit_ratio,
            "active_connections": self.active_connections,
            "circuit_breaker_states": self.circuit_breaker_states,
            "custom_metrics": self.custom_metrics
        } 