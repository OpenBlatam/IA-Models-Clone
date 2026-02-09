"""
Auto-Scaling Patterns
Automatic scaling based on metrics and load
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScalingPolicy(Enum):
    """Scaling policy types"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    QUEUE_BASED = "queue_based"
    CUSTOM = "custom"


@dataclass
class ScalingMetrics:
    """Scaling metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    queue_size: int = 0
    response_time: float = 0.0
    error_rate: float = 0.0


class AutoScaler:
    """
    Auto-scaler for services
    
    Features:
    - CPU-based scaling
    - Memory-based scaling
    - Request-based scaling
    - Queue-based scaling
    - Custom scaling policies
    """
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self._current_instances: Dict[str, int] = {}
        self._metrics_history: Dict[str, List[ScalingMetrics]] = {}
    
    def should_scale_up(self, service_name: str, metrics: ScalingMetrics) -> bool:
        """Check if service should scale up"""
        # CPU-based
        if metrics.cpu_usage > self.scale_up_threshold * 100:
            return True
        
        # Memory-based
        if metrics.memory_usage > self.scale_up_threshold * 100:
            return True
        
        # Request-based
        if metrics.request_rate > 1000:  # requests per second
            return True
        
        # Queue-based
        if metrics.queue_size > 100:
            return True
        
        # Response time
        if metrics.response_time > 1000:  # milliseconds
            return True
        
        return False
    
    def should_scale_down(self, service_name: str, metrics: ScalingMetrics) -> bool:
        """Check if service should scale down"""
        current = self._current_instances.get(service_name, self.min_instances)
        
        if current <= self.min_instances:
            return False
        
        # CPU-based
        if metrics.cpu_usage < self.scale_down_threshold * 100:
            return True
        
        # Memory-based
        if metrics.memory_usage < self.scale_down_threshold * 100:
            return True
        
        # Request-based
        if metrics.request_rate < 100:  # requests per second
            return True
        
        # Queue-based
        if metrics.queue_size < 10:
            return True
        
        return False
    
    def calculate_desired_instances(
        self,
        service_name: str,
        metrics: ScalingMetrics
    ) -> int:
        """Calculate desired number of instances"""
        current = self._current_instances.get(service_name, self.min_instances)
        
        if self.should_scale_up(service_name, metrics):
            # Scale up by 50% or add 1, whichever is more
            desired = min(
                self.max_instances,
                max(current + 1, int(current * 1.5))
            )
        elif self.should_scale_down(service_name, metrics):
            # Scale down by 25% or remove 1, whichever is less
            desired = max(
                self.min_instances,
                min(current - 1, int(current * 0.75))
            )
        else:
            desired = current
        
        return desired
    
    def record_metrics(self, service_name: str, metrics: ScalingMetrics) -> None:
        """Record metrics for service"""
        if service_name not in self._metrics_history:
            self._metrics_history[service_name] = []
        
        self._metrics_history[service_name].append(metrics)
        
        # Keep only last 100 metrics
        if len(self._metrics_history[service_name]) > 100:
            self._metrics_history[service_name] = self._metrics_history[service_name][-100:]
    
    def get_scaling_recommendation(
        self,
        service_name: str
    ) -> Dict[str, Any]:
        """Get scaling recommendation"""
        if service_name not in self._metrics_history:
            return {"action": "no_change", "current": self.min_instances}
        
        # Get latest metrics
        latest = self._metrics_history[service_name][-1]
        current = self._current_instances.get(service_name, self.min_instances)
        desired = self.calculate_desired_instances(service_name, latest)
        
        if desired > current:
            action = "scale_up"
        elif desired < current:
            action = "scale_down"
        else:
            action = "no_change"
        
        return {
            "action": action,
            "current": current,
            "desired": desired,
            "metrics": {
                "cpu": latest.cpu_usage,
                "memory": latest.memory_usage,
                "request_rate": latest.request_rate
            }
        }


# Global auto-scaler
_auto_scaler: Optional[AutoScaler] = None


def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler"""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler















