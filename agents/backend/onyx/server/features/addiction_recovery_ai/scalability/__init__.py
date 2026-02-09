"""
Scalability Module
Auto-scaling, throttling, and resource management
"""

from .auto_scaling import (
    AutoScaler,
    ScalingMetrics,
    ScalingPolicy,
    get_auto_scaler
)
from .throttling import (
    IntelligentThrottler,
    ThrottleConfig,
    get_throttler
)
from .resource_manager import (
    ResourceManager,
    get_resource_manager
)

__all__ = [
    "AutoScaler",
    "ScalingMetrics",
    "ScalingPolicy",
    "get_auto_scaler",
    "IntelligentThrottler",
    "ThrottleConfig",
    "get_throttler",
    "ResourceManager",
    "get_resource_manager"
]















