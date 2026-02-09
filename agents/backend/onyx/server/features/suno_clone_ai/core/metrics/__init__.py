"""
Advanced Metrics Module

Provides:
- Advanced training metrics
- Real-time metrics tracking
- Metrics aggregation
- Custom metrics
"""

from .tracker import (
    MetricsTracker,
    track_metric,
    aggregate_metrics,
    compute_average_metrics
)

from .custom_metrics import (
    create_custom_metric,
    register_metric,
    get_metric
)

__all__ = [
    # Metrics tracking
    "MetricsTracker",
    "track_metric",
    "aggregate_metrics",
    "compute_average_metrics",
    # Custom metrics
    "create_custom_metric",
    "register_metric",
    "get_metric"
]



