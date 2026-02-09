"""
Observability Module

Provides:
- Advanced observability
- Distributed tracing
- Metrics collection
- Logging aggregation
"""

from .tracer import (
    Tracer,
    trace_function,
    create_span,
    get_trace_context
)

from .metrics_collector import (
    MetricsCollector,
    collect_metric,
    get_metrics_summary
)

__all__ = [
    # Tracing
    "Tracer",
    "trace_function",
    "create_span",
    "get_trace_context",
    # Metrics
    "MetricsCollector",
    "collect_metric",
    "get_metrics_summary"
]



