from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any
from ...core.interfaces.metrics_interface import IMetricsService
from ...core.entities.metrics import MetricsData
import logging
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from typing import Any, List, Dict, Optional
import asyncio
"""
Prometheus Metrics Implementation
=================================

Concrete implementation of metrics service using Prometheus.
"""


logger = logging.getLogger(__name__)

try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available, metrics will be limited")


class PrometheusMetricsService(IMetricsService):
    """Prometheus-based metrics service implementation."""
    
    def __init__(self) -> Any:
        self.metrics_data = MetricsData.create_empty()
        
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        else:
            self.registry = None
    
    def _init_prometheus_metrics(self) -> Any:
        """Initialize Prometheus metrics."""
        self.request_count = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'api_active_connections',
            'Active connections',
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'cache_operations_total',
            'Cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['service'],
            registry=self.registry
        )
        
        self.error_count = Counter(
            'api_errors_total',
            'Total API errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record a request metric."""
        is_error = status >= 400
        self.metrics_data.add_request(duration, is_error)
        
        if PROMETHEUS_AVAILABLE:
            self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_error(self, error_type: str, endpoint: str):
        """Record an error metric."""
        if PROMETHEUS_AVAILABLE:
            self.error_count.labels(error_type=error_type, endpoint=endpoint).inc()
    
    def record_cache_operation(self, operation: str, result: str):
        """Record a cache operation metric."""
        if PROMETHEUS_AVAILABLE:
            self.cache_operations.labels(operation=operation, result=result).inc()
    
    def update_active_connections(self, count: int):
        """Update active connections count."""
        self.metrics_data.active_connections = count
        
        if PROMETHEUS_AVAILABLE:
            self.active_connections.set(count)
    
    def update_circuit_breaker_state(self, service: str, state: str):
        """Update circuit breaker state."""
        self.metrics_data.update_circuit_breaker_state(service, state)
        
        if PROMETHEUS_AVAILABLE:
            state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)
            self.circuit_breaker_state.labels(service=service).set(state_value)
    
    def get_metrics_data(self) -> MetricsData:
        """Get current metrics data."""
        return self.metrics_data
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE and self.registry:
            return generate_latest(self.registry).decode()
        return "# Prometheus not available\n"
    
    def add_custom_metric(self, name: str, value: Any, labels: Dict[str, str] = None):
        """Add a custom metric."""
        self.metrics_data.custom_metrics[name] = {
            "value": value,
            "labels": labels or {},
            "timestamp": self.metrics_data.timestamp.isoformat()
        } 