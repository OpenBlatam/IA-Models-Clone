"""
Observability Service Implementations
Provides implementations for metrics and tracing
"""

import logging
from typing import Dict, Any, Optional

from core.interfaces import IMetricsService, ITracingService, IServiceFactory
from config.aws_settings import get_aws_settings

logger = logging.getLogger(__name__)


class CloudWatchMetricsService(IMetricsService):
    """CloudWatch implementation of metrics service"""
    
    def __init__(self):
        from aws.aws_services import CloudWatchService
        self.service = CloudWatchService()
        self.settings = get_aws_settings()
    
    async def record(self, metric_name: str, value: float, **kwargs) -> None:
        """Record metric in CloudWatch"""
        unit = kwargs.get("unit", "Count")
        dimensions = kwargs.get("dimensions", {})
        self.service.put_metric(metric_name, value, unit, dimensions)
    
    async def increment(self, metric_name: str, **kwargs) -> None:
        """Increment counter in CloudWatch"""
        await self.record(metric_name, 1.0, **kwargs)


class PrometheusMetricsService(IMetricsService):
    """Prometheus implementation of metrics service"""
    
    def __init__(self):
        try:
            from prometheus_client import Counter, Histogram, Gauge
            self.Counter = Counter
            self.Histogram = Histogram
            self.Gauge = Gauge
            self._counters: Dict[str, Counter] = {}
            self._histograms: Dict[str, Histogram] = {}
            self._gauges: Dict[str, Gauge] = {}
        except ImportError:
            logger.warning("Prometheus client not available")
            self.Counter = self.Histogram = self.Gauge = None
    
    async def record(self, metric_name: str, value: float, **kwargs) -> None:
        """Record metric in Prometheus"""
        if not self.Histogram:
            return
        
        if metric_name not in self._histograms:
            self._histograms[metric_name] = self.Histogram(
                metric_name,
                f"Metric: {metric_name}",
                list(kwargs.get("labels", {}).keys())
            )
        
        labels = list(kwargs.get("labels", {}).values())
        self._histograms[metric_name].observe(value, *labels)
    
    async def increment(self, metric_name: str, **kwargs) -> None:
        """Increment counter in Prometheus"""
        if not self.Counter:
            return
        
        if metric_name not in self._counters:
            self._counters[metric_name] = self.Counter(
                metric_name,
                f"Counter: {metric_name}",
                list(kwargs.get("labels", {}).keys())
            )
        
        labels = list(kwargs.get("labels", {}).values())
        self._counters[metric_name].inc(*labels)


class OpenTelemetryTracingService(ITracingService):
    """OpenTelemetry implementation of tracing service"""
    
    def __init__(self):
        try:
            from opentelemetry import trace
            self.tracer = trace.get_tracer(__name__)
            self._current_span = None
        except ImportError:
            logger.warning("OpenTelemetry not available")
            self.tracer = None
            self._current_span = None
    
    def start_span(self, name: str, **kwargs):
        """Start a new span"""
        if not self.tracer:
            return self._NullSpan()
        
        span = self.tracer.start_span(name, **kwargs)
        self._current_span = span
        return span
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute"""
        if self._current_span:
            self._current_span.set_attribute(key, value)
    
    def record_exception(self, exception: Exception) -> None:
        """Record exception in span"""
        if self._current_span:
            self._current_span.record_exception(exception)
    
    class _NullSpan:
        """Null object for when tracing is not available"""
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def set_attribute(self, *args, **kwargs):
            pass
        
        def record_exception(self, *args, **kwargs):
            pass


class ObservabilityServiceFactory(IServiceFactory):
    """Factory for creating observability services"""
    
    @staticmethod
    def create_metrics_service(backend: str = "cloudwatch") -> IMetricsService:
        """Create metrics service"""
        if backend == "cloudwatch":
            return CloudWatchMetricsService()
        elif backend == "prometheus":
            return PrometheusMetricsService()
        else:
            raise ValueError(f"Unsupported metrics backend: {backend}")
    
    @staticmethod
    def create_tracing_service(backend: str = "opentelemetry") -> ITracingService:
        """Create tracing service"""
        if backend == "opentelemetry":
            return OpenTelemetryTracingService()
        else:
            raise ValueError(f"Unsupported tracing backend: {backend}")
    
    def create_metrics_service(self) -> IMetricsService:
        """Create metrics service (factory method)"""
        settings = get_aws_settings()
        backend = "cloudwatch" if settings.is_lambda else "prometheus"
        return self.create_metrics_service(backend)
    
    def create_tracing_service(self) -> ITracingService:
        """Create tracing service (factory method)"""
        return self.create_tracing_service()
    
    def create_storage_service(self):
        """Not implemented in observability factory"""
        raise NotImplementedError
    
    def create_cache_service(self):
        """Not implemented in observability factory"""
        raise NotImplementedError
    
    def create_file_storage_service(self):
        """Not implemented in observability factory"""
        raise NotImplementedError
    
    def create_message_queue_service(self):
        """Not implemented in observability factory"""
        raise NotImplementedError
    
    def create_notification_service(self):
        """Not implemented in observability factory"""
        raise NotImplementedError
    
    def create_authentication_service(self):
        """Not implemented in observability factory"""
        raise NotImplementedError















