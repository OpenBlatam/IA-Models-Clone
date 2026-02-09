"""
Advanced Observability
======================

Distributed tracing, metrics, and logging for microservices.
"""

import logging
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import os

logger = logging.getLogger(__name__)


class DistributedTracer:
    """Distributed tracing with OpenTelemetry."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._tracer = None
        self._initialized = False
    
    def initialize(self):
        """Initialize OpenTelemetry."""
        if self._initialized:
            return
        
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource
            
            resource = Resource.create({
                "service.name": self.service_name,
            })
            
            trace.set_tracer_provider(TracerProvider(resource=resource))
            
            otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            processor = BatchSpanProcessor(exporter)
            trace.get_tracer_provider().add_span_processor(processor)
            
            self._tracer = trace.get_tracer(self.service_name)
            self._initialized = True
            logger.info(f"Distributed tracing initialized for {self.service_name}")
        except ImportError:
            logger.warning("OpenTelemetry not installed, tracing disabled")
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
    
    @asynccontextmanager
    async def trace(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Trace an operation."""
        if not self._initialized:
            self.initialize()
        
        if not self._tracer:
            yield None
            return
        
        with self._tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span


class MetricsCollector:
    """Metrics collection for Prometheus."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._metrics = {}
        self._initialized = False
    
    def initialize(self):
        """Initialize Prometheus metrics."""
        if self._initialized:
            return
        
        try:
            from prometheus_client import Counter, Histogram, Gauge
            
            self._metrics = {
                "requests_total": Counter(
                    f"{self.service_name}_requests_total",
                    "Total requests",
                    ["method", "endpoint", "status"]
                ),
                "request_duration": Histogram(
                    f"{self.service_name}_request_duration_seconds",
                    "Request duration",
                    ["method", "endpoint"]
                ),
                "active_requests": Gauge(
                    f"{self.service_name}_active_requests",
                    "Active requests"
                ),
            }
            
            self._initialized = True
            logger.info(f"Metrics initialized for {self.service_name}")
        except ImportError:
            logger.warning("Prometheus client not installed, metrics disabled")
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metric."""
        if not self._initialized:
            self.initialize()
        
        if "requests_total" in self._metrics:
            self._metrics["requests_total"].labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()
        
        if "request_duration" in self._metrics:
            self._metrics["request_duration"].labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)


class StructuredLogger:
    """Structured logging with context."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._logger = logging.getLogger(service_name)
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """Log request with structured data."""
        self._logger.info(
            "Request completed",
            extra={
                "service": self.service_name,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration": duration,
                "request_id": request_id,
                **kwargs
            }
        )
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log error with context."""
        self._logger.error(
            "Error occurred",
            extra={
                "service": self.service_name,
                "error": str(error),
                "error_type": type(error).__name__,
                **(context or {})
            },
            exc_info=True
        )


class ObservabilityManager:
    """Manages all observability components."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = DistributedTracer(service_name)
        self.metrics = MetricsCollector(service_name)
        self.logger = StructuredLogger(service_name)
    
    def initialize(self):
        """Initialize all observability components."""
        self.tracer.initialize()
        self.metrics.initialize()
    
    @asynccontextmanager
    async def observe(self, operation_name: str, method: str = "", endpoint: str = ""):
        """Observe an operation (trace + metrics + logging)."""
        start_time = time.time()
        request_id = f"req-{int(start_time * 1000)}"
        
        async with self.tracer.trace(operation_name, {
            "method": method,
            "endpoint": endpoint,
            "request_id": request_id
        }):
            try:
                yield request_id
                duration = time.time() - start_time
                self.metrics.record_request(method, endpoint, 200, duration)
                self.logger.log_request(method, endpoint, 200, duration, request_id)
            except Exception as e:
                duration = time.time() - start_time
                self.metrics.record_request(method, endpoint, 500, duration)
                self.logger.log_error(e, {
                    "method": method,
                    "endpoint": endpoint,
                    "request_id": request_id
                })
                raise


# Global observability managers
_observability_managers: Dict[str, ObservabilityManager] = {}


def get_observability_manager(service_name: str) -> ObservabilityManager:
    """Get observability manager for service."""
    if service_name not in _observability_managers:
        _observability_managers[service_name] = ObservabilityManager(service_name)
        _observability_managers[service_name].initialize()
    return _observability_managers[service_name]















