"""
Observability - OpenTelemetry and Prometheus Metrics
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)

# OpenTelemetry (optional)
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
    
    # Define metrics
    webhook_deliveries_total = Counter(
        'webhook_deliveries_total',
        'Total webhook delivery attempts',
        ['status', 'event_type']
    )
    
    webhook_delivery_duration = Histogram(
        'webhook_delivery_duration_seconds',
        'Webhook delivery duration in seconds',
        ['event_type'],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    )
    
    webhook_queue_size = Gauge(
        'webhook_queue_size',
        'Current webhook queue size'
    )
    
    webhook_circuit_breaker_state = Gauge(
        'webhook_circuit_breaker_state',
        'Circuit breaker state (0=closed, 1=open, 2=half_open)',
        ['endpoint_id']
    )
    
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Dummy metrics for when Prometheus is not available
    class DummyMetric:
        def labels(self, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
    
    webhook_deliveries_total = DummyMetric()
    webhook_delivery_duration = DummyMetric()
    webhook_queue_size = DummyMetric()
    webhook_circuit_breaker_state = DummyMetric()


class ObservabilityManager:
    """Manager for observability features"""
    
    def __init__(self, enable_tracing: bool = True, enable_metrics: bool = True):
        """
        Initialize observability manager
        
        Args:
            enable_tracing: Enable OpenTelemetry tracing
            enable_metrics: Enable Prometheus metrics
        """
        self.enable_tracing = enable_tracing and OPENTELEMETRY_AVAILABLE
        self.enable_metrics = enable_metrics and PROMETHEUS_AVAILABLE
        self._tracer = None
        
        if self.enable_tracing:
            self._setup_tracing()
    
    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing"""
        try:
            resource = Resource.create({"service.name": "content-redundancy-detector"})
            provider = TracerProvider(resource=resource)
            
            # Add OTLP exporter if endpoint is configured
            import os
            otlp_endpoint = os.getenv("OTLP_ENDPOINT")
            if otlp_endpoint:
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(__name__)
            logger.info("OpenTelemetry tracing initialized")
        except Exception as e:
            logger.warning(f"Failed to setup tracing: {e}")
            self.enable_tracing = False
    
    def get_tracer(self):
        """Get OpenTelemetry tracer"""
        return self._tracer
    
    def trace_operation(self, operation_name: str):
        """Decorator for tracing operations"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enable_tracing or not self._tracer:
                    return await func(*args, **kwargs)
                
                with self._tracer.start_as_current_span(operation_name) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enable_tracing or not self._tracer:
                    return func(*args, **kwargs)
                
                with self._tracer.start_as_current_span(operation_name) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
            
            # Check if function is async
            try:
                import asyncio
                if asyncio.iscoroutinefunction(func):
                    return async_wrapper
            except (ImportError, AttributeError):
                pass
            
            return sync_wrapper
        
        return decorator
    
    def record_webhook_delivery(
        self,
        status: str,
        event_type: str,
        duration: Optional[float] = None
    ) -> None:
        """
        Record webhook delivery metric
        
        Args:
            status: Delivery status (success, failed, timeout)
            event_type: Type of webhook event
            duration: Delivery duration in seconds
        """
        if self.enable_metrics:
            try:
                webhook_deliveries_total.labels(
                    status=status,
                    event_type=event_type
                ).inc()
                
                if duration is not None:
                    webhook_delivery_duration.labels(
                        event_type=event_type
                    ).observe(duration)
            except Exception as e:
                logger.warning(f"Failed to record metrics: {e}")
    
    def update_queue_size(self, size: int) -> None:
        """Update queue size metric"""
        if self.enable_metrics:
            try:
                webhook_queue_size.set(size)
            except Exception as e:
                logger.warning(f"Failed to update queue size metric: {e}")
    
    def update_circuit_breaker_state(self, endpoint_id: str, state: str) -> None:
        """
        Update circuit breaker state metric
        
        Args:
            endpoint_id: Endpoint identifier
            state: Circuit breaker state (closed=0, open=1, half_open=2)
        """
        if self.enable_metrics:
            try:
                state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)
                webhook_circuit_breaker_state.labels(
                    endpoint_id=endpoint_id
                ).set(state_value)
            except Exception as e:
                logger.warning(f"Failed to update circuit breaker metric: {e}")


# Global observability manager
observability_manager = ObservabilityManager()

