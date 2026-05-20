"""
OpenTelemetry Integration for Distributed Tracing
Structured logging with JSON format for better observability
Integrates with Prometheus, Grafana, and centralized logging systems
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)


class StructuredLogger:
    """
    Structured JSON logger for better log analysis
    Optimized for ELK Stack, CloudWatch, and centralized logging
    """
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self.logger.debug(message, **kwargs)


# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    # Request metrics
    http_requests_total = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status_code']
    )
    
    http_request_duration = Histogram(
        'http_request_duration_seconds',
        'HTTP request duration in seconds',
        ['method', 'endpoint']
    )
    
    # Business metrics
    analysis_requests_total = Counter(
        'analysis_requests_total',
        'Total analysis requests',
        ['type']
    )
    
    analysis_duration = Histogram(
        'analysis_duration_seconds',
        'Analysis processing duration',
        ['type']
    )
    
    # System metrics
    active_connections = Gauge(
        'active_connections',
        'Number of active connections'
    )
    
    cache_hits = Counter(
        'cache_hits_total',
        'Total cache hits',
        ['cache_type']
    )
    
    cache_misses = Counter(
        'cache_misses_total',
        'Total cache misses',
        ['cache_type']
    )
    
    # Circuit breaker metrics
    circuit_breaker_state = Gauge(
        'circuit_breaker_state',
        'Circuit breaker state (0=closed, 1=half_open, 2=open)',
        ['breaker_name']
    )
    
    circuit_breaker_failures = Counter(
        'circuit_breaker_failures_total',
        'Total circuit breaker failures',
        ['breaker_name']
    )
else:
    # Dummy metrics if Prometheus not available
    http_requests_total = None
    http_request_duration = None
    analysis_requests_total = None
    analysis_duration = None
    active_connections = None
    cache_hits = None
    cache_misses = None
    circuit_breaker_state = None
    circuit_breaker_failures = None


class TracingContext:
    """
    Distributed tracing context manager
    Creates spans for request tracing across microservices
    """
    
    def __init__(self, tracer_name: str = "content_redundancy_detector"):
        self.tracer_name = tracer_name
        self.tracer = None
        self._initialized = False
    
    def initialize(self, service_name: str, service_version: str = "1.0.0"):
        """Initialize OpenTelemetry tracing"""
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        try:
            resource = Resource.create({
                "service.name": service_name,
                "service.version": service_version,
            })
            
            trace.set_tracer_provider(TracerProvider(resource=resource))
            
            # Configure OTLP exporter (for Jaeger, Tempo, etc.)
            otlp_exporter = OTLPSpanExporter(
                endpoint="localhost:4317",  # OTLP gRPC endpoint
                insecure=True
            )
            
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = trace.get_tracer(service_name)
            self._initialized = True
            
            logger = logging.getLogger(__name__)
            logger.info(f"OpenTelemetry tracing initialized for {service_name}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to initialize OpenTelemetry: {e}")
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a tracing span"""
        if not self._initialized or not self.tracer:
            # No-op if not initialized
            yield
            return
        
        span = self.tracer.start_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            span.end()


# Global tracing context
tracing_context = TracingContext()


def setup_observability(service_name: str, service_version: str = "1.0.0"):
    """
    Setup complete observability stack:
    - OpenTelemetry distributed tracing
    - Prometheus metrics
    - Structured logging
    """
    # Initialize tracing
    tracing_context.initialize(service_name, service_version)
    
    # Instrument FastAPI if available
    if OPENTELEMETRY_AVAILABLE:
        try:
            FastAPIInstrumentor().instrument()
            HTTPXClientInstrumentor().instrument()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to instrument FastAPI: {e}")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Observability stack initialized for {service_name}")


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)


@contextmanager
def track_request_metrics(method: str, endpoint: str):
    """Track request metrics with Prometheus"""
    start_time = time.time()
    
    try:
        yield
        status = 200
    except Exception as e:
        status = getattr(e, 'status_code', 500)
        raise
    finally:
        duration = time.time() - start_time
        
        if PROMETHEUS_AVAILABLE:
            if http_requests_total:
                http_requests_total.labels(method=method, endpoint=endpoint, status_code=status).inc()
            if http_request_duration:
                http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)


def track_analysis_metrics(analysis_type: str):
    """Decorator to track analysis metrics"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                
                if PROMETHEUS_AVAILABLE:
                    if analysis_requests_total:
                        analysis_requests_total.labels(type=analysis_type).inc()
                    if analysis_duration:
                        analysis_duration.labels(type=analysis_type).observe(duration)
        
        return wrapper
    return decorator






