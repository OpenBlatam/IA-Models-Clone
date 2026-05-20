"""
Observability setup for OpenTelemetry, Prometheus, and structured logging
"""

import os
import logging

logger = logging.getLogger(__name__)


def setup_observability():
    """
    Setup observability stack:
    - OpenTelemetry for distributed tracing
    - Prometheus for metrics (handled by middleware)
    - Structured logging (handled by logger setup)
    """
    try:
        # Setup OpenTelemetry if enabled
        if os.getenv("OTEL_ENABLED", "false").lower() == "true":
            _setup_opentelemetry()
        
        logger.info("✅ Observability stack initialized")
    except Exception as e:
        logger.warning(f"Observability setup failed: {e}")


def _setup_opentelemetry():
    """Setup OpenTelemetry tracing"""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        
        # Create resource
        resource = Resource.create({
            "service.name": os.getenv("SERVICE_NAME", "dermatology-ai"),
            "service.version": os.getenv("SERVICE_VERSION", "6.0.0"),
            "deployment.environment": os.getenv("ENVIRONMENT", "production"),
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true",
        )
        
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(provider)
        
        logger.info(f"✅ OpenTelemetry configured (endpoint: {otlp_endpoint})")
        
    except ImportError:
        logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
    except Exception as e:
        logger.warning(f"OpenTelemetry setup failed: {e}")


def get_health_metrics() -> dict:
    """Get health metrics for monitoring"""
    import psutil
    import time
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "uptime_seconds": time.time() - process.create_time(),
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
        }
    except Exception as e:
        logger.warning(f"Failed to get health metrics: {e}")
        return {}















