"""
Observability - OpenTelemetry y Prometheus
==========================================

Sistema completo de observabilidad con:
- OpenTelemetry para distributed tracing
- Prometheus para métricas
- Structured logging
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Prometheus metrics
_prometheus_client: Optional[Any] = None
_metrics_registry: Optional[Any] = None

# OpenTelemetry
_otel_tracer: Optional[Any] = None
_otel_meter: Optional[Any] = None


def setup_prometheus() -> Optional[Any]:
    """
    Configurar Prometheus metrics.
    
    Returns:
        Prometheus client registry o None si no está disponible.
    """
    global _prometheus_client, _metrics_registry
    
    if _metrics_registry is not None:
        return _metrics_registry
    
    try:
        from prometheus_client import Counter, Histogram, Gauge, REGISTRY, generate_latest
        from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
        
        # Métricas HTTP
        http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)
        )
        
        http_request_size_bytes = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint']
        )
        
        # Métricas del agente
        agent_tasks_total = Counter(
            'agent_tasks_total',
            'Total tasks processed',
            ['status']
        )
        
        agent_tasks_duration_seconds = Histogram(
            'agent_tasks_duration_seconds',
            'Task processing duration in seconds',
            ['status']
        )
        
        agent_active_tasks = Gauge(
            'agent_active_tasks',
            'Number of active tasks'
        )
        
        agent_queue_size = Gauge(
            'agent_queue_size',
            'Number of tasks in queue'
        )
        
        # Métricas de sistema
        system_memory_bytes = Gauge(
            'system_memory_bytes',
            'System memory usage in bytes',
            ['type']  # used, free, total
        )
        
        system_cpu_percent = Gauge(
            'system_cpu_percent',
            'System CPU usage percentage'
        )
        
        _prometheus_client = {
            'http_requests_total': http_requests_total,
            'http_request_duration_seconds': http_request_duration_seconds,
            'http_request_size_bytes': http_request_size_bytes,
            'agent_tasks_total': agent_tasks_total,
            'agent_tasks_duration_seconds': agent_tasks_duration_seconds,
            'agent_active_tasks': agent_active_tasks,
            'agent_queue_size': agent_queue_size,
            'system_memory_bytes': system_memory_bytes,
            'system_cpu_percent': system_cpu_percent,
            'generate_latest': generate_latest,
            'CONTENT_TYPE_LATEST': CONTENT_TYPE_LATEST
        }
        
        _metrics_registry = REGISTRY
        logger.info("Prometheus metrics configured")
        
        return _metrics_registry
    
    except ImportError:
        logger.warning("prometheus-client not installed, metrics disabled")
        return None


def setup_opentelemetry(
    service_name: str = "cursor-agent-24-7",
    service_version: str = "1.0.0",
    endpoint: Optional[str] = None
) -> Optional[Any]:
    """
    Configurar OpenTelemetry para distributed tracing.
    
    Args:
        service_name: Nombre del servicio.
        service_version: Versión del servicio.
        endpoint: Endpoint del collector (opcional).
    
    Returns:
        Tracer o None si no está disponible.
    """
    global _otel_tracer, _otel_meter
    
    if _otel_tracer is not None:
        return _otel_tracer
    
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        
        # Resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": service_version,
        })
        
        # Tracer Provider
        trace_provider = TracerProvider(resource=resource)
        
        # Exporters
        exporters = []
        
        # Console exporter (para desarrollo)
        if os.getenv("OTEL_LOG_EXPORTER", "false").lower() == "true":
            exporters.append(ConsoleSpanExporter())
        
        # OTLP exporter (para collector)
        if endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
                exporters.append(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
                )
            except ImportError:
                logger.warning("OTLP exporter not available")
        
        # Agregar exporters
        for exporter in exporters:
            trace_provider.add_span_processor(BatchSpanProcessor(exporter))
        
        trace.set_tracer_provider(trace_provider)
        _otel_tracer = trace.get_tracer(service_name, service_version)
        
        # Meter Provider
        metrics_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(metrics_provider)
        _otel_meter = metrics.get_meter(service_name, service_version)
        
        logger.info(f"OpenTelemetry configured for {service_name}")
        
        return _otel_tracer
    
    except ImportError:
        logger.warning("OpenTelemetry not installed, tracing disabled")
        return None


def get_tracer() -> Optional[Any]:
    """Obtener tracer de OpenTelemetry."""
    return _otel_tracer


def get_meter() -> Optional[Any]:
    """Obtener meter de OpenTelemetry."""
    return _otel_meter


def get_prometheus_metrics() -> Optional[Dict[str, Any]]:
    """Obtener métricas de Prometheus."""
    return _prometheus_client


def record_http_metric(method: str, endpoint: str, status: int, duration: float, size: int = 0) -> None:
    """Registrar métrica HTTP."""
    if not _prometheus_client:
        return
    
    try:
        _prometheus_client['http_requests_total'].labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        _prometheus_client['http_request_duration_seconds'].labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        if size > 0:
            _prometheus_client['http_request_size_bytes'].labels(
                method=method,
                endpoint=endpoint
            ).observe(size)
    except Exception as e:
        logger.warning(f"Error recording HTTP metric: {e}")


def record_task_metric(status: str, duration: float) -> None:
    """Registrar métrica de tarea."""
    if not _prometheus_client:
        return
    
    try:
        _prometheus_client['agent_tasks_total'].labels(status=status).inc()
        _prometheus_client['agent_tasks_duration_seconds'].labels(status=status).observe(duration)
    except Exception as e:
        logger.warning(f"Error recording task metric: {e}")


def update_active_tasks(count: int) -> None:
    """Actualizar contador de tareas activas."""
    if not _prometheus_client:
        return
    
    try:
        _prometheus_client['agent_active_tasks'].set(count)
    except Exception as e:
        logger.warning(f"Error updating active tasks: {e}")


def update_queue_size(size: int) -> None:
    """Actualizar tamaño de cola."""
    if not _prometheus_client:
        return
    
    try:
        _prometheus_client['agent_queue_size'].set(size)
    except Exception as e:
        logger.warning(f"Error updating queue size: {e}")


@asynccontextmanager
async def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager para crear spans de OpenTelemetry."""
    if not _otel_tracer:
        yield None
        return
    
    span = _otel_tracer.start_span(name)
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
    
    try:
        yield span
    finally:
        span.end()




