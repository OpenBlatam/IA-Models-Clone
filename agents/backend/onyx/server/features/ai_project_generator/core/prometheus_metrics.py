"""
Prometheus Metrics - Métricas Prometheus para monitoreo
======================================================

Métricas Prometheus para monitoreo de:
- Request rates y latencias
- Error rates
- Resource usage
- Business metrics
"""

import time
import logging
from typing import Optional, Dict, Any
from functools import wraps
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("prometheus_client not available. Install with: pip install prometheus-client")

from .microservices_config import get_microservices_config

if PROMETHEUS_AVAILABLE:
    logger = logging.getLogger(__name__)
    config = get_microservices_config()
    
    # Métricas HTTP
    http_requests_total = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"]
    )
    
    http_request_duration_seconds = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0)
    )
    
    http_request_size_bytes = Histogram(
        "http_request_size_bytes",
        "HTTP request size in bytes",
        ["method", "endpoint"],
        buckets=(100, 1000, 10000, 100000, 1000000)
    )
    
    http_response_size_bytes = Histogram(
        "http_response_size_bytes",
        "HTTP response size in bytes",
        ["method", "endpoint"],
        buckets=(100, 1000, 10000, 100000, 1000000)
    )
    
    # Métricas de proyectos
    projects_generated_total = Counter(
        "projects_generated_total",
        "Total projects generated",
        ["status", "ai_type", "framework"]
    )
    
    project_generation_duration_seconds = Histogram(
        "project_generation_duration_seconds",
        "Project generation duration in seconds",
        ["ai_type", "framework"],
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
    )
    
    projects_in_queue = Gauge(
        "projects_in_queue",
        "Number of projects in queue"
    )
    
    # Métricas de cache
    cache_hits_total = Counter(
        "cache_hits_total",
        "Total cache hits",
        ["cache_type"]
    )
    
    cache_misses_total = Counter(
        "cache_misses_total",
        "Total cache misses",
        ["cache_type"]
    )
    
    # Métricas de recursos
    cpu_usage_percent = Gauge(
        "cpu_usage_percent",
        "CPU usage percentage"
    )
    
    memory_usage_bytes = Gauge(
        "memory_usage_bytes",
        "Memory usage in bytes"
    )
    
    disk_usage_bytes = Gauge(
        "disk_usage_bytes",
        "Disk usage in bytes"
    )
    
    # Métricas de workers
    worker_tasks_total = Counter(
        "worker_tasks_total",
        "Total worker tasks",
        ["task_type", "status"]
    )
    
    worker_task_duration_seconds = Histogram(
        "worker_task_duration_seconds",
        "Worker task duration in seconds",
        ["task_type"],
        buckets=(0.1, 1.0, 5.0, 10.0, 30.0, 60.0)
    )
    
    # Métricas de circuit breakers
    circuit_breaker_state = Gauge(
        "circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half_open)",
        ["circuit_breaker_name"]
    )
    
    circuit_breaker_failures_total = Counter(
        "circuit_breaker_failures_total",
        "Total circuit breaker failures",
        ["circuit_breaker_name"]
    )
    
    # Métricas de retry
    retry_attempts_total = Counter(
        "retry_attempts_total",
        "Total retry attempts",
        ["function_name", "status"]
    )


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware para recopilar métricas Prometheus"""
    
    async def dispatch(self, request: Request, call_next):
        if not PROMETHEUS_AVAILABLE:
            return await call_next(request)
        
        # Obtener endpoint (sin parámetros)
        endpoint = request.url.path
        
        # Medir tiempo
        start_time = time.time()
        
        # Medir tamaño de request
        request_size = 0
        if hasattr(request, "body"):
            try:
                body = await request.body()
                request_size = len(body)
            except Exception:
                pass
        
        # Procesar request
        response = await call_next(request)
        
        # Calcular duración
        duration = time.time() - start_time
        
        # Medir tamaño de response
        response_size = 0
        if hasattr(response, "body"):
            try:
                response_size = len(response.body) if hasattr(response.body, "__len__") else 0
            except Exception:
                pass
        
        # Registrar métricas
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code
        ).inc()
        
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration)
        
        if request_size > 0:
            http_request_size_bytes.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(request_size)
        
        if response_size > 0:
            http_response_size_bytes.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(response_size)
        
        return response


def record_project_generation(
    status: str,
    ai_type: str = "unknown",
    framework: str = "unknown",
    duration: Optional[float] = None
):
    """Registra métricas de generación de proyecto"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    projects_generated_total.labels(
        status=status,
        ai_type=ai_type,
        framework=framework
    ).inc()
    
    if duration is not None:
        project_generation_duration_seconds.labels(
            ai_type=ai_type,
            framework=framework
        ).observe(duration)


def record_cache_operation(cache_type: str, hit: bool):
    """Registra operación de cache"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    if hit:
        cache_hits_total.labels(cache_type=cache_type).inc()
    else:
        cache_misses_total.labels(cache_type=cache_type).inc()


def record_worker_task(
    task_type: str,
    status: str,
    duration: Optional[float] = None
):
    """Registra métricas de worker task"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    worker_tasks_total.labels(
        task_type=task_type,
        status=status
    ).inc()
    
    if duration is not None:
        worker_task_duration_seconds.labels(
            task_type=task_type
        ).observe(duration)


def record_circuit_breaker_state(name: str, state: str):
    """Registra estado de circuit breaker"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)
    circuit_breaker_state.labels(circuit_breaker_name=name).set(state_value)


def record_retry_attempt(function_name: str, success: bool):
    """Registra intento de retry"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    status = "success" if success else "failure"
    retry_attempts_total.labels(
        function_name=function_name,
        status=status
    ).inc()


def update_queue_size(size: int):
    """Actualiza tamaño de cola"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    projects_in_queue.set(size)


def update_resource_metrics(cpu: float, memory: int, disk: int):
    """Actualiza métricas de recursos"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    cpu_usage_percent.set(cpu)
    memory_usage_bytes.set(memory)
    disk_usage_bytes.set(disk)


def get_metrics() -> bytes:
    """Obtiene métricas en formato Prometheus"""
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not available\n"
    
    return generate_latest(REGISTRY)


def get_metrics_content_type() -> str:
    """Obtiene content type para métricas"""
    return CONTENT_TYPE_LATEST















