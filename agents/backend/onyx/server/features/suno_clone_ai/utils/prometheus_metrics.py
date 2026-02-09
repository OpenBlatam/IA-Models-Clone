"""
Métricas Prometheus para monitoreo avanzado
"""

import logging
import time
from typing import Optional, Dict, Any
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Métricas HTTP
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000)
)

# Métricas de generación de música
music_generation_requests = Counter(
    'music_generation_requests_total',
    'Total music generation requests',
    ['status', 'source', 'genre']
)

music_generation_duration = Histogram(
    'music_generation_duration_seconds',
    'Music generation duration in seconds',
    ['source', 'genre'],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

music_generation_queue_size = Gauge(
    'music_generation_queue_size',
    'Current size of music generation queue'
)

music_generation_active = Gauge(
    'music_generation_active',
    'Number of active music generations'
)

# Métricas de caché
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

cache_size = Gauge(
    'cache_size_bytes',
    'Current cache size in bytes',
    ['cache_type']
)

# Métricas de sistema
active_websocket_connections = Gauge(
    'websocket_connections_active',
    'Number of active WebSocket connections'
)

database_connection_pool_size = Gauge(
    'database_connection_pool_size',
    'Database connection pool size'
)

database_query_duration = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Métricas de errores
error_total = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'endpoint']
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware para registrar métricas Prometheus"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        method = request.method
        endpoint = request.url.path
        
        # Obtener tamaño de request si es posible
        request_size = 0
        if hasattr(request, '_body'):
            request_size = len(request._body) if request._body else 0
        
        # Procesar request
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Registrar métricas
            duration = time.time() - start_time
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            if request_size > 0:
                http_request_size_bytes.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(request_size)
            
            # Obtener tamaño de respuesta si es posible
            if hasattr(response, 'body'):
                response_size = len(response.body) if hasattr(response.body, '__len__') else 0
                if response_size > 0:
                    http_response_size_bytes.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(response_size)
            
            return response
            
        except Exception as e:
            status_code = 500
            error_total.labels(
                error_type=type(e).__name__,
                endpoint=endpoint
            ).inc()
            
            duration = time.time() - start_time
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            raise


def record_music_generation(status: str, source: str = "direct", genre: Optional[str] = None, duration: Optional[float] = None):
    """Registra una generación de música"""
    music_generation_requests.labels(
        status=status,
        source=source,
        genre=genre or "unknown"
    ).inc()
    
    if duration is not None:
        music_generation_duration.labels(
            source=source,
            genre=genre or "unknown"
        ).observe(duration)


def record_cache_hit(cache_type: str):
    """Registra un cache hit"""
    cache_hits.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str):
    """Registra un cache miss"""
    cache_misses.labels(cache_type=cache_type).inc()


def update_cache_size(cache_type: str, size_bytes: int):
    """Actualiza el tamaño del caché"""
    cache_size.labels(cache_type=cache_type).set(size_bytes)


def update_websocket_connections(count: int):
    """Actualiza el número de conexiones WebSocket"""
    active_websocket_connections.set(count)


def update_generation_queue_size(size: int):
    """Actualiza el tamaño de la cola de generación"""
    music_generation_queue_size.set(size)


def update_active_generations(count: int):
    """Actualiza el número de generaciones activas"""
    music_generation_active.set(count)


def record_error(error_type: str, endpoint: str):
    """Registra un error"""
    error_total.labels(error_type=error_type, endpoint=endpoint).inc()


async def metrics_endpoint(request: Request) -> Response:
    """Endpoint para exponer métricas Prometheus"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

