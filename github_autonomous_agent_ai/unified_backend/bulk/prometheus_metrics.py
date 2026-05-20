"""
Métricas Prometheus para la API BUL
Expone métricas en formato Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server
from fastapi import FastAPI, Response
from typing import Dict, Any
import time

REQUEST_COUNT = Counter(
    'bul_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'bul_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_TASKS = Gauge(
    'bul_active_tasks',
    'Number of active tasks'
)

DOCUMENT_GENERATION_TIME = Histogram(
    'bul_document_generation_seconds',
    'Document generation time in seconds'
)

CACHE_HITS = Counter(
    'bul_cache_hits_total',
    'Total cache hits'
)

CACHE_MISSES = Counter(
    'bul_cache_misses_total',
    'Total cache misses'
)

ERROR_COUNT = Counter(
    'bul_errors_total',
    'Total number of errors',
    ['error_type']
)

def setup_prometheus_metrics(app: FastAPI):
    """Configura métricas Prometheus en la app FastAPI."""
    
    @app.get("/metrics")
    async def metrics():
        """Endpoint de métricas Prometheus."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        """Middleware para recopilar métricas."""
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        if response.status_code >= 400:
            ERROR_COUNT.labels(error_type=f"http_{response.status_code}").inc()
        
        return response

def start_prometheus_server(port: int = 9090):
    """Inicia servidor Prometheus."""
    start_http_server(port)
    print(f"📊 Métricas Prometheus disponibles en http://localhost:{port}/metrics")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Servidor de Métricas Prometheus")
    parser.add_argument("--port", type=int, default=9090, help="Puerto del servidor")
    
    args = parser.parse_args()
    
    start_prometheus_server(args.port)
    
    print("Servidor Prometheus iniciado. Presiona Ctrl+C para detener.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServidor detenido")
































