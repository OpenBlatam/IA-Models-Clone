from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import orjson
from loguru import logger
import psutil
import asyncio_throttle
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import uvicorn
from .service_ultra_optimized import UltraOptimizedSEOService, scrape
from .models import SEOScrapeRequest, SEOScrapeResponse
    from .service_ultra_optimized import seo_service
from typing import Any, List, Dict, Optional
import logging
"""
API Ultra-Optimizada para el Servicio SEO con máxima eficiencia.
"""



# Configurar logging ultra-eficiente
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configurar rate limiter ultra-eficiente
limiter = Limiter(key_func=get_remote_address)

# Métricas Prometheus ultra-optimizadas
REQUEST_COUNT = Counter('seo_requests_total', 'Total SEO requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('seo_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('seo_active_requests', 'Number of active requests')
CACHE_HIT_RATIO = Gauge('seo_cache_hit_ratio', 'Cache hit ratio')
MEMORY_USAGE = Gauge('seo_memory_usage_bytes', 'Memory usage in bytes')

# Configurar throttler global
throttler = asyncio_throttle.Throttler(rate_limit=200, period=60)

# Crear aplicación FastAPI ultra-optimizada
app = FastAPI(
    title="SEO Analysis API - Ultra Optimized",
    description="API ultra-optimizada para análisis SEO con máxima eficiencia",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Agregar middleware ultra-optimizado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configurar rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware para métricas ultra-optimizado."""
    
    async def dispatch(self, request: Request, call_next):
        
    """dispatch function."""
start_time = time.perf_counter()
        ACTIVE_REQUESTS.inc()
        
        try:
            response = await call_next(request)
            
            # Registrar métricas
            duration = time.perf_counter() - start_time
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            return response
            
        finally:
            ACTIVE_REQUESTS.dec()


app.add_middleware(MetricsMiddleware)


class BatchRequest(BaseModel):
    """Modelo para requests en lote ultra-optimizado."""
    urls: List[str]
    options: Optional[Dict[str, Any]] = {}
    max_concurrent: Optional[int] = 10


class ComparisonRequest(BaseModel):
    """Modelo para comparación de URLs."""
    url1: str
    url2: str
    options: Optional[Dict[str, Any]] = {}


class HealthResponse(BaseModel):
    """Modelo para respuesta de salud."""
    status: str
    timestamp: float
    version: str
    memory_usage: float
    cache_stats: Dict[str, Any]
    system_info: Dict[str, Any]


# Dependencias ultra-optimizadas
async def get_seo_service() -> UltraOptimizedSEOService:
    """Obtiene instancia del servicio SEO."""
    return seo_service


async def validate_url(url: str) -> str:
    """Valida URL ultra-rápido."""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


@app.on_event("startup")
async def startup_event():
    """Evento de inicio ultra-optimizado."""
    logger.info("🚀 Iniciando API SEO Ultra-Optimizada")
    
    # Actualizar métricas de sistema
    MEMORY_USAGE.set(psutil.Process().memory_info().rss)
    
    # Configurar actualización periódica de métricas
    asyncio.create_task(update_system_metrics())


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre ultra-optimizado."""
    logger.info("🛑 Cerrando API SEO Ultra-Optimizada")
    seo_service = await get_seo_service()
    await seo_service.close()


async def update_system_metrics():
    """Actualiza métricas del sistema periódicamente."""
    while True:
        try:
            MEMORY_USAGE.set(psutil.Process().memory_info().rss)
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz ultra-optimizado."""
    return {"message": "SEO Analysis API - Ultra Optimized", "version": "2.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health_check(seo_service: UltraOptimizedSEOService = Depends(get_seo_service)):
    """Health check ultra-optimizado."""
    cache_stats = seo_service.get_cache_stats()
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="2.0.0",
        memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
        cache_stats=cache_stats,
        system_info={
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
    )


@app.post("/scrape", response_model=SEOScrapeResponse)
@limiter.limit("100/minute")
async def scrape_url(
    request: SEOScrapeRequest,
    background_tasks: BackgroundTasks,
    seo_service: UltraOptimizedSEOService = Depends(get_seo_service)
):
    """Scraping SEO ultra-optimizado con rate limiting."""
    async with throttler:
        try:
            # Validar URL
            request.url = await validate_url(request.url)
            
            # Realizar scraping
            result = await seo_service.scrape(request)
            
            # Actualizar métricas de caché
            cache_stats = seo_service.get_cache_stats()
            CACHE_HIT_RATIO.set(cache_stats.get("hit_rate", 0.0))
            
            # Tarea en background para logging
            background_tasks.add_task(log_request, request.url, result.success)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in scrape endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
@limiter.limit("50/minute")
async def analyze_url(
    url: str,
    options: Optional[Dict[str, Any]] = {},
    seo_service: UltraOptimizedSEOService = Depends(get_seo_service)
):
    """Análisis SEO ultra-optimizado."""
    async with throttler:
        try:
            url = await validate_url(url)
            request = SEOScrapeRequest(url=url, options=options)
            result = await seo_service.scrape(request)
            
            return {
                "url": url,
                "analysis": result.data.get("analysis", {}),
                "metrics": result.metrics
            }
            
        except Exception as e:
            logger.error(f"Error in analyze endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
@limiter.limit("20/minute")
async def batch_analysis(
    batch_request: BatchRequest,
    seo_service: UltraOptimizedSEOService = Depends(get_seo_service)
):
    """Análisis en lote ultra-optimizado."""
    async with throttler:
        try:
            results = []
            semaphore = asyncio.Semaphore(batch_request.max_concurrent)
            
            async def process_url(url: str) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        url = await validate_url(url)
                        request = SEOScrapeRequest(url=url, options=batch_request.options)
                        result = await seo_service.scrape(request)
                        return {
                            "url": url,
                            "success": result.success,
                            "data": result.data if result.success else None,
                            "error": result.error if not result.success else None,
                            "metrics": result.metrics
                        }
                    except Exception as e:
                        return {
                            "url": url,
                            "success": False,
                            "error": str(e),
                            "metrics": None
                        }
            
            # Procesar URLs en paralelo
            tasks = [process_url(url) for url in batch_request.urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "total_urls": len(batch_request.urls),
                "successful": sum(1 for r in results if isinstance(r, dict) and r.get("success")),
                "failed": sum(1 for r in results if isinstance(r, dict) and not r.get("success")),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
@limiter.limit("30/minute")
async def compare_urls(
    comparison_request: ComparisonRequest,
    seo_service: UltraOptimizedSEOService = Depends(get_seo_service)
):
    """Comparación de URLs ultra-optimizada."""
    async with throttler:
        try:
            url1 = await validate_url(comparison_request.url1)
            url2 = await validate_url(comparison_request.url2)
            
            # Analizar ambas URLs en paralelo
            request1 = SEOScrapeRequest(url=url1, options=comparison_request.options)
            request2 = SEOScrapeRequest(url=url2, options=comparison_request.options)
            
            results = await asyncio.gather(
                seo_service.scrape(request1),
                seo_service.scrape(request2),
                return_exceptions=True
            )
            
            if isinstance(results[0], Exception) or isinstance(results[1], Exception):
                raise HTTPException(status_code=500, detail="Error analyzing URLs")
            
            result1, result2 = results
            
            # Comparar resultados
            comparison = {
                "url1": {
                    "url": url1,
                    "score": result1.data.get("analysis", {}).get("score", 0),
                    "recommendations": result1.data.get("analysis", {}).get("recommendations", [])
                },
                "url2": {
                    "url": url2,
                    "score": result2.data.get("analysis", {}).get("score", 0),
                    "recommendations": result2.data.get("analysis", {}).get("recommendations", [])
                },
                "comparison": {
                    "score_difference": abs(result1.data.get("analysis", {}).get("score", 0) - 
                                          result2.data.get("analysis", {}).get("score", 0)),
                    "better_url": url1 if result1.data.get("analysis", {}).get("score", 0) > 
                                  result2.data.get("analysis", {}).get("score", 0) else url2
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error in comparison: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def get_cache_stats(seo_service: UltraOptimizedSEOService = Depends(get_seo_service)):
    """Obtiene estadísticas del caché."""
    return seo_service.get_cache_stats()


@app.delete("/cache/clear")
async def clear_cache(seo_service: UltraOptimizedSEOService = Depends(get_seo_service)):
    """Limpia el caché."""
    cleared_count = seo_service.clear_cache()
    return {"message": f"Cache cleared. {cleared_count} items removed."}


@app.get("/metrics")
async def metrics():
    """Endpoint para métricas Prometheus."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/system/info")
async def system_info():
    """Información del sistema ultra-optimizada."""
    return {
        "cpu": {
            "count": psutil.cpu_count(),
            "percent": psutil.cpu_percent(interval=1),
            "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent,
            "used": psutil.virtual_memory().used
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "used": psutil.disk_usage('/').used,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        },
        "network": {
            "connections": len(psutil.net_connections())
        }
    }


async def log_request(url: str, success: bool):
    """Log de request en background."""
    logger.info(f"Request processed: {url} - Success: {success}")


# Configuración de uvicorn ultra-optimizada
if __name__ == "__main__":
    uvicorn.run(
        "api_ultra_optimized:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        loop="uvloop",
        http="httptools",
        access_log=False,
        log_level="info"
    ) 