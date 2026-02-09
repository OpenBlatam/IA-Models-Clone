from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import time
import logging
import asyncio
import orjson
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
import tracemalloc
from contextlib import asynccontextmanager
import uvicorn
from pydantic import ValidationError
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .models import SEOScrapeRequest, SEOScrapeResponse, SEOAnalysis
from .service import SEOService
from typing import Any, List, Dict, Optional
"""
API SEO Ultra-Optimizada con arquitectura modular y refactorizada.
"""



# Configurar logging optimizado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar tracemalloc para monitoreo de memoria
tracemalloc.start()


@dataclass
class APIMetrics:
    """Métricas de rendimiento de la API."""
    request_count: int
    average_response_time: float
    error_rate: float
    cache_hit_rate: float


class MetricsCollector:
    """Recolector de métricas de la API."""
    
    def __init__(self) -> Any:
        self.request_count = 0
        self.response_times = []
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_request(self, response_time: float, is_error: bool = False, cache_hit: bool = False):
        """Registra una nueva solicitud."""
        self.request_count += 1
        self.response_times.append(response_time)
        
        if is_error:
            self.error_count += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_metrics(self) -> APIMetrics:
        """Obtiene las métricas actuales."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        return APIMetrics(
            request_count=self.request_count,
            average_response_time=round(avg_response_time, 3),
            error_rate=round(error_rate, 2),
            cache_hit_rate=round(cache_hit_rate, 2)
        )


class ServiceManager:
    """Gestor de servicios SEO."""
    
    def __init__(self) -> Any:
        self._service_pool = []
        self._cache = TTLCache(maxsize=2000, ttl=3600)
        self._metrics_collector = MetricsCollector()
    
    def get_service(self) -> SEOService:
        """Obtiene un servicio SEO del pool."""
        if not self._service_pool:
            self._service_pool.append(SEOService())
        return self._service_pool[0]
    
    def get_cache(self) -> TTLCache:
        """Obtiene el cache."""
        return self._cache
    
    def get_metrics(self) -> APIMetrics:
        """Obtiene las métricas de la API."""
        return self._metrics_collector.get_metrics()
    
    def record_request(self, response_time: float, is_error: bool = False, cache_hit: bool = False):
        """Registra una solicitud."""
        self._metrics_collector.record_request(response_time, is_error, cache_hit)
    
    async def cleanup(self) -> Any:
        """Limpia los recursos."""
        for service in self._service_pool:
            await service.close()


class RequestValidator:
    """Validador de solicitudes."""
    
    @staticmethod
    def validate_url(url: str) -> str:
        """Valida y normaliza una URL."""
        if not url:
            raise HTTPException(status_code=400, detail="URL no puede estar vacía")
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        return url
    
    @staticmethod
    def validate_batch_size(urls: List[str]) -> None:
        """Valida el tamaño del lote."""
        if len(urls) > 20:
            raise HTTPException(status_code=400, detail="Máximo 20 URLs por lote")
        
        if not urls:
            raise HTTPException(status_code=400, detail="Lista de URLs no puede estar vacía")


class ResponseFormatter:
    """Formateador de respuestas."""
    
    @staticmethod
    def format_error_response(error: str, status_code: int = 500) -> JSONResponse:
        """Formatea una respuesta de error."""
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": error,
                "timestamp": time.time()
            }
        )
    
    @staticmethod
    def format_success_response(data: Any, cache_hit: bool = False) -> Dict[str, Any]:
        """Formatea una respuesta exitosa."""
        response = {
            "success": True,
            "data": data,
            "timestamp": time.time()
        }
        
        if cache_hit:
            response["cache_hit"] = True
        
        return response


class HealthChecker:
    """Verificador de salud del sistema."""
    
    def __init__(self, service_manager: ServiceManager):
        
    """__init__ function."""
self.service_manager = service_manager
    
    def check_health(self) -> Dict[str, Any]:
        """Verifica el estado del sistema."""
        try:
            current, peak = tracemalloc.get_traced_memory()
            service = self.service_manager.get_service()
            cache = self.service_manager.get_cache()
            metrics = self.service_manager.get_metrics()
            
            return {
                "status": "healthy",
                "service": "SEO Analysis Service - Ultra Optimized",
                "version": "3.0.0",
                "timestamp": time.time(),
                "performance": {
                    "memory_current_mb": round(current / 1024 / 1024, 2),
                    "memory_peak_mb": round(peak / 1024 / 1024, 2),
                    "cache_size": len(cache),
                    "cache_hit_rate": metrics.cache_hit_rate
                },
                "components": {
                    "langchain": service.seo_analyzer.llm is not None,
                    "selenium": service.selenium_manager.driver is not None,
                    "httpx_client": service.http_client.client is not None,
                    "cache": True,
                    "tracemalloc": True
                },
                "api_metrics": {
                    "request_count": metrics.request_count,
                    "average_response_time": metrics.average_response_time,
                    "error_rate": metrics.error_rate
                },
                "features": [
                    "Web scraping con lxml (ultra-rápido)",
                    "Análisis SEO con LangChain optimizado",
                    "Compatibilidad móvil avanzada",
                    "Medición de velocidad de página",
                    "Extracción de tags de redes sociales",
                    "Análisis en lote paralelo",
                    "Sistema de cache con TTL",
                    "Retry automático con backoff exponencial",
                    "Monitoreo de memoria en tiempo real",
                    "Procesamiento asíncrono completo"
                ],
                "endpoints": {
                    "scrape": "POST /seo/scrape",
                    "analyze": "GET /seo/analyze",
                    "batch": "POST /seo/batch",
                    "compare": "GET /seo/compare",
                    "health": "GET /seo/health",
                    "cache_stats": "GET /seo/cache/stats",
                    "cache_clear": "DELETE /seo/cache",
                    "performance": "GET /seo/performance"
                }
            }
        except Exception as e:
            logger.error(f"Error en health check: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Instancia global del gestor de servicios
service_manager = ServiceManager()

# Router principal
router = APIRouter(prefix="/seo", tags=["seo"])


def get_seo_service() -> SEOService:
    """Dependency para obtener el servicio SEO."""
    return service_manager.get_service()


@asynccontextmanager
async def lifespan(app) -> Any:
    """Gestión del ciclo de vida de la aplicación."""
    logger.info("🚀 Iniciando Servicio SEO Ultra-Optimizado")
    yield
    logger.info("🛑 Cerrando Servicio SEO")
    await service_manager.cleanup()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async async def _process_seo_request(request: SEOScrapeRequest, service: SEOService) -> SEOScrapeResponse:
    """Procesa request SEO con retry y optimizaciones."""
    return await service.scrape(request)


@router.post("/scrape", response_model=SEOScrapeResponse)
async def scrape(
    request: SEOScrapeRequest,
    background_tasks: BackgroundTasks,
    seo_service: SEOService = Depends(get_seo_service)
):
    """Realiza scraping SEO ultra-optimizado de una URL usando librerías modernas."""
    start_time = time.time()
    cache_hit = False
    
    try:
        # Validar URL
        url = RequestValidator.validate_url(request.url)
        
        # Verificar cache optimizado
        cache = service_manager.get_cache()
        cache_key = f"{url}_{hash(str(request.options))}"
        
        if cache_key in cache:
            cached_result = cache[cache_key]
            cache_hit = True
            logger.info(f"Resultado obtenido de cache para: {url}")
            
            response_time = time.time() - start_time
            service_manager.record_request(response_time, False, True)
            return cached_result
        
        # Procesar request con retry
        response = await _process_seo_request(request, seo_service)
        
        if response.success:
            # Guardar en cache optimizado
            cache[cache_key] = response
            
            # Limpiar cache antiguo en background
            background_tasks.add_task(_clean_old_cache)
            
            response_time = time.time() - start_time
            service_manager.record_request(response_time, False, False)
            
            logger.info(f"Análisis completado en {response_time:.2f}s para: {url}")
            return response
        else:
            response_time = time.time() - start_time
            service_manager.record_request(response_time, True, False)
            raise HTTPException(status_code=400, detail=response.error)
            
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Error de validación: {e}")
        response_time = time.time() - start_time
        service_manager.record_request(response_time, True, False)
        raise HTTPException(status_code=422, detail=f"Error de validación: {str(e)}")
    except Exception as e:
        logger.error(f"Error inesperado en scraping: {str(e)}")
        response_time = time.time() - start_time
        service_manager.record_request(response_time, True, False)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


@router.get("/analyze", response_model=SEOScrapeResponse)
async def analyze_url(
    url: str = Query(..., description="URL a analizar"),
    use_selenium: bool = Query(False, description="Usar Selenium para contenido dinámico"),
    seo_service: SEOService = Depends(get_seo_service)
):
    """Analiza una URL directamente desde query parameter con optimizaciones."""
    start_time = time.time()
    
    try:
        # Validar URL
        normalized_url = RequestValidator.validate_url(url)
        
        request = SEOScrapeRequest(
            url=normalized_url,
            options={"use_selenium": use_selenium}
        )
        
        response = await _process_seo_request(request, seo_service)
        
        if not response.success:
            response_time = time.time() - start_time
            service_manager.record_request(response_time, True, False)
            raise HTTPException(status_code=400, detail=response.error)
        
        response_time = time.time() - start_time
        service_manager.record_request(response_time, False, False)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analizando URL: {str(e)}")
        response_time = time.time() - start_time
        service_manager.record_request(response_time, True, False)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.post("/batch", response_model=List[SEOScrapeResponse])
async def batch_analyze(
    urls: List[str],
    background_tasks: BackgroundTasks,
    seo_service: SEOService = Depends(get_seo_service)
):
    """Analiza múltiples URLs en lote con procesamiento paralelo optimizado."""
    start_time = time.time()
    
    try:
        # Validar lote
        RequestValidator.validate_batch_size(urls)
        
        # Normalizar URLs
        normalized_urls = [RequestValidator.validate_url(url) for url in urls]
        
        # Procesar URLs en paralelo
        tasks = []
        for url in normalized_urls:
            request = SEOScrapeRequest(url=url)
            task = asyncio.create_task(_process_seo_request(request, seo_service))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SEOScrapeResponse(
                    success=False,
                    error=f"Error analizando {normalized_urls[i]}: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        response_time = time.time() - start_time
        service_manager.record_request(response_time, False, False)
        
        return processed_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis en lote: {str(e)}")
        response_time = time.time() - start_time
        service_manager.record_request(response_time, True, False)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/health")
async def health_check():
    """Verifica el estado del servicio SEO con métricas avanzadas."""
    try:
        health_checker = HealthChecker(service_manager)
        health_status = health_checker.check_health()
        
        if health_status["status"] == "healthy":
            return health_status
        else:
            return JSONResponse(status_code=503, content=health_status)
        
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return ResponseFormatter.format_error_response(str(e), 503)


@router.get("/performance")
async def performance_metrics():
    """Obtiene métricas de rendimiento del servicio."""
    try:
        current, peak = tracemalloc.get_traced_memory()
        cache = service_manager.get_cache()
        metrics = service_manager.get_metrics()
        
        return {
            "memory": {
                "current_mb": round(current / 1024 / 1024, 2),
                "peak_mb": round(peak / 1024 / 1024, 2),
                "usage_percent": round((current / peak) * 100, 2) if peak > 0 else 0
            },
            "cache": {
                "size": len(cache),
                "max_size": cache.maxsize,
                "usage_percent": round((len(cache) / cache.maxsize) * 100, 2)
            },
            "api": {
                "request_count": metrics.request_count,
                "average_response_time": metrics.average_response_time,
                "error_rate": metrics.error_rate,
                "cache_hit_rate": metrics.cache_hit_rate
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.delete("/cache")
async def clear_cache():
    """Limpia el cache de resultados SEO con estadísticas."""
    try:
        cache = service_manager.get_cache()
        cache_size = len(cache)
        cache.clear()
        
        logger.info(f"Cache limpiado. Elementos eliminados: {cache_size}")
        
        return {
            "message": "Cache limpiado exitosamente",
            "elements_removed": cache_size,
            "timestamp": time.time(),
            "memory_freed_mb": cache_size * 0.1  # Estimación
        }
        
    except Exception as e:
        logger.error(f"Error limpiando cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/cache/stats")
async def cache_stats():
    """Obtiene estadísticas detalladas del cache optimizado."""
    try:
        cache = service_manager.get_cache()
        service = service_manager.get_service()
        
        cache_stats = service.get_cache_stats()
        
        return {
            "cache_size": cache_stats["size"],
            "max_size": cache_stats["max_size"],
            "ttl_seconds": cache_stats["ttl"],
            "memory_usage_mb": cache_stats["size"] * 0.1,  # Estimación mejorada
            "hit_rate": cache_stats["hit_rate"],
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas del cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/compare")
async def compare_urls(
    url1: str = Query(..., description="Primera URL a comparar"),
    url2: str = Query(..., description="Segunda URL a comparar"),
    seo_service: SEOService = Depends(get_seo_service)
):
    """Compara el SEO de dos URLs con análisis paralelo."""
    start_time = time.time()
    
    try:
        # Validar URLs
        normalized_url1 = RequestValidator.validate_url(url1)
        normalized_url2 = RequestValidator.validate_url(url2)
        
        # Analizar ambas URLs en paralelo
        request1 = SEOScrapeRequest(url=normalized_url1)
        request2 = SEOScrapeRequest(url=normalized_url2)
        
        task1 = asyncio.create_task(_process_seo_request(request1, seo_service))
        task2 = asyncio.create_task(_process_seo_request(request2, seo_service))
        
        response1, response2 = await asyncio.gather(task1, task2)
        
        if not response1.success or not response2.success:
            response_time = time.time() - start_time
            service_manager.record_request(response_time, True, False)
            raise HTTPException(
                status_code=400, 
                detail=f"Error analizando URLs: {response1.error or response2.error}"
            )
        
        # Comparar métricas con análisis más detallado
        comparison = {
            "url1": {
                "url": normalized_url1,
                "seo_score": response1.data.seo_score,
                "load_time": response1.data.load_time,
                "content_length": response1.data.content_length,
                "mobile_friendly": response1.data.mobile_friendly,
                "page_speed": response1.data.page_speed
            },
            "url2": {
                "url": normalized_url2,
                "seo_score": response2.data.seo_score,
                "load_time": response2.data.load_time,
                "content_length": response2.data.content_length,
                "mobile_friendly": response2.data.mobile_friendly,
                "page_speed": response2.data.page_speed
            },
            "comparison": {
                "seo_score_difference": (response1.data.seo_score or 0) - (response2.data.seo_score or 0),
                "load_time_difference": (response1.data.load_time or 0) - (response2.data.load_time or 0),
                "content_length_difference": response1.data.content_length - response2.data.content_length,
                "winner": "url1" if (response1.data.seo_score or 0) > (response2.data.seo_score or 0) else "url2",
                "performance_winner": "url1" if (response1.data.load_time or 0) < (response2.data.load_time or 0) else "url2"
            }
        }
        
        response_time = time.time() - start_time
        service_manager.record_request(response_time, False, False)
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparando URLs: {str(e)}")
        response_time = time.time() - start_time
        service_manager.record_request(response_time, True, False)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


async def _clean_old_cache():
    """Limpia entradas antiguas del cache de forma asíncrona."""
    try:
        cache = service_manager.get_cache()
        logger.debug(f"Cache cleanup: {len(cache)} elementos activos")
    except Exception as e:
        logger.error(f"Error en limpieza de cache: {e}")


# Middleware para tracking de métricas
@router.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware para tracking de métricas de la API."""
    response = await call_next(request)
    
    # Las métricas se registran en cada endpoint individualmente
    # para mayor precisión y control
    
    return response 