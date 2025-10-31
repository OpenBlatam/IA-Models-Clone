from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
from loguru import logger
import orjson
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
import slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from pydantic import BaseModel, HttpUrl, validator
import hashlib
import jwt
from datetime import datetime, timedelta
from ..services.ultra_optimized_seo_service_v2 import UltraOptimizedSEOServiceV2, SEOAnalysisResult
from ..core.config import get_settings
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Optimized SEO API v2.0 - Production Ready
FastAPI with advanced features, metrics, security, and performance optimizations
"""




# Pydantic Models
class SEOAnalysisRequest(BaseModel):
    url: HttpUrl
    force_refresh: bool = False
    include_recommendations: bool = True
    include_warnings: bool = True
    include_errors: bool = True

class BatchAnalysisRequest(BaseModel):
    urls: List[HttpUrl]
    max_concurrent: Optional[int] = None
    force_refresh: bool = False

class SEOAnalysisResponse(BaseModel):
    url: str
    title: str
    meta_description: str
    seo_score: float
    recommendations: List[str]
    warnings: List[str]
    errors: List[str]
    stats: Dict[str, Any]
    cached: bool
    analysis_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime: float
    components: Dict[str, Any]
    stats: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime
    request_id: str


# Metrics
REQUEST_COUNT = Counter('seo_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('seo_api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ANALYSIS_COUNT = Counter('seo_analysis_total', 'Total SEO analyses', ['status'])
ANALYSIS_DURATION = Histogram('seo_analysis_duration_seconds', 'Analysis duration')
CACHE_HITS = Counter('seo_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('seo_cache_misses_total', 'Cache misses')
ACTIVE_REQUESTS = Gauge('seo_active_requests', 'Active requests')


# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
slowapi.rate_limit_exceeded_handler = _rate_limit_exceeded_handler


# Security
security = HTTPBearer(auto_error=False)


class UltraOptimizedAPIv2:
    """API ultra-optimizada para producción."""
    
    def __init__(self) -> Any:
        self.settings = get_settings()
        self.start_time = time.time()
        self.seo_service = None
        self.app = None
        self._init_app()
    
    def _init_app(self) -> Any:
        """Inicializar aplicación FastAPI."""
        self.app = FastAPI(
            title="Ultra-Optimized SEO API v2.0",
            description="API de análisis SEO ultra-optimizada para producción",
            version="2.0.0",
            docs_url="/docs" if self.settings.debug else None,
            redoc_url="/redoc" if self.settings.debug else None,
            openapi_url="/openapi.json" if self.settings.debug else None
        )
        
        # Configurar middleware
        self._setup_middleware()
        
        # Configurar rutas
        self._setup_routes()
        
        # Configurar métricas
        self._setup_metrics()
        
        # Configurar rate limiting
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        logger.info("Ultra-Optimized API v2.0 initialized")
    
    def _setup_middleware(self) -> Any:
        """Configurar middleware."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware para métricas y logging
        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            
    """metrics_middleware function."""
start_time = time.time()
            ACTIVE_REQUESTS.inc()
            
            # Generar request ID
            request_id = hashlib.md5(f"{time.time()}:{request.client.host}".encode()).hexdigest()
            request.state.request_id = request_id
            
            try:
                response = await call_next(request)
                
                # Registrar métricas
                duration = time.time() - start_time
                REQUEST_DURATION.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)
                
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                
                # Logging estructurado
                logger.info(
                    "Request processed",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration": duration,
                        "client_ip": request.client.host,
                        "user_agent": request.headers.get("user-agent", "")
                    }
                )
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=500
                ).inc()
                
                logger.error(
                    "Request failed",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "error": str(e),
                        "duration": duration
                    }
                )
                raise
            finally:
                ACTIVE_REQUESTS.dec()
    
    def _setup_routes(self) -> Any:
        """Configurar rutas de la API."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Evento de inicio."""
            logger.info("Starting Ultra-Optimized SEO API v2.0")
            self.seo_service = UltraOptimizedSEOServiceV2(self.settings.seo_service_config)
            logger.info("SEO service initialized")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Evento de cierre."""
            logger.info("Shutting down Ultra-Optimized SEO API v2.0")
            if self.seo_service:
                await self.seo_service.close()
            logger.info("SEO service closed")
        
        # Health check
        @self.app.get("/health", response_model=HealthResponse)
        @limiter.limit("100/minute")
        async def health_check(request: Request):
            """Health check del servicio."""
            try:
                uptime = time.time() - self.start_time
                
                # Verificar componentes
                seo_health = await self.seo_service.health_check() if self.seo_service else {"status": "unavailable"}
                
                return HealthResponse(
                    status="healthy" if seo_health["status"] == "healthy" else "degraded",
                    timestamp=datetime.utcnow(),
                    version="2.0.0",
                    uptime=uptime,
                    components={
                        "seo_service": seo_health
                    },
                    stats=self.seo_service.get_stats() if self.seo_service else {}
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Service unhealthy")
        
        # Análisis SEO individual
        @self.app.post("/analyze", response_model=SEOAnalysisResponse)
        @limiter.limit("50/minute")
        async def analyze_url(
            request: SEOAnalysisRequest,
            background_tasks: BackgroundTasks,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Analizar URL individual."""
            try:
                # Verificar autenticación
                if not self._verify_token(auth):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                start_time = time.time()
                
                # Realizar análisis
                result = await self.seo_service.analyze_url(
                    str(request.url),
                    force_refresh=request.force_refresh
                )
                
                analysis_time = time.time() - start_time
                
                # Registrar métricas
                ANALYSIS_COUNT.labels(status="success").inc()
                ANALYSIS_DURATION.observe(analysis_time)
                
                if result.cached:
                    CACHE_HITS.inc()
                else:
                    CACHE_MISSES.inc()
                
                # Preparar respuesta
                response_data = {
                    "url": result.url,
                    "title": result.title,
                    "meta_description": result.meta_description,
                    "seo_score": result.seo_score,
                    "recommendations": result.recommendations if request.include_recommendations else [],
                    "warnings": result.warnings if request.include_warnings else [],
                    "errors": result.errors if request.include_errors else [],
                    "stats": {
                        "word_count": result.word_count,
                        "character_count": result.character_count,
                        "link_count": result.link_count,
                        "image_count": result.image_count,
                        "form_count": result.form_count,
                        "load_time": result.load_time,
                        "parsing_time": result.parsing_time,
                        "analysis_time": result.analysis_time,
                        "total_time": result.total_time
                    },
                    "cached": result.cached,
                    "analysis_time": analysis_time
                }
                
                # Background task para logging
                background_tasks.add_task(
                    self._log_analysis,
                    result.url,
                    result.seo_score,
                    analysis_time,
                    result.cached
                )
                
                return SEOAnalysisResponse(**response_data)
                
            except Exception as e:
                ANALYSIS_COUNT.labels(status="error").inc()
                logger.error(f"Analysis failed for {request.url}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Análisis SEO en lote
        @self.app.post("/analyze/batch")
        @limiter.limit("10/minute")
        async def analyze_urls_batch(
            request: BatchAnalysisRequest,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Analizar múltiples URLs en lote."""
            try:
                # Verificar autenticación
                if not self._verify_token(auth):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Validar límites
                if len(request.urls) > 100:
                    raise HTTPException(status_code=400, detail="Maximum 100 URLs per batch")
                
                start_time = time.time()
                
                # Realizar análisis en lote
                results = await self.seo_service.analyze_urls_batch(
                    [str(url) for url in request.urls],
                    max_concurrent=request.max_concurrent
                )
                
                batch_time = time.time() - start_time
                
                # Procesar resultados
                processed_results = []
                for i, result in enumerate(results):
                    if result is None:
                        processed_results.append({
                            "url": str(request.urls[i]),
                            "error": "Analysis failed"
                        })
                    else:
                        processed_results.append({
                            "url": result.url,
                            "title": result.title,
                            "seo_score": result.seo_score,
                            "cached": result.cached,
                            "analysis_time": result.total_time
                        })
                
                return {
                    "batch_id": hashlib.md5(f"{time.time()}:{len(request.urls)}".encode()).hexdigest(),
                    "total_urls": len(request.urls),
                    "successful_analyses": len([r for r in results if r is not None]),
                    "failed_analyses": len([r for r in results if r is None]),
                    "batch_time": batch_time,
                    "results": processed_results
                }
                
            except Exception as e:
                logger.error(f"Batch analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Métricas Prometheus
        @self.app.get("/metrics")
        async def metrics():
            """Endpoint de métricas Prometheus."""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
        
        # Estadísticas del servicio
        @self.app.get("/stats")
        @limiter.limit("30/minute")
        async def get_stats(auth: HTTPAuthorizationCredentials = Depends(security)):
            """Obtener estadísticas del servicio."""
            try:
                # Verificar autenticación
                if not self._verify_token(auth):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                return {
                    "service_stats": self.seo_service.get_stats() if self.seo_service else {},
                    "api_stats": {
                        "uptime": time.time() - self.start_time,
                        "version": "2.0.0",
                        "timestamp": datetime.utcnow()
                    }
                }
                
            except Exception as e:
                logger.error(f"Stats retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Limpiar cache
        @self.app.post("/cache/clear")
        @limiter.limit("5/minute")
        async def clear_cache(auth: HTTPAuthorizationCredentials = Depends(security)):
            """Limpiar cache del servicio."""
            try:
                # Verificar autenticación
                if not self._verify_token(auth):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                await self.seo_service.clear_cache()
                
                return {"message": "Cache cleared successfully"}
                
            except Exception as e:
                logger.error(f"Cache clear failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Comparar URLs
        @self.app.post("/compare")
        @limiter.limit("20/minute")
        async def compare_urls(
            urls: List[HttpUrl],
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Comparar múltiples URLs."""
            try:
                # Verificar autenticación
                if not self._verify_token(auth):
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Validar límites
                if len(urls) < 2 or len(urls) > 10:
                    raise HTTPException(status_code=400, detail="Compare 2-10 URLs")
                
                # Analizar URLs
                results = await self.seo_service.analyze_urls_batch(
                    [str(url) for url in urls]
                )
                
                # Preparar comparación
                comparison = {
                    "urls": [str(url) for url in urls],
                    "comparison": {
                        "seo_scores": [r.seo_score if r else 0 for r in results],
                        "word_counts": [r.word_count if r else 0 for r in results],
                        "link_counts": [r.link_count if r else 0 for r in results],
                        "image_counts": [r.image_count if r else 0 for r in results]
                    },
                    "recommendations": self._generate_comparison_recommendations(results)
                }
                
                return comparison
                
            except Exception as e:
                logger.error(f"URL comparison failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Error handlers
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Manejador de errores HTTP."""
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error=exc.detail,
                    detail=str(exc),
                    timestamp=datetime.utcnow(),
                    request_id=getattr(request.state, 'request_id', 'unknown')
                ).dict()
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Manejador de errores generales."""
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(exc),
                    timestamp=datetime.utcnow(),
                    request_id=getattr(request.state, 'request_id', 'unknown')
                ).dict()
            )
    
    def _setup_metrics(self) -> Any:
        """Configurar métricas Prometheus."""
        Instrumentator().instrument(self.app).expose(self.app)
    
    def _verify_token(self, auth: HTTPAuthorizationCredentials) -> bool:
        """Verificar token de autenticación."""
        if not auth:
            return False
        
        try:
            # Verificar JWT token
            payload = jwt.decode(
                auth.credentials,
                self.settings.jwt_secret_key,
                algorithms=["HS256"]
            )
            return True
        except jwt.InvalidTokenError:
            return False
    
    async def _log_analysis(self, url: str, seo_score: float, analysis_time: float, cached: bool):
        """Log de análisis en background."""
        logger.info(
            "SEO analysis completed",
            extra={
                "url": url,
                "seo_score": seo_score,
                "analysis_time": analysis_time,
                "cached": cached
            }
        )
    
    def _generate_comparison_recommendations(self, results: List[SEOAnalysisResult]) -> List[str]:
        """Generar recomendaciones de comparación."""
        recommendations = []
        
        if not results or all(r is None for r in results):
            return ["No valid results to compare"]
        
        valid_results = [r for r in results if r is not None]
        
        if len(valid_results) < 2:
            return ["Need at least 2 valid results for comparison"]
        
        # Comparar puntuaciones SEO
        scores = [r.seo_score for r in valid_results]
        best_score = max(scores)
        worst_score = min(scores)
        
        if best_score - worst_score > 0.3:
            recommendations.append("Significant SEO score differences detected")
        
        # Comparar contenido
        word_counts = [r.word_count for r in valid_results]
        if max(word_counts) - min(word_counts) > 500:
            recommendations.append("Content length varies significantly")
        
        return recommendations
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Ejecutar servidor de desarrollo."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info" if self.settings.debug else "warning",
            access_log=True,
            **kwargs
        )


# Instancia global de la API
api = UltraOptimizedAPIv2()
app = api.app


match __name__:
    case "__main__":
    api.run() 