from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
import uuid
from ..nlp.core.engine import ProductionNLPEngine, RequestContext
from ..nlp.utils.cache import get_global_cache
    import uvicorn
from typing import Any, List, Dict, Optional
"""
🚀 Production REST API
======================

API REST de producción para el sistema NLP con FastAPI.
"""


# Internal imports


# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    """Request model para análisis."""
    text: str = Field(..., min_length=1, max_length=10000, description="Texto a analizar")
    analyzers: Optional[List[str]] = Field(default=None, description="Analizadores a usar")
    user_id: Optional[str] = Field(default=None, description="ID del usuario")
    workspace_id: Optional[str] = Field(default=None, description="ID del workspace")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadatos adicionales")
    
    @validator('analyzers')
    def validate_analyzers(cls, v) -> bool:
        if v is not None:
            allowed = ['sentiment', 'engagement', 'emotion', 'readability', 'topics']
            invalid = [a for a in v if a not in allowed]
            if invalid:
                raise ValueError(f'Invalid analyzers: {invalid}. Allowed: {allowed}')
        return v
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "text": "¡Esta es una publicación increíble! ¿Qué opinas? 😊 #marketing",
                "analyzers": ["sentiment", "engagement"],
                "user_id": "user123",
                "workspace_id": "workspace456"
            }
        }


class AnalysisResponse(BaseModel):
    """Response model para análisis."""
    request_id: str
    results: Dict[str, Any]
    processing_time_ms: float
    timestamp: str
    success: bool
    cached: bool = False
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "request_id": "abc123",
                "results": {
                    "sentiment": {
                        "polarity": 0.8,
                        "label": "positive",
                        "confidence": 0.9
                    }
                },
                "processing_time_ms": 45.2,
                "timestamp": "2024-01-01T12:00:00Z",
                "success": True,
                "cached": False
            }
        }


class HealthResponse(BaseModel):
    """Response model para health check."""
    status: str
    timestamp: str
    uptime_seconds: float
    checks: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Response model para métricas."""
    requests: Dict[str, Any]
    performance: Dict[str, Any]
    cache: Dict[str, Any]
    status: str


class ErrorResponse(BaseModel):
    """Response model para errores."""
    error: str
    error_type: str
    request_id: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


# Global engine instance
engine: Optional[ProductionNLPEngine] = None


async def get_engine() -> ProductionNLPEngine:
    """Dependency para obtener engine."""
    global engine
    if engine is None:
        engine = ProductionNLPEngine()
    return engine


# FastAPI app
app = FastAPI(
    title="Facebook Posts NLP API",
    description="API de producción para análisis NLP de Facebook posts",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción sería más restrictivo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nlp_api")


@app.on_event("startup")
async def startup_event():
    """Inicialización al startup."""
    global engine
    try:
        logger.info("Starting NLP API")
        engine = ProductionNLPEngine()
        logger.info("NLP API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al shutdown."""
    global engine
    try:
        logger.info("Shutting down NLP API")
        if engine:
            await engine.shutdown()
        logger.info("NLP API shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc) -> Any:
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=str(exc),
            error_type="ValueError",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(HTTPException)
async async def http_exception_handler(request, exc) -> Any:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_type="HTTPException",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# API Endpoints
@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint con información básica."""
    return {
        "service": "Facebook Posts NLP API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "/analyze",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analizar texto",
    description="Analizar texto con múltiples analizadores NLP"
)
async def analyze_text(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    engine: ProductionNLPEngine = Depends(get_engine)
):
    """
    Analizar texto con el motor NLP.
    
    - **text**: Texto a analizar (1-10000 caracteres)
    - **analyzers**: Lista de analizadores ['sentiment', 'engagement', 'emotion']
    - **user_id**: ID del usuario (opcional)
    - **workspace_id**: ID del workspace (opcional)
    """
    try:
        # Crear contexto
        context = RequestContext(
            user_id=request.user_id,
            request_id=str(uuid.uuid4())
        )
        
        # Verificar cache
        cache = await get_global_cache()
        cache_key = f"nlp:{hash(request.text + str(request.analyzers))}"
        cached_result = await cache.get(cache_key)
        
        if cached_result:
            return AnalysisResponse(
                request_id=context.request_id,
                results=cached_result,
                processing_time_ms=1.0,
                timestamp=datetime.now().isoformat(),
                success=True,
                cached=True
            )
        
        # Realizar análisis
        result = await engine.analyze_text(
            text=request.text,
            analyzers=request.analyzers,
            context=context
        )
        
        # Cachear resultado
        background_tasks.add_task(
            cache.set,
            cache_key,
            result,
            ttl=3600
        )
        
        return AnalysisResponse(
            request_id=context.request_id,
            results=result,
            processing_time_ms=context.elapsed_ms(),
            timestamp=datetime.now().isoformat(),
            success=True,
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Verificar estado de salud del sistema"
)
async def health_check(engine: ProductionNLPEngine = Depends(get_engine)):
    """Health check comprehensivo del sistema."""
    try:
        health_data = await engine.health_check()
        
        return HealthResponse(
            status=health_data["status"],
            timestamp=health_data["timestamp"],
            uptime_seconds=0.0,  # Calcular uptime real
            checks=health_data.get("checks", {})
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Obtener métricas",
    description="Métricas de performance y uso del sistema"
)
async def get_metrics(engine: ProductionNLPEngine = Depends(get_engine)):
    """Obtener métricas detalladas del sistema."""
    try:
        metrics = await engine.get_metrics()
        cache = await get_global_cache()
        cache_stats = cache.get_stats()
        
        return MetricsResponse(
            requests=metrics.get("requests", {}),
            performance=metrics.get("performance", {}),
            cache=cache_stats,
            status=metrics.get("status", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics collection failed: {str(e)}"
        )


@app.get("/cache/stats", summary="Estadísticas de cache")
async def get_cache_stats():
    """Obtener estadísticas detalladas del cache."""
    try:
        cache = await get_global_cache()
        return cache.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/clear", summary="Limpiar cache")
async def clear_cache():
    """Limpiar todo el cache."""
    try:
        cache = await get_global_cache()
        await cache.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/analyze", summary="Análisis en lote")
async def batch_analyze(
    requests: List[AnalysisRequest],
    engine: ProductionNLPEngine = Depends(get_engine)
):
    """Analizar múltiples textos en paralelo."""
    if len(requests) > 10:  # Límite de batch
        raise HTTPException(
            status_code=400,
            detail="Batch size too large (max 10)"
        )
    
    try:
        tasks = []
        for req in requests:
            context = RequestContext(
                user_id=req.user_id,
                request_id=str(uuid.uuid4())
            )
            task = engine.analyze_text(req.text, req.analyzers, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        response = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                response.append({
                    "success": False,
                    "error": str(result),
                    "request_index": i
                })
            else:
                response.append({
                    "success": True,
                    "result": result,
                    "request_index": i
                })
        
        return {"batch_results": response}
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


# Debugging endpoints (solo en desarrollo)
@app.get("/debug/analyzers", include_in_schema=False)
async def debug_analyzers(engine: ProductionNLPEngine = Depends(get_engine)):
    """Debug info sobre analizadores."""
    return {
        "available_analyzers": list(engine.analyzers.keys()),
        "analyzer_health": engine.analyzer_health if hasattr(engine, 'analyzer_health') else {}
    }


if __name__ == "__main__":
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 