from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import APIRouter, HTTPException, Query, Depends, Body, status, Request, Security
from models import CopywritingInput, CopywritingOutput, Feedback, SectionFeedback, CopyVariantHistory, get_settings
from service import CopywritingService
from tasks import generate_copywriting_task
from typing import List, Optional
from celery.result import AsyncResult
import logging
import os

# Import the new v11 optimized engine and API components
from ultra_optimized_engine_v11 import UltraOptimizedEngineV11, get_engine, cleanup_engine
from optimized_api_v11 import APIConfig, PerformanceTracker

from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
from sqlalchemy.orm import Session
from fastapi import Depends
from fastapi_limiter.depends import RateLimiter
from fastapi_cache2.decorator import cache
import sys
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi_filter import FilterDepends, with_prefix
from fastapi_filter.contrib.sqlalchemy import Filter
from typing import Any, List, Dict, Optional
import asyncio
import time
import psutil
import os
# Prometheus metrics
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# FastAPI Security

settings = get_settings()

# Use settings.api_key in get_api_key
API_KEY = settings.api_key

# Use settings.allowed_cors_origins for CORS (in main app, but show here as comment)
# allow_origins=settings.allowed_cors_origins

# Use settings.redis_url for Redis (limiter/cache)
# redis = await aioredis.create_redis_pool(settings.redis_url)

# Use settings.mlflow_tracking_uri and settings.dask_scheduler_address as needed in other modules

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

# SQLAlchemy para persistencia de feedback

SQLITE_URL = os.environ.get("COPYWRITING_FEEDBACK_DB", "sqlite:///copywriting_feedback.db")
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class FeedbackDB(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    variant_id = Column(String, index=True)
    type = Column(String)
    score = Column(Float, nullable=True)
    comments = Column(Text, nullable=True)
    user_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    raw_json = Column(Text)

Base.metadata.create_all(bind=engine)

router = APIRouter(prefix="/copywriting", tags=["copywriting"])
logger = logging.getLogger("copywriting.api")

AVAILABLE_MODELS = ["gpt2", "distilgpt2"]  # Puedes expandir esta lista
MAX_BATCH_SIZE = 20

# Instrumentator para métricas
instrumentator = Instrumentator() if PROMETHEUS_AVAILABLE else None

# Initialize v11 performance tracker
performance_tracker = PerformanceTracker()

# System monitoring
def get_system_stats():
    """Get current system statistics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2)
        }
    except Exception as e:
        logger.warning(f"Failed to get system stats: {e}")
        return {"error": "System stats unavailable"}

# Seguridad básica: placeholder para autenticación/roles
# from fastapi import Security, Depends
# def get_current_user(): ...


def get_service(model_name: str = Query("gpt2", description="HuggingFace model name")):
    return CopywritingService(model_name=model_name)

# Get v11 engine instance
async def get_v11_engine():
    """Get the v11 optimized engine instance."""
    return await get_engine()

@router.on_event("startup")
def _setup_metrics():
    
    """_setup_metrics function."""
if PROMETHEUS_AVAILABLE and instrumentator:
        # Instrumenta solo si no está ya instrumentado
        if not getattr(sys.modules[__name__], "_instrumented", False):
            app = router.routes[0].app if router.routes else None
            if isinstance(app, FastAPI):
                instrumentator.instrument(app).expose(app, endpoint="/copywriting/metrics", include_in_schema=True)
                setattr(sys.modules[__name__], "_instrumented", True)

@router.get("/metrics", include_in_schema=False)
def metrics():
    """metrics function."""
    if PROMETHEUS_AVAILABLE and instrumentator:
        return instrumentator.registry.generate_latest(), 200, {"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}
    else:
        return {"error": "Prometheus metrics not available. Install prometheus_fastapi_instrumentator."}

# Enhanced v11 endpoints with better monitoring and error handling
@router.get("/v11/health", summary="Health check for v11 optimized system", tags=["copywriting"])
async def health_check_v11():
    """Enhanced health check for the v11 optimized system."""
    start_time = time.time()
    
    try:
        engine = await get_v11_engine()
        system_stats = get_system_stats()
        
        # Check engine components
        engine_status = {
            "engine_initialized": engine is not None,
            "cache_available": hasattr(engine, 'intelligent_cache') if engine else False,
            "memory_manager_active": hasattr(engine, 'memory_manager') if engine else False,
            "batch_processor_active": hasattr(engine, 'batch_processor') if engine else False,
            "circuit_breaker_active": hasattr(engine, 'circuit_breaker') if engine else False
        }
        
        response_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "version": "v11",
            "engine_status": engine_status,
            "system_stats": system_stats,
            "response_time_ms": round(response_time * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "version": "v11",
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/v11/performance-stats", summary="Get v11 performance statistics", tags=["copywriting"])
async def get_performance_stats_v11():
    """Get detailed performance statistics from the v11 engine."""
    try:
        engine = await get_v11_engine()
        stats = engine.get_performance_stats()
        system_stats = get_system_stats()
        
        # Add API-level stats
        api_stats = {
            "total_requests": performance_tracker.total_requests,
            "average_response_time": performance_tracker.get_average_response_time(),
            "error_rate": performance_tracker.get_error_rate(),
            "cache_hit_ratio": performance_tracker.get_cache_hit_ratio()
        }
        
        return {
            "version": "v11",
            "engine_stats": stats,
            "api_stats": api_stats,
            "system_stats": system_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}")

@router.post("/v11/generate", summary="Generate copywriting using v11 optimized engine", tags=["copywriting"])
async def generate_copywriting_v11(
    request: CopywritingInput = Body(..., example={
        "product_description": "Zapatos deportivos de alta gama",
        "target_platform": "Instagram",
        "tone": "inspirational",
        "target_audience": "Jóvenes activos",
        "key_points": ["Comodidad", "Estilo", "Durabilidad"],
        "instructions": "Enfatiza la innovación.",
        "restrictions": ["no mencionar precio"],
        "creativity_level": 0.8,
        "language": "es"
    }),
    api_key: str = Depends(get_api_key)
):
    """Generate copywriting using the v11 optimized engine with enhanced monitoring."""
    start_time = time.time()
    
    try:
        engine = await get_v11_engine()
        
        # Check system resources before generation
        system_stats_before = get_system_stats()
        
        result = await engine.generate_copywriting(request)
        
        # Calculate processing metrics
        processing_time = time.time() - start_time
        system_stats_after = get_system_stats()
        
        # Track performance
        performance_tracker.track_request(processing_time, cache_hit=False)
        
        return {
            "version": "v11",
            "result": result,
            "processing_time_seconds": round(processing_time, 3),
            "system_stats_before": system_stats_before,
            "system_stats_after": system_stats_after,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"v11 generation failed: {e}")
        performance_tracker.track_error()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/v11/batch-generate", summary="Batch generate using v11 optimized engine", tags=["copywriting"])
async def batch_generate_copywriting_v11(
    requests: List[CopywritingInput] = Body(..., example=[{
        "product_description": "Zapatos deportivos de alta gama",
        "target_platform": "Instagram",
        "tone": "inspirational",
        "language": "es"
    }]),
    api_key: str = Depends(get_api_key)
):
    """Batch generate copywriting using the v11 optimized engine with enhanced monitoring."""
    if len(requests) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")
    
    start_time = time.time()
    results = []
    successful_count = 0
    failed_count = 0
    
    try:
        engine = await get_v11_engine()
        system_stats_before = get_system_stats()
        
        for i, request in enumerate(requests):
            try:
                request_start = time.time()
                result = await engine.generate_copywriting(request)
                request_time = time.time() - request_start
                
                results.append({
                    "index": i,
                    "success": True,
                    "result": result,
                    "processing_time_seconds": round(request_time, 3)
                })
                successful_count += 1
                
            except Exception as e:
                request_time = time.time() - request_start
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e),
                    "processing_time_seconds": round(request_time, 3)
                })
                failed_count += 1
        
        total_processing_time = time.time() - start_time
        system_stats_after = get_system_stats()
        
        performance_tracker.track_batch_request(total_processing_time, len(requests))
        
        return {
            "version": "v11",
            "results": results,
            "total_processing_time_seconds": round(total_processing_time, 3),
            "successful_generations": successful_count,
            "failed_generations": failed_count,
            "success_rate": round(successful_count / len(requests) * 100, 2),
            "average_time_per_request": round(total_processing_time / len(requests), 3),
            "system_stats_before": system_stats_before,
            "system_stats_after": system_stats_after,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        total_processing_time = time.time() - start_time
        logger.error(f"v11 batch generation failed: {e}")
        performance_tracker.track_error()
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

@router.post("/v11/cache/clear", summary="Clear v11 cache", tags=["copywriting"])
async def clear_cache_v11(api_key: str = Depends(get_api_key)):
    """Clear the v11 engine cache with detailed response."""
    start_time = time.time()
    
    try:
        engine = await get_v11_engine()
        
        # Get cache stats before clearing
        cache_stats_before = engine.intelligent_cache.get_stats()
        
        # Clear cache
        engine.intelligent_cache.clear_all()
        
        # Get cache stats after clearing
        cache_stats_after = engine.intelligent_cache.get_stats()
        
        processing_time = time.time() - start_time
        
        return {
            "status": "cache_cleared",
            "version": "v11",
            "cache_stats_before": cache_stats_before,
            "cache_stats_after": cache_stats_after,
            "processing_time_seconds": round(processing_time, 3),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/v11/cache/stats", summary="Get v11 cache statistics", tags=["copywriting"])
async def get_cache_stats_v11(api_key: str = Depends(get_api_key)):
    """Get detailed cache statistics from the v11 engine."""
    try:
        engine = await get_v11_engine()
        stats = engine.intelligent_cache.get_stats()
        
        # Add cache efficiency metrics
        cache_efficiency = {
            "hit_ratio_percent": round(stats.get("hit_ratio", 0) * 100, 2),
            "miss_ratio_percent": round((1 - stats.get("hit_ratio", 0)) * 100, 2),
            "eviction_rate": stats.get("eviction_rate", 0),
            "memory_usage_mb": round(stats.get("memory_usage", 0) / (1024 * 1024), 2)
        }
        
        return {
            "version": "v11",
            "cache_stats": stats,
            "cache_efficiency": cache_efficiency,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

# New utility endpoints for v11 system
@router.get("/v11/system-info", summary="Get v11 system information", tags=["copywriting"])
async def get_system_info_v11(api_key: str = Depends(get_api_key)):
    """Get comprehensive system information for v11."""
    try:
        engine = await get_v11_engine()
        system_stats = get_system_stats()
        
        # Get engine configuration
        config_info = {
            "max_workers": engine.config.max_workers if hasattr(engine, 'config') else "unknown",
            "batch_size": engine.config.batch_size if hasattr(engine, 'config') else "unknown",
            "cache_size": engine.config.cache_size if hasattr(engine, 'config') else "unknown",
            "gpu_memory_fraction": engine.config.gpu_memory_fraction if hasattr(engine, 'config') else "unknown"
        }
        
        return {
            "version": "v11",
            "system_stats": system_stats,
            "engine_config": config_info,
            "python_version": sys.version,
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")

@router.post("/v11/optimize", summary="Optimize v11 engine configuration", tags=["copywriting"])
async def optimize_v11_engine(
    optimization_params: dict = Body(..., example={
        "max_workers": 64,
        "batch_size": 20,
        "cache_size": 10000,
        "gpu_memory_fraction": 0.8
    }),
    api_key: str = Depends(get_api_key)
):
    """Optimize v11 engine configuration based on current system performance."""
    try:
        engine = await get_v11_engine()
        
        # Apply optimization parameters
        if hasattr(engine, 'config'):
            for key, value in optimization_params.items():
                if hasattr(engine.config, key):
                    setattr(engine.config, key, value)
        
        return {
            "status": "optimization_applied",
            "version": "v11",
            "applied_parameters": optimization_params,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to optimize engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize engine: {str(e)}")

# Legacy endpoints (maintained for backward compatibility)
@router.get("/models", summary="List available copywriting models", tags=["copywriting"])
@cache(expire=60)  # Cache this endpoint for 60 seconds
def list_models():
    """Devuelve los modelos de copywriting disponibles."""
    return {"available_models": AVAILABLE_MODELS}

@router.get("/task-status/{task_id}", summary="Get Celery task status/result", tags=["copywriting"])
def get_task_status(task_id: str):
    """Consulta el estado y resultado de una tarea de copywriting enviada por Celery."""
    result = AsyncResult(task_id)
    if result.state == "PENDING":
        return {"status": result.state}
    elif result.state == "SUCCESS":
        return {"status": result.state, "result": result.result}
    elif result.state == "FAILURE":
        return {"status": result.state, "error": str(result.info)}
    else:
        return {"status": result.state}

@router.post(
    "/batch-status",
    summary="Get status/results for multiple Celery tasks",
    tags=["copywriting"],
    responses={
        200: {"description": "Batch status/result for tasks", "content": {"application/json": {}}},
        401: {"description": "API Key inválida o ausente"},
    },
)
def batch_task_status(
    task_ids: List[str] = Body(..., description="List of Celery task IDs to check status for"),
    api_key: str = Depends(get_api_key)
):
    """
    Devuelve el estado y resultado de múltiples tareas de Celery.
    """
    results = []
    for task_id in task_ids:
        result = AsyncResult(task_id)
        results.append({
            "task_id": task_id,
            "status": result.state,
            "result": result.result if result.state == "SUCCESS" else None,
            "error": str(result.info) if result.state == "FAILURE" else None,
        })
    return {"tasks": results}

@router.post(
    "/generate",
    response_model=CopywritingOutput,
    summary="Genera copywriting con modelo LLM (legacy endpoint)",
    tags=["copywriting"],
    dependencies=[Depends(RateLimiter(times=5, seconds=60))],  # 5 requests per minute per IP
    responses={
        200: {"description": "Copywriting generado exitosamente", "model": CopywritingOutput},
        400: {"description": "Modelo no soportado o input inválido"},
        401: {"description": "API Key inválida o ausente"},
        500: {"description": "Error interno del modelo"},
    },
)
async def generate_copywriting(
    request: CopywritingInput = Body(..., example={
        "product_description": "Zapatos deportivos de alta gama",
        "target_platform": "Instagram",
        "tone": "inspirational",
        "target_audience": "Jóvenes activos",
        "key_points": ["Comodidad", "Estilo", "Durabilidad"],
        "instructions": "Enfatiza la innovación.",
        "restrictions": ["no mencionar precio"],
        "creativity_level": 0.8,
        "language": "es"
    }),
    model_name: str = Query("gpt2", description="Nombre del modelo HuggingFace a usar", enum=AVAILABLE_MODELS),
    service: CopywritingService = Depends(get_service),
    request_obj: Request = None,
    api_key: str = Depends(get_api_key)
):
    # TODO: Log this generation with MLflow for experiment tracking
    logger.info(f"/generate called from {request_obj.client.host if request_obj else 'unknown'} with model={model_name}")
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no soportado.")
    try:
        result = await service.generate(request)
        return result
    except Exception as e:
        logger.error(f"Error in /generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/batch-generate",
    summary="Batch submit copywriting jobs via Celery (legacy endpoint)",
    tags=["copywriting"],
    responses={
        200: {"description": "Batch procesado", "content": {"application/json": {}}},
        400: {"description": "Modelo no soportado o batch demasiado grande"},
        401: {"description": "API Key inválida o ausente"},
        500: {"description": "Error interno"},
    },
)
async def batch_generate_copywriting(
    requests: List[CopywritingInput] = Body(..., example=[{"product_description": "Zapatos...", "target_platform": "Instagram", "tone": "inspirational"}]),
    model_name: str = Query("gpt2", description="HuggingFace model name", enum=AVAILABLE_MODELS),
    wait: bool = Query(False, description="Wait for results (synchronous)"),
    api_key: str = Depends(get_api_key)
):
    """Batch submit copywriting jobs via Celery. Returns task IDs or results if wait=True."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no soportado.")
    if len(requests) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"El batch máximo permitido es {MAX_BATCH_SIZE}.")
    tasks = [generate_copywriting_task.delay(req.dict(), model_name=model_name) for req in requests]
    if wait:
        # Wait for all results (no recomendado para batches grandes)
        try:
            results = [t.get(timeout=120) for t in tasks]
        except Exception as e:
            logger.error(f"Error en batch wait: {e}")
            raise HTTPException(status_code=500, detail="Error procesando el batch: " + str(e))
        return {"results": results}
    else:
        return {"task_ids": [t.id for t in tasks]}

@router.post(
    "/feedback",
    summary="Envía feedback sobre una variante de copywriting (incluye feedback granular por sección e historial de variantes)",
    tags=["copywriting"],
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Feedback recibido"},
        400: {"description": "Input inválido"},
        401: {"description": "API Key inválida o ausente"},
    },
)
async def submit_feedback(
    variant_id: str = Body(..., embed=True, description="ID de la variante a la que se da feedback"),
    feedback: Feedback = Body(..., description="Feedback estructurado sobre la variante"),
    section_feedback: Optional[list[SectionFeedback]] = Body(None, description="Feedback granular por sección (opcional)"),
    variant_history: Optional[CopyVariantHistory] = Body(None, description="Historial de la variante (opcional)"),
    api_key: str = Depends(get_api_key)
):
    """
    Recibe feedback sobre una variante de copywriting. Persiste en SQLite. Ahora soporta feedback granular por sección e historial de variantes.
    """
    logger.info(f"Feedback recibido para variante {variant_id}: {feedback}")
    db = SessionLocal()
    try:
        feedback_data = feedback.dict()
        if section_feedback:
            feedback_data["section_feedback"] = [sf.dict() for sf in section_feedback]
        if variant_history:
            feedback_data["variant_history"] = variant_history.dict()
        db_feedback = FeedbackDB(
            variant_id=variant_id,
            type=feedback.type.value,
            score=feedback.score,
            comments=feedback.comments,
            user_id=feedback.user_id,
            timestamp=datetime.fromisoformat(feedback.timestamp) if feedback.timestamp else datetime.utcnow(),
            raw_json=json.dumps(feedback_data)
        )
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        logger.info(f"Feedback guardado con id={db_feedback.id}")
        return {"status": "accepted", "variant_id": variant_id, "feedback_id": db_feedback.id}
    except Exception as e:
        db.rollback()
        logger.error(f"Error guardando feedback: {e}")
        raise HTTPException(status_code=500, detail="Error guardando feedback")
    finally:
        db.close()

# --- FastAPI-Filter for advanced filtering ---
# pip install fastapi-filter

# FeedbackFilter for advanced filtering
class FeedbackFilter(Filter):
    variant_id: str | None = None
    user_id: str | None = None
    type: str | None = None
    score__gte: float | None = None
    score__lte: float | None = None
    timestamp__gte: str | None = None
    timestamp__lte: str | None = None

    class Constants(Filter.Constants):
        model = FeedbackDB

@router.get(
    "/feedback",
    summary="Consulta feedback guardado (incluye feedback granular por sección e historial de variantes si existe)",
    tags=["copywriting"],
    response_model=List[dict],
    responses={
        200: {"description": "Lista de feedback"},
        401: {"description": "API Key inválida o ausente"},
    },
)
def list_feedback(
    feedback_filter: FeedbackFilter = FilterDepends(FeedbackFilter),
    skip: int = Query(0, ge=0, description="Salto de paginación"),
    limit: int = Query(20, ge=1, le=100, description="Límite de resultados"),
    api_key: str = Depends(get_api_key),
    # Backward compatibility for old query params
    variant_id: str | None = Query(None, description="Filtrar por variant_id"),
    user_id: str | None = Query(None, description="Filtrar por user_id"),
):
    """
    Devuelve feedback guardado, filtrable por cualquier campo, con paginación y filtros avanzados. Incluye feedback granular por sección e historial de variantes si existe.
    """
    db: Session = SessionLocal()
    try:
        query = db.query(FeedbackDB)
        # Apply FastAPI-Filter
        query = feedback_filter.filter(query)
        # Backward compatibility: apply old filters if present
        if variant_id:
            query = query.filter(FeedbackDB.variant_id == variant_id)
        if user_id:
            query = query.filter(FeedbackDB.user_id == user_id)
        results = query.order_by(FeedbackDB.timestamp.desc()).offset(skip).limit(limit).all()
        return [
            {
                "id": f.id,
                "variant_id": f.variant_id,
                "type": f.type,
                "score": f.score,
                "comments": f.comments,
                "user_id": f.user_id,
                "timestamp": f.timestamp.isoformat() if f.timestamp else None,
                "raw_json": f.raw_json,
                "section_feedback": json.loads(f.raw_json).get("section_feedback") if f.raw_json else None,
                "variant_history": json.loads(f.raw_json).get("variant_history") if f.raw_json else None
            }
            for f in results
        ]
    finally:
        db.close()

@router.get(
    "/variant-history/{variant_id}",
    summary="Consulta el historial de una variante de copywriting (si existe)",
    tags=["copywriting"],
    response_model=Optional[dict],
    responses={
        200: {"description": "Historial de la variante (si existe)"},
        404: {"description": "No se encontró historial para ese variant_id"},
        401: {"description": "API Key inválida o ausente"},
    },
)
def get_variant_history(
    variant_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Devuelve el historial de una variante de copywriting, si fue almacenado junto con el feedback.
    """
    db: Session = SessionLocal()
    try:
        feedbacks = db.query(FeedbackDB).filter(FeedbackDB.variant_id == variant_id).order_by(FeedbackDB.timestamp.desc()).all()
        for f in feedbacks:
            raw = f.raw_json
            if raw:
                data = json.loads(raw)
                if "variant_history" in data:
                    return data["variant_history"]
        raise HTTPException(status_code=404, detail="No se encontró historial para ese variant_id")
    finally:
        db.close()

@router.get(
    "/optimization-results/{tracking_id}",
    summary="Consulta los resultados de optimización asociados a un tracking_id (si existen)",
    tags=["copywriting"],
    response_model=Optional[dict],
    responses={
        200: {"description": "Resultados de optimización (si existen)"},
        404: {"description": "No se encontraron resultados para ese tracking_id"},
        401: {"description": "API Key inválida o ausente"},
    },
)
def get_optimization_results(
    tracking_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Devuelve los resultados de optimización asociados a un tracking_id, si fueron almacenados junto con el feedback o el output.
    """
    db: Session = SessionLocal()
    try:
        # Buscar en feedbacks
        feedbacks = db.query(FeedbackDB).filter(FeedbackDB.raw_json.contains(tracking_id)).order_by(FeedbackDB.timestamp.desc()).all()
        for f in feedbacks:
            raw = f.raw_json
            if raw:
                data = json.loads(raw)
                if "optimization_results" in data:
                    return data["optimization_results"]
        # (Opcional) Aquí podrías buscar en outputs generados si los guardas en otra tabla
        raise HTTPException(status_code=404, detail="No se encontraron resultados para ese tracking_id")
    finally:
        db.close()

@router.post(
    "/refactor",
    response_model=CopywritingOutput,
    summary="Refactoriza un texto usando el modelo LLM",
    tags=["copywriting"],
    dependencies=[Depends(RateLimiter(times=5, seconds=60))],
    responses={
        200: {"description": "Texto refactorizado exitosamente", "model": CopywritingOutput},
        400: {"description": "Modelo no soportado o input inválido"},
        401: {"description": "API Key inválida o ausente"},
        500: {"description": "Error interno del modelo"},
    },
)
async def refactor_copywriting(
    text: str = Body(..., embed=True, description="Texto a refactorizar"),
    model_name: str = Query("gpt2", description="Nombre del modelo HuggingFace a usar", enum=AVAILABLE_MODELS),
    service: CopywritingService = Depends(get_service),
    request_obj: Request = None,
    api_key: str = Depends(get_api_key)
):
    """
    Refactoriza un texto usando el modelo LLM, reutilizando la lógica de generación con la instrucción 'refactor'.
    """
    logger.info(f"/refactor called from {request_obj.client.host if request_obj else 'unknown'} with model={model_name}")
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no soportado.")
    try:
        request = CopywritingInput(
            product_description=text,
            target_platform="refactor",
            tone="informative",
            instructions="refactor",
            language="es"
        )
        result = await service.generate(request)
        return result
    except Exception as e:
        logger.error(f"Error in /refactor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/optimiza",
    response_model=CopywritingOutput,
    summary="Optimiza un texto usando el modelo LLM",
    tags=["copywriting"],
    dependencies=[Depends(RateLimiter(times=5, seconds=60))],
    responses={
        200: {"description": "Texto optimizado exitosamente", "model": CopywritingOutput},
        400: {"description": "Modelo no soportado o input inválido"},
        401: {"description": "API Key inválida o ausente"},
        500: {"description": "Error interno del modelo"},
    },
)
async def optimiza_copywriting(
    text: str = Body(..., embed=True, description="Texto a optimizar"),
    model_name: str = Query("gpt2", description="Nombre del modelo HuggingFace a usar", enum=AVAILABLE_MODELS),
    service: CopywritingService = Depends(get_service),
    request_obj: Request = None,
    api_key: str = Depends(get_api_key)
):
    """
    Optimiza un texto usando el modelo LLM, reutilizando la lógica de generación con la instrucción 'optimiza'.
    """
    logger.info(f"/optimiza called from {request_obj.client.host if request_obj else 'unknown'} with model={model_name}")
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no soportado.")
    try:
        request = CopywritingInput(
            product_description=text,
            target_platform="optimiza",
            tone="informative",
            instructions="optimiza",
            language="es"
        )
        result = await service.generate(request)
        return result
    except Exception as e:
        logger.error(f"Error in /optimiza: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/optimiza-con-librerias",
    response_model=CopywritingOutput,
    summary="Optimiza un texto con librerías usando el modelo LLM",
    tags=["copywriting"],
    dependencies=[Depends(RateLimiter(times=5, seconds=60))],
    responses={
        200: {"description": "Texto optimizado con librerías exitosamente", "model": CopywritingOutput},
        400: {"description": "Modelo no soportado o input inválido"},
        401: {"description": "API Key inválida o ausente"},
        500: {"description": "Error interno del modelo"},
    },
)
async def optimiza_con_librerias_copywriting(
    text: str = Body(..., embed=True, description="Texto a optimizar con librerías"),
    model_name: str = Query("gpt2", description="Nombre del modelo HuggingFace a usar", enum=AVAILABLE_MODELS),
    service: CopywritingService = Depends(get_service),
    request_obj: Request = None,
    api_key: str = Depends(get_api_key)
):
    """
    Optimiza un texto usando el modelo LLM, reutilizando la lógica de generación con la instrucción 'optimiza con librerias'.
    """
    logger.info(f"/optimiza-con-librerias called from {request_obj.client.host if request_obj else 'unknown'} with model={model_name}")
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no soportado.")
    try:
        request = CopywritingInput(
            product_description=text,
            target_platform="optimiza-con-librerias",
            tone="informative",
            instructions="optimiza con librerias",
            language="es"
        )
        result = await service.generate(request)
        return result
    except Exception as e:
        logger.error(f"Error in /optimiza-con-librerias: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup on shutdown
@router.on_event("shutdown")
async def cleanup_v11():
    """Cleanup v11 engine on shutdown."""
    try:
        await cleanup_engine()
        logger.info("v11 engine cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during v11 cleanup: {e}") 