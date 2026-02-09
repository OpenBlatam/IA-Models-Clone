"""
Performance Router - Performance monitoring and optimization endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any

from ...core.async_inference_engine import AsyncInferenceEngine
from ...core.performance_optimizer import PerformanceOptimizer
from ...core.connection_pool import ConnectionPool
from ...core.ml_model_manager import MLModelManager
from ...api.services_locator import get_service
from ...utils.logger import logger

router = APIRouter(prefix="/dermatology", tags=["performance"])

# Global instances
_async_engine: Optional[AsyncInferenceEngine] = None
_perf_optimizer: Optional[PerformanceOptimizer] = None


def get_async_engine() -> AsyncInferenceEngine:
    """Get or create async inference engine"""
    global _async_engine
    if _async_engine is None:
        _async_engine = AsyncInferenceEngine(max_workers=4)
    return _async_engine


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create performance optimizer"""
    global _perf_optimizer
    if _perf_optimizer is None:
        _perf_optimizer = PerformanceOptimizer()
    return _perf_optimizer


def get_ml_model_manager() -> MLModelManager:
    """Get ML model manager from service locator or create new"""
    try:
        return get_service("ml_model_manager")
    except Exception:
        from ...core.ml_model_manager import MLModelManager
        return MLModelManager()


@router.get("/performance/stats")
async def get_performance_stats():
    """Obtiene estadísticas de rendimiento"""
    try:
        stats = {}
        
        # Async engine stats
        try:
            engine = get_async_engine()
            stats["async_engine"] = engine.get_stats()
        except Exception:
            pass
        
        # ML model manager stats
        try:
            manager = get_ml_model_manager()
            stats["ml_models"] = manager.get_stats()
        except Exception:
            pass
        
        # Performance optimizer stats
        try:
            optimizer = get_performance_optimizer()
            cache_stats = {}
            for name, cache in optimizer.caches.items():
                cache_stats[name] = {
                    "size": len(cache.cache),
                    "maxsize": cache.maxsize
                }
            stats["caches"] = cache_stats
        except Exception:
            pass
        
        return JSONResponse(content={"success": True, "stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/performance/cache/clear")
async def clear_cache(cache_name: Optional[str] = Query(None)):
    """Limpia caché"""
    try:
        optimizer = get_performance_optimizer()
        if cache_name:
            if cache_name in optimizer.caches:
                optimizer.caches[cache_name].clear()
        else:
            for cache in optimizer.caches.values():
                cache.clear()
        
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/performance/models/prefetch")
async def prefetch_model(model_id: str):
    """Prefetch a model in background"""
    try:
        manager = get_ml_model_manager()
        manager.prefetch_model(model_id)
        return JSONResponse(content={"success": True, "model_id": model_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

