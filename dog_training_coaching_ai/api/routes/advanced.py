"""
Advanced Endpoints
==================
Endpoints avanzados para funcionalidades especiales.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, Optional
from pydantic import BaseModel

from ...services.coaching_service import DogTrainingCoach
from ...utils.rate_limiting_advanced import TokenBucket, SlidingWindowRateLimiter
from ...utils.cache_advanced import LRUCache, cache_key_generator
from ...utils.logger import get_logger

router = APIRouter(prefix="/api/v1/advanced", tags=["advanced"])
logger = get_logger(__name__)


class RateLimitCheckRequest(BaseModel):
    """Request para verificar rate limit."""
    key: str
    tokens: int = 1


class CacheStatsRequest(BaseModel):
    """Request para estadísticas de cache."""
    cache_type: Optional[str] = None


# Instancias globales para demostración
token_buckets: Dict[str, TokenBucket] = {}
sliding_windows: Dict[str, SlidingWindowRateLimiter] = {}
caches: Dict[str, LRUCache] = {}


@router.post("/rate-limit/check")
async def check_rate_limit(request: RateLimitCheckRequest) -> Dict[str, Any]:
    """
    Verificar rate limit usando Token Bucket.
    
    Returns:
        Estado del rate limit
    """
    try:
        if request.key not in token_buckets:
            # Crear bucket si no existe (10 tokens, 1 token/segundo)
            token_buckets[request.key] = TokenBucket(capacity=10, refill_rate=1.0)
        
        bucket = token_buckets[request.key]
        allowed = await bucket.consume(request.tokens)
        
        if not allowed:
            wait_time = await bucket.get_wait_time(request.tokens)
            return {
                "allowed": False,
                "wait_time_seconds": wait_time,
                "message": f"Rate limit exceeded. Wait {wait_time:.2f} seconds"
            }
        
        return {
            "allowed": True,
            "message": "Request allowed"
        }
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rate-limit/sliding-window/check")
async def check_sliding_window(
    key: str,
    max_requests: int = 10,
    window_seconds: int = 60
) -> Dict[str, Any]:
    """
    Verificar rate limit usando ventana deslizante.
    
    Returns:
        Estado del rate limit
    """
    try:
        window_key = f"{key}_{max_requests}_{window_seconds}"
        
        if window_key not in sliding_windows:
            sliding_windows[window_key] = SlidingWindowRateLimiter(
                max_requests=max_requests,
                window_seconds=window_seconds
            )
        
        limiter = sliding_windows[window_key]
        allowed, wait_time = await limiter.is_allowed(key)
        stats = await limiter.get_stats(key)
        
        return {
            "allowed": allowed,
            "wait_time_seconds": wait_time,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error checking sliding window: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats(cache_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Obtener estadísticas de caches.
    
    Returns:
        Estadísticas de caches
    """
    try:
        if cache_name:
            if cache_name in caches:
                return {
                    cache_name: caches[cache_name].get_stats()
                }
            else:
                raise HTTPException(status_code=404, detail=f"Cache '{cache_name}' not found")
        
        # Retornar todas las estadísticas
        all_stats = {
            name: cache.get_stats()
            for name, cache in caches.items()
        }
        
        return {
            "caches": all_stats,
            "total_caches": len(caches)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/create")
async def create_cache(
    cache_name: str,
    maxsize: int = 128
) -> Dict[str, Any]:
    """
    Crear un nuevo cache.
    
    Returns:
        Información del cache creado
    """
    try:
        if cache_name in caches:
            raise HTTPException(status_code=400, detail=f"Cache '{cache_name}' already exists")
        
        caches[cache_name] = LRUCache(maxsize=maxsize)
        
        return {
            "success": True,
            "cache_name": cache_name,
            "maxsize": maxsize,
            "message": f"Cache '{cache_name}' created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/{cache_name}")
async def delete_cache(cache_name: str) -> Dict[str, Any]:
    """
    Eliminar un cache.
    
    Returns:
        Confirmación de eliminación
    """
    try:
        if cache_name not in caches:
            raise HTTPException(status_code=404, detail=f"Cache '{cache_name}' not found")
        
        caches[cache_name].clear()
        del caches[cache_name]
        
        return {
            "success": True,
            "message": f"Cache '{cache_name}' deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

