"""
Distributed Cache endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.distributed_cache import DistributedCacheService

router = APIRouter()
cache_service = DistributedCacheService()


@router.get("/get/{key}")
async def get_cache(key: str) -> Dict[str, Any]:
    """Obtener valor del cache"""
    try:
        value = cache_service.get(key)
        return {
            "key": key,
            "value": value,
            "found": value is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set")
async def set_cache(
    key: str,
    value: Any,
    ttl: int = 3600
) -> Dict[str, Any]:
    """Establecer valor en cache"""
    try:
        success = cache_service.set(key, value, ttl)
        return {
            "key": key,
            "success": success,
            "ttl": ttl,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{key}")
async def delete_cache(key: str) -> Dict[str, Any]:
    """Eliminar del cache"""
    try:
        success = cache_service.delete(key)
        return {
            "key": key,
            "deleted": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """Obtener estadísticas de cache"""
    try:
        stats = cache_service.get_stats()
        return {
            "total_keys": stats.total_keys,
            "hit_count": stats.hit_count,
            "miss_count": stats.miss_count,
            "hit_rate": stats.hit_rate,
            "memory_usage": stats.memory_usage,
            "evicted_keys": stats.evicted_keys,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




