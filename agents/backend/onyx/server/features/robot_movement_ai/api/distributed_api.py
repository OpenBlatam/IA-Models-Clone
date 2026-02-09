"""
Distributed API Endpoints
=========================

Endpoints para distributed cache y distributed state.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional
import logging

from ..core.distributed_cache import (
    create_distributed_cache,
    get_distributed_cache,
    CacheStrategy
)
from ..core.distributed_state import (
    create_distributed_state,
    get_distributed_state
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/distributed", tags=["distributed"])


@router.post("/caches")
async def create_cache(
    name: str,
    max_size: int = 1000,
    default_ttl: float = 3600.0,
    strategy: str = "lru"
) -> Dict[str, Any]:
    """Crear cache distribuido."""
    try:
        strategy_enum = CacheStrategy(strategy.lower())
        cache = create_distributed_cache(
            name=name,
            max_size=max_size,
            default_ttl=default_ttl,
            strategy=strategy_enum
        )
        return {
            "name": cache.name,
            "max_size": cache.max_size,
            "strategy": cache.strategy.value
        }
    except Exception as e:
        logger.error(f"Error creating cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/caches/{name}/get")
async def get_cache_value(
    name: str,
    key: str
) -> Dict[str, Any]:
    """Obtener valor del cache."""
    try:
        cache = get_distributed_cache(name)
        if not cache:
            raise HTTPException(status_code=404, detail="Cache not found")
        
        value = cache.get(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        
        return {
            "key": key,
            "value": value
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache value: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/caches/{name}/set")
async def set_cache_value(
    name: str,
    key: str,
    value: Any = Body(...),
    ttl: Optional[float] = None
) -> Dict[str, Any]:
    """Establecer valor en cache."""
    try:
        cache = get_distributed_cache(name)
        if not cache:
            raise HTTPException(status_code=404, detail="Cache not found")
        
        cache.set(key, value, ttl=ttl)
        return {
            "key": key,
            "set": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting cache value: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/caches/{name}/statistics")
async def get_cache_statistics(name: str) -> Dict[str, Any]:
    """Obtener estadísticas del cache."""
    try:
        cache = get_distributed_cache(name)
        if not cache:
            raise HTTPException(status_code=404, detail="Cache not found")
        
        stats = cache.get_statistics()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/states")
async def create_state(name: str) -> Dict[str, Any]:
    """Crear estado distribuido."""
    try:
        state = create_distributed_state(name)
        return {
            "name": state.name
        }
    except Exception as e:
        logger.error(f"Error creating state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/states/{name}/get")
async def get_state_value(
    name: str,
    key: str
) -> Dict[str, Any]:
    """Obtener valor del estado."""
    try:
        state = get_distributed_state(name)
        if not state:
            raise HTTPException(status_code=404, detail="State not found")
        
        value = state.get(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        
        return {
            "key": key,
            "value": value
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting state value: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/states/{name}/set")
async def set_state_value(
    name: str,
    key: str,
    value: Any = Body(...)
) -> Dict[str, Any]:
    """Establecer valor en estado."""
    try:
        state = get_distributed_state(name)
        if not state:
            raise HTTPException(status_code=404, detail="State not found")
        
        entry = state.set(key, value)
        return {
            "key": key,
            "version": entry.version,
            "set": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting state value: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/states/{name}/statistics")
async def get_state_statistics(name: str) -> Dict[str, Any]:
    """Obtener estadísticas del estado."""
    try:
        state = get_distributed_state(name)
        if not state:
            raise HTTPException(status_code=404, detail="State not found")
        
        stats = state.get_statistics()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting state statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






