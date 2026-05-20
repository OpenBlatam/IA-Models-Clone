"""
Cache Router - Cache management endpoints
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from cache import get_cache_stats, clear_cache
except ImportError:
    logging.warning("cache module not available")
    def get_cache_stats(): return {}
    def clear_cache(): pass

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/cache", tags=["Cache"])


@router.post("/clear", response_model=Dict[str, Any])
async def clear_cache_endpoint() -> JSONResponse:
    """Clear system cache"""
    logger.info("Cache clear requested")
    
    try:
        clear_cache()
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Cache cleared successfully",
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.get("/stats", response_model=Dict[str, Any])
async def get_cache_stats_endpoint() -> JSONResponse:
    """Get cache statistics"""
    logger.info("Cache stats requested")
    
    try:
        stats = get_cache_stats()
        return JSONResponse(content={
            "success": True,
            "data": stats,
            "error": None
        })
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache stats")






