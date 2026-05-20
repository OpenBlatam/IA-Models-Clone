"""Cache management endpoints"""
from fastapi import APIRouter, HTTPException
from utils.cache import get_cache

router = APIRouter(prefix="/cache", tags=["Cache"])


@router.get("/stats")
async def get_cache_stats():
    """Get cache statistics"""
    cache = get_cache()
    stats = cache.get_stats()
    return stats


@router.post("/clear")
async def clear_cache():
    """Clear cache"""
    cache = get_cache()
    cache.clear()
    return {
        "status": "success",
        "message": "Cache cleared"
    }

