"""
Cache router for Multi-Model API
Handles cache management endpoints
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional

from ...api.dependencies import get_cache_instance
from ...core.cache import EnhancedCache

router = APIRouter(prefix="/multi-model", tags=["Cache"])


@router.delete("/cache")
async def clear_cache(
    level: Optional[str] = Query(None, description="Cache level: l1, l2, l3, or all"),
    cache: EnhancedCache = Depends(get_cache_instance)
):
    """
    Clear cache
    
    Args:
        level: Cache level to clear (l1, l2, l3, or all). If not specified, clears all levels.
        
    Returns:
        Success message
    """
    success = await cache.clear(level=level)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear cache")
    return {
        "message": "Cache cleared successfully",
        "level": level or "all"
    }




