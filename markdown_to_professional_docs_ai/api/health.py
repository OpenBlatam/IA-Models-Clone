"""Health check endpoints"""
from fastapi import APIRouter
from config import settings
from utils.cache import get_cache
from utils.health_checks import get_health_checker

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from utils.health_checks import get_health_checker
    
    health_checker = get_health_checker()
    health_status = await health_checker.run_checks()
    
    cache = get_cache()
    cache_stats = cache.get_stats()
    
    return {
        **health_status,
        "service": settings.app_name,
        "version": settings.app_version,
        "cache": {
            "hits": cache_stats.get("hits", 0),
            "misses": cache_stats.get("misses", 0),
            "size": cache_stats.get("size", 0)
        }
    }

