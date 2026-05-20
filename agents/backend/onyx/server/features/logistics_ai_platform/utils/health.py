"""Health check utilities"""

from typing import Dict, Any
from datetime import datetime

from utils.cache import cache_service
from utils.logger import logger


async def check_cache_health() -> Dict[str, Any]:
    """Check cache service health"""
    try:
        # Try to set and get a test value
        test_key = "health_check_test"
        test_value = "ok"
        await cache_service.set(test_key, test_value, ttl=1)
        cached = await cache_service.get(test_key)
        
        if cached == test_value:
            return {
                "status": "healthy",
                "service": "cache",
                "connected": True
            }
        else:
            return {
                "status": "unhealthy",
                "service": "cache",
                "connected": True,
                "issue": "Cache read/write mismatch"
            }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "cache",
            "connected": False,
            "error": str(e)
        }


async def check_database_health() -> Dict[str, Any]:
    """Check database health"""
    # TODO: Implement actual database health check when database is integrated
    return {
        "status": "healthy",
        "service": "database",
        "connected": True,
        "note": "In-memory storage currently"
    }


async def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status"""
    cache_health = await check_cache_health()
    db_health = await check_database_health()
    
    overall_status = "healthy"
    if cache_health["status"] != "healthy" or db_health["status"] != "healthy":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "cache": cache_health,
            "database": db_health,
        },
        "version": "1.0.0"
    }








