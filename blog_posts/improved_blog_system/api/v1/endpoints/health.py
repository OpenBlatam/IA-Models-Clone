"""
Health check API endpoints
"""

from fastapi import APIRouter, Depends
from datetime import datetime

from ....models.schemas import HealthResponse
from ....config.database import get_db_session
from ....core.caching import CacheService
from ....api.dependencies import CacheServiceDep
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check(
    db_session: AsyncSession = Depends(get_db_session),
    cache_service: CacheServiceDep = Depends()
):
    """Comprehensive health check endpoint."""
    try:
        # Check database connection
        db_healthy = await check_database_health(db_session)
        
        # Check cache connection
        cache_healthy = await check_cache_health(cache_service)
        
        # Determine overall health
        overall_healthy = db_healthy and cache_healthy
        
        return HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            timestamp=datetime.now(),
            services={
                "database": db_healthy,
                "cache": cache_healthy
            },
            version="1.0.0"
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            services={
                "database": False,
                "cache": False
            },
            version="1.0.0"
        )


@router.get("/simple")
async def simple_health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }


async def check_database_health(session: AsyncSession) -> bool:
    """Check database connection health."""
    try:
        # Simple query to test database connection
        await session.execute("SELECT 1")
        return True
    except Exception:
        return False


async def check_cache_health(cache_service: CacheService) -> bool:
    """Check cache connection health."""
    try:
        # Test cache connection
        test_key = "health_check_test"
        await cache_service.set(test_key, "test", ttl=10)
        result = await cache_service.get(test_key)
        await cache_service.delete(test_key)
        return result == "test"
    except Exception:
        return False






























