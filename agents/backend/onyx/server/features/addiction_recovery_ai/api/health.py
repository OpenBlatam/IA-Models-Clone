"""
Health check endpoints with detailed status
"""

from fastapi import APIRouter, status
from datetime import datetime
from typing import Dict, Any
import sys
import platform
import time
import logging

from config.app_config import get_config

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)

_start_time = time.time()


async def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0]
    }


async def check_database() -> Dict[str, Any]:
    """Check database connectivity"""
    config = get_config()
    start_time = time.time()
    
    if not config.database_url:
        return {
            "status": "not_configured",
            "response_time_ms": 0
        }
    
    try:
        if config.database_url.startswith("postgresql") or config.database_url.startswith("postgres"):
            import asyncpg
            try:
                conn = await asyncpg.connect(config.database_url, timeout=5)
                await conn.execute("SELECT 1")
                await conn.close()
                response_time = (time.time() - start_time) * 1000
                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time, 2)
                }
            except Exception as e:
                logger.error(f"Database check failed: {str(e)}")
                return {
                    "status": "unhealthy",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "error": str(e)
                }
        elif config.database_url.startswith("sqlite"):
            import aiosqlite
            try:
                async with aiosqlite.connect(config.database_url.replace("sqlite:///", "")) as db:
                    await db.execute("SELECT 1")
                response_time = (time.time() - start_time) * 1000
                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time, 2)
                }
            except Exception as e:
                logger.error(f"Database check failed: {str(e)}")
                return {
                    "status": "unhealthy",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "error": str(e)
                }
        else:
            return {
                "status": "unknown",
                "response_time_ms": 0
            }
    except ImportError:
        return {
            "status": "driver_not_available",
            "response_time_ms": 0
        }


async def check_cache() -> Dict[str, Any]:
    """Check cache connectivity"""
    config = get_config()
    start_time = time.time()
    
    if not config.cache_enabled:
        return {
            "status": "disabled",
            "response_time_ms": 0
        }
    
    try:
        from infrastructure.cache import RedisCacheService, InMemoryCacheService
        from config.aws_settings import get_aws_settings
        
        aws_settings = get_aws_settings()
        if aws_settings.redis_endpoint:
            cache_service = RedisCacheService()
            try:
                test_key = "__health_check__"
                await cache_service.set(test_key, "ok", ttl=1)
                result = await cache_service.get(test_key)
                await cache_service.delete(test_key)
                response_time = (time.time() - start_time) * 1000
                return {
                    "status": "healthy" if result == "ok" else "unhealthy",
                    "response_time_ms": round(response_time, 2)
                }
            except Exception as e:
                logger.error(f"Cache check failed: {str(e)}")
                return {
                    "status": "unhealthy",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "error": str(e)
                }
        else:
            cache_service = InMemoryCacheService()
            test_key = "__health_check__"
            await cache_service.set(test_key, "ok", ttl=1)
            result = await cache_service.get(test_key)
            response_time = (time.time() - start_time) * 1000
            return {
                "status": "healthy" if result == "ok" else "unhealthy",
                "response_time_ms": round(response_time, 2),
                "type": "in_memory"
            }
    except ImportError:
        return {
            "status": "not_available",
            "response_time_ms": 0
        }
    except Exception as e:
        logger.error(f"Cache check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "error": str(e)
        }


async def get_service_status() -> Dict[str, Any]:
    """Get overall service status"""
    database_status = await check_database()
    cache_status = await check_cache()
    
    all_healthy = (
        database_status["status"] == "healthy" and
        cache_status["status"] == "healthy"
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": database_status,
            "cache": cache_status
        }
    }


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Basic health check"
)
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint
    
    Returns simple health status
    """
    return {
        "status": "healthy",
        "service": "Addiction Recovery AI",
        "version": "3.3.0",
        "timestamp": datetime.now().isoformat()
    }


@router.get(
    "/detailed",
    status_code=status.HTTP_200_OK,
    summary="Detailed health check"
)
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with system information
    
    Returns comprehensive health status including:
    - Service status
    - Database connectivity
    - Cache connectivity
    - System information
    """
    service_status = await get_service_status()
    system_info = await get_system_info()
    
    uptime_seconds = int(time.time() - _start_time)
    
    return {
        **service_status,
        "system": system_info,
        "uptime_seconds": uptime_seconds,
        "uptime_formatted": f"{uptime_seconds // 86400}d {(uptime_seconds % 86400) // 3600}h {(uptime_seconds % 3600) // 60}m {uptime_seconds % 60}s"
    }


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe"
)
async def readiness_check() -> Dict[str, Any]:
    """
    Kubernetes readiness probe
    
    Checks if service is ready to accept traffic
    """
    service_status = await get_service_status()
    
    is_ready = service_status["status"] == "healthy"
    
    if not is_ready:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready"
        )
    
    return {
        "status": "ready",
        "timestamp": datetime.now().isoformat()
    }


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe"
)
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes liveness probe
    
    Checks if service is alive
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }

