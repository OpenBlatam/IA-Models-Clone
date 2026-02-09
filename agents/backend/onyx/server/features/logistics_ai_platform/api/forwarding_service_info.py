"""
Forwarding service information utilities

This module provides utilities for generating service information,
health checks, and metrics for the forwarding service.
"""

import asyncio
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import status

from api.quotes import router as quotes_router
from api.bookings import router as bookings_router
from api.shipments import router as shipments_router
from api.containers import router as containers_router
from utils.cache.helpers import get_cached_or_fetch
from utils.logger import logger
from utils.constants import SERVICE_INFO_CACHE_TTL

FORWARDING_SERVICE_VERSION = "1.0.0"
FORWARDING_SERVICE_NAME = "Forwarding Service"


def get_routers() -> List[Tuple[Any, str]]:
    """Get list of forwarding routers"""
    return [
        (quotes_router, "Quotes"),
        (bookings_router, "Bookings"),
        (shipments_router, "Shipments"),
        (containers_router, "Containers"),
    ]


async def build_service_info(request: Request) -> Dict[str, Any]:
    """Build service information dictionary"""
    return {
        "service": FORWARDING_SERVICE_NAME,
        "version": FORWARDING_SERVICE_VERSION,
        "timestamp": datetime.now().isoformat(),
        "routes": {
            "quotes": {
                "base_path": "/forwarding/quotes",
                "description": "Create and manage freight quotes",
                "endpoints": [
                    "POST /forwarding/quotes - Create a new quote",
                    "GET /forwarding/quotes/{quote_id} - Get quote by ID",
                ],
            },
            "bookings": {
                "base_path": "/forwarding/bookings",
                "description": "Create and manage bookings from quotes",
                "endpoints": [
                    "POST /forwarding/bookings - Create a new booking",
                    "GET /forwarding/bookings/{booking_id} - Get booking by ID",
                ],
            },
            "shipments": {
                "base_path": "/forwarding/shipments",
                "description": "Manage shipment lifecycle",
                "endpoints": [
                    "POST /forwarding/shipments - Create a new shipment",
                    "GET /forwarding/shipments - List shipments with filters",
                    "GET /forwarding/shipments/{shipment_id} - Get shipment by ID",
                    "PATCH /forwarding/shipments/{shipment_id}/status - Update shipment status",
                ],
            },
            "containers": {
                "base_path": "/forwarding/containers",
                "description": "Track and manage container status",
                "endpoints": [
                    "POST /forwarding/containers - Create a new container",
                    "GET /forwarding/containers/{container_id} - Get container by ID",
                    "GET /forwarding/containers/shipment/{shipment_id} - Get containers by shipment",
                    "PATCH /forwarding/containers/{container_id}/status - Update container status",
                ],
            },
        },
        "documentation": {
            "swagger_ui": f"{request.base_url}docs",
            "redoc": f"{request.base_url}redoc",
            "openapi_json": f"{request.base_url}openapi.json",
        },
        "management_endpoints": {
            "health": f"{request.base_url}forwarding/health",
            "metrics": f"{request.base_url}forwarding/metrics",
            "services_status": f"{request.base_url}forwarding/services/status",
            "validate": f"{request.base_url}forwarding/validate",
            "version": f"{request.base_url}forwarding/version",
            "stats": f"{request.base_url}forwarding/stats"
        },
        "health_check": f"{request.base_url}forwarding/health",
    }


async def get_service_info(request: Request) -> JSONResponse:
    """Get forwarding service information with caching"""
    cache_key = "forwarding:info"
    
    info = await get_cached_or_fetch(
        cache_key,
        lambda: build_service_info(request),
        ttl=SERVICE_INFO_CACHE_TTL
    )
    
    return JSONResponse(content=info)


async def check_router_health() -> Dict[str, str]:
    """Check health of all routers"""
    routers_status: Dict[str, str] = {}
    routers_to_check = get_routers()
    
    for name, router_instance in routers_to_check:
        try:
            if router_instance is None:
                routers_status[name] = "unavailable"
                logger.warning(f"Router '{name}' is unavailable")
            else:
                routers_status[name] = "active"
        except Exception as e:
            logger.error(f"Error checking {name} router: {e}", exc_info=True)
            routers_status[name] = "error"
    
    return routers_status


async def check_cache_health() -> str:
    """Check cache service health"""
    from utils.cache import cache_service
    
    try:
        test_key = "health:check"
        await cache_service.set(test_key, "ok", ttl=1)
        cached = await cache_service.get(test_key)
        status = "healthy" if cached == "ok" else "degraded"
        await cache_service.delete(test_key)
        return status
    except Exception as e:
        logger.warning(f"Cache health check failed: {e}")
        return "unhealthy"


async def get_health_status() -> Tuple[Dict[str, Any], int]:
    """Get comprehensive health status (parallel checks)"""
    start_time = time.time()
    
    routers_status, cache_status = await asyncio.gather(
        check_router_health(),
        check_cache_health(),
        return_exceptions=True
    )
    
    if isinstance(routers_status, Exception):
        routers_status = {}
    if isinstance(cache_status, Exception):
        cache_status = "unknown"
    
    overall_status = "healthy"
    if any(status == "error" for status in routers_status.values()):
        overall_status = "unhealthy"
    elif any(status == "unavailable" for status in routers_status.values()) or cache_status == "unhealthy":
        overall_status = "degraded"
    elif cache_status == "degraded":
        overall_status = "degraded"
    
    duration = time.time() - start_time
    
    health_info = {
        "status": overall_status,
        "service": FORWARDING_SERVICE_NAME,
        "version": FORWARDING_SERVICE_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "routers": routers_status,
        "cache": cache_status,
        "response_time_ms": round(duration * 1000, 2),
        "uptime": "N/A",
    }
    
    status_code = status.HTTP_200_OK
    if overall_status == "unhealthy":
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return health_info, status_code


async def get_service_metrics(include_cache: bool = False) -> Dict[str, Any]:
    """Get service metrics"""
    routers_list = [router for router, _ in get_routers()]
    registered_count = len([r for r in routers_list if r is not None])
    
    metrics: Dict[str, Any] = {
        "service": FORWARDING_SERVICE_NAME,
        "version": FORWARDING_SERVICE_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "routers": {
            "total": 4,
            "registered": registered_count,
            "status": "healthy" if registered_count == 4 else "degraded"
        },
        "endpoints": {
            "quotes": 2,
            "bookings": 3,
            "shipments": 4,
            "containers": 5,
        },
        "total_endpoints": 14,
    }
    
    if include_cache:
        from utils.cache import cache_service
        metrics["cache"] = cache_service.get_stats()
    
    return metrics

