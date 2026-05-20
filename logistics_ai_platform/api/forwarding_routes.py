"""
Forwarding routes - aggregates all forwarding sub-routes

This module aggregates all forwarding-related routes including:
- Quotes: Create and manage freight quotes
- Bookings: Create and manage bookings from quotes
- Shipments: Manage shipment lifecycle
- Containers: Track and manage container status

All routes are prefixed with `/forwarding` and grouped under the "Forwarding" tag.

Features:
- Automatic router registration with error handling
- Health check endpoint
- Service information endpoint
- Caching for static information
- Comprehensive logging
- OpenAPI documentation
"""

from datetime import datetime
from typing import List, Tuple
from fastapi import APIRouter, Request, status, Query
from fastapi.responses import JSONResponse

from api.forwarding_service_info import (
    get_service_info,
    get_health_status,
    get_service_metrics,
    FORWARDING_SERVICE_NAME,
    FORWARDING_SERVICE_VERSION,
    get_routers,
)
from utils.exceptions import ServiceUnavailableError
from utils.logger import logger

# Create main forwarding router
router = APIRouter(
    prefix="/forwarding",
    tags=["Forwarding"],
    responses={
        404: {"description": "Not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)


def _register_routers() -> None:
    """
    Register all forwarding sub-routers.
    
    This function safely registers all sub-routers and logs any issues.
    If a router fails to load, it logs the error but continues with others.
    """
    routers_to_register = get_routers()
    registered_count = 0
    
    for sub_router, name in routers_to_register:
        try:
            if sub_router is None:
                logger.warning(f"Router '{name}' is None, skipping registration")
                continue
            
            router.include_router(sub_router)
            registered_count += 1
            logger.debug(f"Successfully registered '{name}' router")
        except Exception as e:
            logger.error(
                f"Failed to register '{name}' router: {str(e)}",
                exc_info=True
            )
    
    logger.info(
        f"Forwarding routes initialized: {registered_count}/{len(routers_to_register)} "
        f"routers registered successfully"
    )


# Register all sub-routers on module import
_register_routers()


@router.get(
    "",
    summary="Get forwarding routes information",
    description="Returns information about available forwarding routes",
    tags=["Forwarding"],
    response_description="Service information and available routes",
)
async def get_forwarding_info(request: Request) -> JSONResponse:
    """
    Get information about available forwarding routes.
    
    Returns a summary of all available forwarding endpoints grouped by category.
    This endpoint is cached for 1 hour to reduce load.
    
    Returns:
        JSONResponse: Service information including routes, version, and documentation links
    """
    return await get_service_info(request)


@router.get(
    "/health",
    summary="Health check for forwarding service",
    description="Returns the health status of the forwarding service",
    tags=["Forwarding"],
    response_description="Service health status",
)
async def health_check(request: Request) -> JSONResponse:
    """
    Health check endpoint for the forwarding service.
    
    Returns the health status and basic service information.
    This endpoint performs basic checks on all sub-routers and dependencies.
    
    Returns:
        JSONResponse: Health status with router information
        
    Raises:
        ServiceUnavailableError: If service is completely unavailable
    """
    try:
        health_info, status_code = await get_health_status()
        return JSONResponse(content=health_info, status_code=status_code)
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise ServiceUnavailableError(
            detail=f"Health check failed: {str(e)}",
            retry_after=30
        )


@router.get(
    "/metrics",
    summary="Get forwarding service metrics",
    description="Returns metrics about the forwarding service",
    tags=["Forwarding"],
    response_description="Service metrics",
    responses={
        200: {"description": "Metrics retrieved successfully"},
        500: {"description": "Internal server error"},
    },
)
async def get_metrics(
    include_cache: bool = Query(
        default=False,
        description="Include cache statistics in metrics"
    )
) -> JSONResponse:
    """
    Get metrics about the forwarding service.
    
    Returns basic metrics including router counts and service information.
    This is a simple metrics endpoint. For production, consider integrating
    with Prometheus or similar monitoring systems.
    
    Returns:
        JSONResponse: Service metrics with router and endpoint information
    """
    try:
        metrics = await get_service_metrics(include_cache=include_cache)
        logger.debug("Metrics retrieved successfully")
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        return JSONResponse(
            content={
                "error": "Failed to retrieve metrics",
                "service": FORWARDING_SERVICE_NAME,
                "timestamp": datetime.utcnow().isoformat(),
                "message": str(e)
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get(
    "/version",
    summary="Get forwarding service version",
    description="Returns the current version of the forwarding service",
    tags=["Forwarding"],
    response_description="Service version information",
)
async def get_version() -> JSONResponse:
    """
    Get the current version of the forwarding service.
    
    Returns version information including API version, service name,
    and compatibility information.
    
    Returns:
        JSONResponse: Version information
    """
    return JSONResponse(
        content={
            "service": FORWARDING_SERVICE_NAME,
            "version": FORWARDING_SERVICE_VERSION,
            "api_version": "v1",
            "timestamp": datetime.utcnow().isoformat(),
            "compatibility": {
                "min_client_version": "1.0.0",
                "deprecated": False,
            },
        }
    )


@router.get(
    "/status",
    summary="Get forwarding service status",
    description="Returns a quick status check of the forwarding service",
    tags=["Forwarding"],
    response_description="Service status",
)
async def get_status() -> JSONResponse:
    """
    Get a quick status check of the forwarding service.
    
    This is a lightweight endpoint that returns basic status information
    without performing deep health checks. Use /health for comprehensive checks.
    
    Returns:
        JSONResponse: Quick status information
    """
    routers_list = [r for r, _ in get_routers()]
    active_routers = sum(1 for r in routers_list if r is not None)
    total_routers = len(routers_list)
    
    return JSONResponse(
        content={
            "status": "operational" if active_routers == total_routers else "degraded",
            "service": FORWARDING_SERVICE_NAME,
            "version": FORWARDING_SERVICE_VERSION,
            "active_routers": active_routers,
            "total_routers": total_routers,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
