"""
Public Tracking Routes

This module provides public API endpoints for tracking shipments and containers.
No account or authentication required.

Supports tracking by:
- Tracking Number
- Container Number
- House Bill Number
- Master Bill Number
- Shipment ID
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Query, Path, status
from fastapi.responses import JSONResponse

from models.schemas import TrackingEvent
from utils.dependencies import (
    get_shipment_repository,
    get_container_repository,
    TrackingServiceDep,
)
from repositories.shipment_repository import ShipmentRepository
from repositories.container_repository import ContainerRepository
from handlers.tracking_handlers import handle_public_tracking
from utils.exceptions import NotFoundError, ValidationError
from utils.cache import cache_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/tracking",
    tags=["Public Tracking"],
    responses={
        404: {"description": "Shipment or container not found"},
        422: {"description": "Validation error"},
    }
)


@router.get(
    "/{identifier}",
    summary="Track Shipment or Container (Public - No Account Required)",
    description="""
    Public tracking endpoint - No account or authentication required.
    
    Track shipments or containers using any of the following identifiers:
    - **Tracking Number** (e.g., TRK123456789)
    - **Container Number** (e.g., CONT1234567)
    - **House Bill Number** (e.g., HBL20241114ABC123)
    - **Master Bill Number** (e.g., MBL20241114XYZ789)
    - **Shipment ID** (e.g., S12345678)
    
    Returns comprehensive tracking information including:
    - Current status and location
    - Tracking event history
    - Estimated arrival time
    - Container information (if applicable)
    - Next milestone
    """,
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK
)
async def track_shipment_or_container(
    identifier: str = Path(
        ...,
        description="Tracking number, container number, house bill, master bill, or shipment ID",
        example="TRK123456789"
    ),
    shipment_repo: ShipmentRepository = Depends(get_shipment_repository),
    container_repo: ContainerRepository = Depends(get_container_repository)
) -> Dict[str, Any]:
    """
    Track shipment or container by identifier
    
    Args:
        identifier: Tracking number, container number, house bill, master bill, or shipment ID
        shipment_repo: Injected shipment repository
        container_repo: Injected container repository
        
    Returns:
        Comprehensive tracking information
        
    Raises:
        NotFoundError: If shipment or container not found
        ValidationError: If identifier is invalid
    """
    if not identifier or not identifier.strip():
        raise ValidationError("Tracking identifier is required", field="identifier")
    
    return await handle_public_tracking(
        identifier=identifier.strip().upper(),
        shipment_repo=shipment_repo,
        container_repo=container_repo
    )


@router.get(
    "/shipment/{shipment_id}",
    summary="Get tracking info for shipment",
    description="Get comprehensive tracking information for a shipment by ID",
    response_model=Dict[str, Any]
)
async def get_tracking_info(
    shipment_id: str = Path(..., description="Shipment ID"),
    tracking_service: TrackingServiceDep = None
) -> Dict[str, Any]:
    """Get comprehensive tracking information for a shipment"""
    if not shipment_id or not shipment_id.strip():
        raise ValidationError("Shipment ID is required", field="shipment_id")
    
    # Try cache first
    cache_key = f"tracking:{shipment_id}"
    try:
        cached = await cache_service.get(cache_key)
        if cached:
            logger.debug(f"Tracking info for {shipment_id} retrieved from cache")
            return cached
    except Exception as e:
        logger.warning(f"Cache retrieval failed for {shipment_id}: {e}")
    
    if not tracking_service:
        from utils.dependencies import get_tracking_service
        tracking_service = get_tracking_service()
    
    tracking_info = await tracking_service.get_tracking_info(shipment_id=shipment_id)
    if not tracking_info.get("shipment"):
        raise NotFoundError("Shipment", shipment_id)
    
    # Cache for 5 minutes
    try:
        await cache_service.set(cache_key, tracking_info, ttl=300)
    except Exception as e:
        logger.warning(f"Failed to cache tracking info for {shipment_id}: {e}")
    
    return tracking_info


@router.get(
    "/container/{container_id}",
    summary="Get tracking info for container",
    description="Get tracking information for a container by ID",
    response_model=Dict[str, Any]
)
async def get_container_tracking(
    container_id: str = Path(..., description="Container ID"),
    tracking_service: TrackingServiceDep = None
) -> Dict[str, Any]:
    """Get tracking information for a container"""
    if not container_id or not container_id.strip():
        raise ValidationError("Container ID is required", field="container_id")
    
    if not tracking_service:
        from utils.dependencies import get_tracking_service
        tracking_service = get_tracking_service()
    
    tracking_info = await tracking_service.get_tracking_info(container_id=container_id)
    if not tracking_info.get("containers"):
        raise NotFoundError("Container", container_id)
    return tracking_info


@router.get(
    "/shipment/{shipment_id}/history",
    summary="Get tracking history",
    description="Get complete tracking event history for a shipment",
    response_model=list[TrackingEvent]
)
async def get_tracking_history(
    shipment_id: str = Path(..., description="Shipment ID"),
    tracking_service: TrackingServiceDep = None
) -> list[TrackingEvent]:
    """Get tracking history for a shipment"""
    if not shipment_id or not shipment_id.strip():
        raise ValidationError("Shipment ID is required", field="shipment_id")
    
    if not tracking_service:
        from utils.dependencies import get_tracking_service
        tracking_service = get_tracking_service()
    
    history = await tracking_service.get_tracking_history(shipment_id)
    return history


@router.get(
    "/summary",
    summary="Get tracking summary",
    description="Get summary of shipments (departing, arriving, in transit)",
    response_model=Dict[str, Any]
)
async def get_tracking_summary(
    tracking_service: TrackingServiceDep = None
) -> Dict[str, Any]:
    """Get tracking summary (departing, arriving, in transit)"""
    # Try cache first
    try:
        cached = await cache_service.get("tracking:summary")
        if cached:
            return cached
    except Exception as e:
        logger.warning(f"Cache retrieval failed for summary: {e}")
    
    if not tracking_service:
        from utils.dependencies import get_tracking_service
        tracking_service = get_tracking_service()
    
    departing = await tracking_service.get_departing_this_week()
    arriving = await tracking_service.get_arriving_this_week()
    in_transit = await tracking_service.get_in_transit()
    
    summary = {
        "departing_this_week": len(departing),
        "arriving_this_week": len(arriving),
        "in_transit": len(in_transit),
        "departing_shipments": [
            {
                "shipment_id": s.shipment_id,
                "tracking_number": s.tracking_number,
                "status": s.status.value,
                "estimated_departure": s.estimated_departure.isoformat() if s.estimated_departure else None,
            }
            for s in departing
        ],
        "arriving_shipments": [
            {
                "shipment_id": s.shipment_id,
                "tracking_number": s.tracking_number,
                "status": s.status.value,
                "estimated_arrival": s.estimated_arrival.isoformat() if s.estimated_arrival else None,
            }
            for s in arriving
        ],
        "in_transit_shipments": [
            {
                "shipment_id": s.shipment_id,
                "tracking_number": s.tracking_number,
                "status": s.status.value,
                "current_location": s.tracking_events[-1].location.model_dump() if s.tracking_events else None,
            }
            for s in in_transit
        ]
    }
    
    # Cache for 1 minute
    try:
        await cache_service.set("tracking:summary", summary, ttl=60)
    except Exception as e:
        logger.warning(f"Failed to cache tracking summary: {e}")
    
    return summary
