"""Shipment request handlers"""

from typing import List, Optional

from models.schemas import (
    ShipmentRequest,
    ShipmentResponse,
    ShipmentStatus,
    Location,
)
from repositories.shipment_repository import ShipmentRepository
from domain.shipments import (
    create_shipment_domain,
    get_shipment_domain,
    get_shipments_domain,
    update_shipment_status_domain,
)
from utils.exceptions import NotFoundError
from utils.handler_helpers import (
    get_entity_or_raise,
    update_entity_with_cache_invalidation
)
from utils.metrics import get_metrics_collector


async def handle_create_shipment(
    request: ShipmentRequest,
    repository: ShipmentRepository
) -> ShipmentResponse:
    """Handle shipment creation request"""
    shipment = await create_shipment_domain(request, repository)
    from utils.handler_helpers import create_entity_with_cache
    from utils.constants import DEFAULT_CACHE_TTL
    result = await create_entity_with_cache(
        shipment,
        f"shipment:{shipment.shipment_id}",
        cache_ttl=DEFAULT_CACHE_TTL
    )
    # Record business metric
    get_metrics_collector().record_shipment_created()
    return result


async def handle_get_shipment(
    shipment_id: str,
    repository: ShipmentRepository
) -> ShipmentResponse:
    """Handle get shipment request"""
    from utils.constants import DEFAULT_CACHE_TTL
    return await get_entity_or_raise(
        shipment_id,
        lambda: get_shipment_domain(shipment_id, repository),
        "Shipment",
        cache_key=f"shipment:{shipment_id}",
        cache_ttl=DEFAULT_CACHE_TTL,
        model_class=ShipmentResponse
    )


async def handle_get_shipments(
    repository: ShipmentRepository,
    status: Optional[ShipmentStatus] = None,
    limit: int = 100,
    offset: int = 0
) -> List[ShipmentResponse]:
    """Handle get shipments request"""
    if limit > 1000:
        limit = 1000
    
    return await get_shipments_domain(
        repository=repository,
        status=status,
        limit=limit,
        offset=offset
    )


async def handle_update_shipment_status(
    shipment_id: str,
    status: ShipmentStatus,
    repository: ShipmentRepository,
    location: Optional[Location] = None,
    description: Optional[str] = None
) -> ShipmentResponse:
    """Handle update shipment status request"""
    shipment = await update_shipment_status_domain(
        shipment_id=shipment_id,
        status=status,
        repository=repository,
        location=location,
        description=description
    )
    
    if not shipment:
        raise NotFoundError("Shipment", shipment_id)
    
    await update_entity_with_cache_invalidation(
        shipment,
        f"tracking:{shipment_id}"
    )
    
    return shipment


