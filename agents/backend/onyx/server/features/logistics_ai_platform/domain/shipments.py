"""Shipment domain logic - pure functions"""

from typing import List, Optional
from datetime import datetime

from models.schemas import (
    ShipmentRequest,
    ShipmentResponse,
    ShipmentStatus,
    TrackingEvent,
    Location,
    QuoteResponse,
    QuoteOption,
)
from repositories.shipment_repository import ShipmentRepository
from business_logic.shipment_logic import (
    generate_shipment_id,
    generate_shipment_reference,
    create_initial_tracking_event,
)
from validators.shipment_validators import validate_shipment_request
from utils.logger import logger


async def create_shipment_domain(
    request: ShipmentRequest,
    repository: ShipmentRepository
) -> ShipmentResponse:
    """Create shipment - pure domain function"""
    validate_shipment_request(request)
    
    from factories.shipment_factory import build_shipment_response
    shipment = build_shipment_response(request)
    
    await repository.save(shipment)
    logger.info(f"Shipment created: {shipment.shipment_id}")
    
    return shipment


async def create_shipment_from_booking_domain(
    booking_id: Optional[str],
    quote: QuoteResponse,
    option: QuoteOption,
    shipment_repo: ShipmentRepository
) -> ShipmentResponse:
    """Create shipment from booking - pure domain function"""
    request = ShipmentRequest(
        booking_id=booking_id,
        origin=quote.origin,
        destination=quote.destination,
        cargo=quote.cargo,
        transportation_mode=option.transportation_mode,
        carrier=option.carrier
    )
    
    shipment = await create_shipment_domain(request, shipment_repo)
    shipment.status = ShipmentStatus.BOOKED
    shipment.estimated_departure = option.estimated_departure
    shipment.estimated_arrival = option.estimated_arrival
    
    # Add initial tracking event
    initial_event = create_initial_tracking_event(quote.origin)
    shipment.tracking_events.append(initial_event)
    
    await shipment_repo.save(shipment)
    return shipment


async def get_shipment_domain(
    shipment_id: str,
    repository: ShipmentRepository
) -> Optional[ShipmentResponse]:
    """Get shipment - pure domain function"""
    return await repository.find_by_id(shipment_id)


async def get_shipments_domain(
    repository: ShipmentRepository,
    status: Optional[ShipmentStatus] = None,
    limit: int = 100,
    offset: int = 0
) -> List[ShipmentResponse]:
    """Get shipments - pure domain function"""
    return await repository.find_all(status=status, limit=limit, offset=offset)


async def update_shipment_status_domain(
    shipment_id: str,
    status: ShipmentStatus,
    repository: ShipmentRepository,
    location: Optional[Location] = None,
    description: Optional[str] = None
) -> Optional[ShipmentResponse]:
    """Update shipment status - pure domain function"""
    shipment = await repository.find_by_id(shipment_id)
    if not shipment:
        return None
    
    shipment.status = status
    shipment.updated_at = datetime.now()
    
    # Add tracking event
    event = TrackingEvent(
        event_type=status.value.upper(),
        location=location or shipment.origin,
        timestamp=datetime.now(),
        description=description or f"Status updated to {status.value}",
        status=status
    )
    shipment.tracking_events.append(event)
    
    # Update timestamps
    if status == ShipmentStatus.IN_TRANSIT and not shipment.actual_departure:
        shipment.actual_departure = datetime.now()
    elif status == ShipmentStatus.DELIVERED and not shipment.actual_arrival:
        shipment.actual_arrival = datetime.now()
    
    await repository.save(shipment)
    return shipment

