"""Booking request handlers"""

from models.schemas import BookingRequest, BookingResponse
from repositories.booking_repository import BookingRepository
from repositories.quote_repository import QuoteRepository
from repositories.shipment_repository import ShipmentRepository
from domain.bookings import create_booking_domain, get_booking_domain
from utils.handler_helpers import (
    get_entity_or_raise,
    update_entity_with_cache_invalidation
)
from utils.metrics import get_metrics_collector


async def handle_create_booking(
    request: BookingRequest,
    booking_repo: BookingRepository,
    quote_repo: QuoteRepository,
    shipment_repo: ShipmentRepository
) -> BookingResponse:
    """Handle booking creation request"""
    booking = await create_booking_domain(
        request=request,
        booking_repo=booking_repo,
        quote_repo=quote_repo,
        shipment_repo=shipment_repo
    )
    
    from utils.handler_helpers import create_entity_with_cache
    from utils.constants import DEFAULT_CACHE_TTL
    
    await create_entity_with_cache(
        booking,
        f"booking:{booking.booking_id}",
        cache_ttl=DEFAULT_CACHE_TTL
    )
    
    await update_entity_with_cache_invalidation(
        booking,
        f"quote:{request.quote_id}"
    )
    
    # Record business metric
    get_metrics_collector().record_booking_created()
    
    return booking


async def handle_get_booking(
    booking_id: str,
    repository: BookingRepository
) -> BookingResponse:
    """Handle get booking request"""
    from utils.constants import DEFAULT_CACHE_TTL
    return await get_entity_or_raise(
        booking_id,
        lambda: get_booking_domain(booking_id, repository),
        "Booking",
        cache_key=f"booking:{booking_id}",
        cache_ttl=DEFAULT_CACHE_TTL,
        model_class=BookingResponse
    )


async def handle_get_bookings_by_shipment(
    shipment_id: str,
    repository: BookingRepository
) -> list[BookingResponse]:
    """Handle get bookings by shipment request"""
    from utils.exceptions import ValidationError
    
    if not shipment_id or not shipment_id.strip():
        raise ValidationError("Shipment ID is required", field="shipment_id")
    
    bookings = await repository.find_by_shipment_id(shipment_id)
    return bookings


