"""Booking domain logic - pure functions"""

from typing import Optional
from datetime import datetime

from models.schemas import (
    BookingRequest,
    BookingResponse,
    ShipmentStatus,
)
from repositories.booking_repository import BookingRepository
from repositories.quote_repository import QuoteRepository
from repositories.shipment_repository import ShipmentRepository
from business_logic.booking_logic import validate_quote_option
from validators.booking_validators import validate_booking_request
from domain.shipments import create_shipment_from_booking_domain
from factories.booking_factory import build_booking_response
from utils.logger import logger
from utils.exceptions import NotFoundError


async def create_booking_domain(
    request: BookingRequest,
    booking_repo: BookingRepository,
    quote_repo: QuoteRepository,
    shipment_repo: ShipmentRepository
) -> BookingResponse:
    """Create booking - pure domain function"""
    validate_booking_request(request)
    
    # Get quote
    quote = await quote_repo.find_by_id(request.quote_id)
    if not quote:
        raise NotFoundError("Quote", request.quote_id)
    
    # Validate option
    selected_option = validate_quote_option(quote, request.selected_option_id)
    
    # Create shipment
    shipment = await create_shipment_from_booking_domain(
        booking_id=None,  # Will be set after booking creation
        quote=quote,
        option=selected_option,
        shipment_repo=shipment_repo
    )
    
    # Build booking using factory
    from factories.booking_factory import build_booking_response
    booking = build_booking_response(
        quote=quote,
        option=selected_option,
        shipment_id=shipment.shipment_id,
        quote_id=request.quote_id
    )
    
    # Update shipment with booking_id
    shipment.booking_id = booking.booking_id
    await shipment_repo.save(shipment)
    
    await booking_repo.save(booking)
    logger.info(f"Booking created: {booking.booking_id}")
    
    return booking


async def get_booking_domain(
    booking_id: str,
    repository: BookingRepository
) -> Optional[BookingResponse]:
    """Get booking - pure domain function"""
    return await repository.find_by_id(booking_id)

