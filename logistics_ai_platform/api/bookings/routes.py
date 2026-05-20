"""
Booking routes

This module provides API endpoints for booking management including:
- Booking creation from quotes
- Booking retrieval
- Booking validation
"""

from typing import Annotated
from fastapi import APIRouter, Depends, status

from models.schemas import BookingRequest, BookingResponse
from handlers.booking_handlers import (
    handle_create_booking,
    handle_get_booking,
    handle_get_bookings_by_shipment,
)
from utils.dependencies import (
    get_booking_repository,
    get_quote_repository,
    get_shipment_repository,
)
from utils.exceptions import ValidationError
from repositories.booking_repository import BookingRepository
from repositories.quote_repository import QuoteRepository
from repositories.shipment_repository import ShipmentRepository

router = APIRouter(
    prefix="/bookings",
    tags=["Bookings"],
    responses={
        404: {"description": "Booking not found"},
        422: {"description": "Validation error"},
        400: {"description": "Business logic error"}
    }
)


BookingRepositoryDep = Annotated[BookingRepository, Depends(get_booking_repository)]
QuoteRepositoryDep = Annotated[QuoteRepository, Depends(get_quote_repository)]
ShipmentRepositoryDep = Annotated[ShipmentRepository, Depends(get_shipment_repository)]


@router.post(
    "",
    response_model=BookingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new booking",
    description="Creates a new booking from a quote with selected option"
)
async def create_booking(
    request: BookingRequest,
    booking_repo: BookingRepositoryDep,
    quote_repo: QuoteRepositoryDep,
    shipment_repo: ShipmentRepositoryDep
) -> BookingResponse:
    """
    Create a new booking from a quote
    
    Args:
        request: Booking creation request with quote ID and selected option
        booking_repo: Injected booking repository
        quote_repo: Injected quote repository
        shipment_repo: Injected shipment repository
        
    Returns:
        BookingResponse: Created booking with shipment information
        
    Raises:
        ValidationError: If request data is invalid
        NotFoundError: If quote not found
        BusinessLogicError: If booking cannot be created
    """
    return await handle_create_booking(
        request,
        booking_repo,
        quote_repo,
        shipment_repo
    )


@router.get(
    "/{booking_id}",
    response_model=BookingResponse,
    summary="Get booking by ID",
    description="Retrieves a booking by its unique identifier"
)
async def get_booking(
    booking_id: str,
    repository: BookingRepositoryDep
) -> BookingResponse:
    """
    Get booking by ID
    
    Args:
        booking_id: Unique booking identifier
        repository: Injected booking repository
        
    Returns:
        BookingResponse: Booking details
        
    Raises:
        NotFoundError: If booking not found
        ValidationError: If booking_id is invalid
    """
    if not booking_id or not booking_id.strip():
        raise ValidationError("Booking ID is required", field="booking_id")
    
    return await handle_get_booking(booking_id, repository)


@router.get(
    "/shipment/{shipment_id}",
    response_model=list[BookingResponse],
    summary="Get bookings by shipment",
    description="Retrieves all bookings associated with a shipment"
)
async def get_bookings_by_shipment(
    shipment_id: str,
    repository: BookingRepositoryDep
) -> list[BookingResponse]:
    """
    Get bookings for a shipment
    
    Args:
        shipment_id: Unique shipment identifier
        repository: Injected booking repository
        
    Returns:
        List of bookings for the shipment
        
    Raises:
        ValidationError: If shipment_id is invalid
    """
    return await handle_get_bookings_by_shipment(shipment_id, repository)

