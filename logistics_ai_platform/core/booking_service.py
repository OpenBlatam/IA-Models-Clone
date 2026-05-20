"""
Booking service for managing freight bookings

This service provides business logic for booking management including:
- Booking creation from quotes
- Booking validation and retrieval
- Integration with quote and shipment services
"""

from typing import Optional, List
from datetime import datetime
import logging

from models.schemas import (
    BookingRequest,
    BookingResponse,
    ShipmentStatus,
)
from repositories.booking_repository import BookingRepository
from core.quote_service import QuoteService
from core.shipment_service import ShipmentService
from business_logic.booking_logic import (
    generate_booking_id,
    generate_booking_reference,
    validate_quote_option,
)
from validators.booking_validators import validate_booking_request
from utils.exceptions import NotFoundError, ValidationError, BusinessLogicError

logger = logging.getLogger(__name__)


class BookingService:
    """Service for managing bookings"""
    
    def __init__(
        self,
        repository: BookingRepository,
        quote_service: QuoteService,
        shipment_service: ShipmentService
    ):
        """Initialize booking service"""
        self.repository = repository
        self.quote_service = quote_service
        self.shipment_service = shipment_service
    
    async def create_booking(self, request: BookingRequest) -> BookingResponse:
        """
        Create a new booking from a quote
        
        Args:
            request: Booking creation request
            
        Returns:
            BookingResponse: Created booking
            
        Raises:
            ValidationError: If request data is invalid
            NotFoundError: If quote not found
            BusinessLogicError: If booking cannot be created
        """
        # Validate request
        try:
            validate_booking_request(request)
        except ValueError as e:
            raise ValidationError(str(e))
        
        # Validate quote exists and is not expired
        quote = await self.quote_service.get_quote_or_raise(request.quote_id)
        
        # Validate and get selected option
        try:
            selected_option = validate_quote_option(quote, request.selected_option_id)
        except ValueError as e:
            raise ValidationError(str(e), field="selected_option_id")
        
        # Generate booking IDs
        booking_id = generate_booking_id()
        booking_reference = generate_booking_reference(booking_id)
        
        # Create shipment
        try:
            shipment = await self.shipment_service.create_shipment_from_booking(
                booking_id=booking_id,
                quote=quote,
                option=selected_option,
                shipper_info=request.shipper_info,
                consignee_info=request.consignee_info
            )
        except Exception as e:
            logger.error(f"Error creating shipment for booking: {e}", exc_info=True)
            raise BusinessLogicError(f"Failed to create shipment: {str(e)}")
        
        # Create booking response
        now = datetime.now()
        booking_response = BookingResponse(
            booking_id=booking_id,
            quote_id=request.quote_id,
            shipment_id=shipment.shipment_id,
            status=ShipmentStatus.BOOKED,
            booking_reference=booking_reference,
            carrier_reference=None,
            created_at=now,
            estimated_departure=selected_option.estimated_departure,
            estimated_arrival=selected_option.estimated_arrival
        )
        
        try:
            await self.repository.save(booking_response)
            logger.info(
                f"Booking created: {booking_id} "
                f"(quote: {request.quote_id}, shipment: {shipment.shipment_id})"
            )
        except Exception as e:
            logger.error(f"Error saving booking: {e}", exc_info=True)
            raise BusinessLogicError(f"Failed to save booking: {str(e)}")
        
        return booking_response
    
    async def get_booking(self, booking_id: str) -> Optional[BookingResponse]:
        """
        Get booking by ID
        
        Args:
            booking_id: Booking identifier
            
        Returns:
            BookingResponse if found, None otherwise
        """
        if not booking_id:
            raise ValidationError("Booking ID is required", field="booking_id")
        
        booking = await self.repository.find_by_id(booking_id)
        
        if not booking:
            logger.debug(f"Booking not found: {booking_id}")
        
        return booking
    
    async def get_booking_or_raise(self, booking_id: str) -> BookingResponse:
        """
        Get booking by ID or raise NotFoundError
        
        Args:
            booking_id: Booking identifier
            
        Returns:
            BookingResponse: Found booking
            
        Raises:
            NotFoundError: If booking not found
        """
        booking = await self.get_booking(booking_id)
        if not booking:
            raise NotFoundError("Booking", booking_id)
        return booking
    
    async def get_bookings_by_shipment(
        self,
        shipment_id: str
    ) -> List[BookingResponse]:
        """
        Get bookings for a shipment
        
        Args:
            shipment_id: Shipment identifier
            
        Returns:
            List of bookings for the shipment
        """
        if not shipment_id:
            raise ValidationError("Shipment ID is required", field="shipment_id")
        
        bookings = await self.repository.find_by_shipment_id(shipment_id)
        logger.debug(f"Found {len(bookings)} bookings for shipment {shipment_id}")
        
        return bookings

