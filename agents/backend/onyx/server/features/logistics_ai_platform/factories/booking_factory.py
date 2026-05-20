"""Factory functions for booking objects"""

from datetime import datetime

from models.schemas import (
    BookingResponse,
    ShipmentStatus,
    QuoteResponse,
    QuoteOption,
)
from business_logic.booking_logic import (
    generate_booking_id,
    generate_booking_reference,
)


def build_booking_response(
    quote: QuoteResponse,
    option: QuoteOption,
    shipment_id: str,
    quote_id: str
) -> BookingResponse:
    """Build booking response - pure factory function"""
    booking_id = generate_booking_id()
    booking_reference = generate_booking_reference(booking_id)
    
    return BookingResponse(
        booking_id=booking_id,
        quote_id=quote_id,
        shipment_id=shipment_id,
        status=ShipmentStatus.BOOKED,
        booking_reference=booking_reference,
        carrier_reference=None,
        created_at=datetime.now(),
        estimated_departure=option.estimated_departure,
        estimated_arrival=option.estimated_arrival
    )









