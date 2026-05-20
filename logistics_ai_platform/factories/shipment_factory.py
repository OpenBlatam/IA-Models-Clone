"""Factory functions for shipment objects"""

from datetime import datetime

from models.schemas import (
    ShipmentRequest,
    ShipmentResponse,
    ShipmentStatus,
)
from business_logic.shipment_logic import (
    generate_shipment_id,
    generate_shipment_reference,
    generate_tracking_number,
    generate_house_bill_number,
    generate_master_bill_number,
)


def build_shipment_response(request: ShipmentRequest) -> ShipmentResponse:
    """Build shipment response from request - pure factory function"""
    shipment_id = generate_shipment_id()
    shipment_reference = generate_shipment_reference(shipment_id)
    tracking_number = generate_tracking_number()
    house_bill_number = generate_house_bill_number()
    master_bill_number = generate_master_bill_number()
    
    return ShipmentResponse(
        shipment_id=shipment_id,
        booking_id=request.booking_id,
        shipment_reference=shipment_reference,
        tracking_number=tracking_number,
        house_bill_number=house_bill_number,
        master_bill_number=master_bill_number,
        origin=request.origin,
        destination=request.destination,
        cargo=request.cargo,
        transportation_mode=request.transportation_mode,
        status=ShipmentStatus.PENDING,
        carrier=request.carrier,
        tracking_events=[],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

