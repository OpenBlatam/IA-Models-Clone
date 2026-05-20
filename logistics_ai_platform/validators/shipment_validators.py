"""Shipment validation functions"""

from models.schemas import ShipmentRequest
from utils.exceptions import ValidationError


def validate_shipment_request(request: ShipmentRequest) -> None:
    """Validate shipment request"""
    if not request.origin.country or not request.origin.city:
        raise ValidationError("Origin must have country and city", field="origin")
    
    if not request.destination.country or not request.destination.city:
        raise ValidationError("Destination must have country and city", field="destination")
    
    if request.cargo.weight_kg <= 0:
        raise ValidationError("Cargo weight must be greater than 0", field="cargo.weight_kg")













