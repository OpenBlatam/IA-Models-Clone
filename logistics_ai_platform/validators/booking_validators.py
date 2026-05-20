"""Booking validation functions"""

from models.schemas import BookingRequest
from utils.exceptions import ValidationError


def validate_booking_request(request: BookingRequest) -> None:
    """Validate booking request"""
    if not request.quote_id:
        raise ValidationError("Quote ID is required", field="quote_id")
    
    if not request.selected_option_id:
        raise ValidationError("Selected option ID is required", field="selected_option_id")
    
    if not request.shipper_info:
        raise ValidationError("Shipper info is required", field="shipper_info")
    
    if not request.consignee_info:
        raise ValidationError("Consignee info is required", field="consignee_info")













