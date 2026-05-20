"""
Quote validation functions

This module provides validation functions for quote-related data,
ensuring data integrity before processing.
"""

import logging
from typing import Optional
from datetime import datetime

from models.schemas import QuoteRequest, Location, CargoDetails
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


def validate_quote_request(request: QuoteRequest) -> None:
    """
    Validate quote request data
    
    Args:
        request: Quote request to validate
        
    Raises:
        ValidationError: If validation fails
        
    Validates:
        - Origin location (country and city required)
        - Destination location (country and city required)
        - Cargo details (weight, quantity, dimensions)
        - Transportation mode
        - Optional fields (preferred departure date)
    """
    if not request:
        raise ValidationError("Quote request is required")
    
    # Validate origin
    if not request.origin:
        raise ValidationError("Origin location is required", field="origin")
    
    if not request.origin.country or not request.origin.country.strip():
        raise ValidationError(
            "Origin country is required",
            field="origin.country"
        )
    
    if not request.origin.city or not request.origin.city.strip():
        raise ValidationError(
            "Origin city is required",
            field="origin.city"
        )
    
    # Validate destination
    if not request.destination:
        raise ValidationError("Destination location is required", field="destination")
    
    if not request.destination.country or not request.destination.country.strip():
        raise ValidationError(
            "Destination country is required",
            field="destination.country"
        )
    
    if not request.destination.city or not request.destination.city.strip():
        raise ValidationError(
            "Destination city is required",
            field="destination.city"
        )
    
    # Validate origin != destination
    if (request.origin.country == request.destination.country and
        request.origin.city == request.destination.city):
        raise ValidationError(
            "Origin and destination cannot be the same",
            field="destination"
        )
    
    # Validate cargo
    if not request.cargo:
        raise ValidationError("Cargo details are required", field="cargo")
    
    if request.cargo.weight_kg <= 0:
        raise ValidationError(
            "Cargo weight must be greater than 0",
            field="cargo.weight_kg"
        )
    
    if request.cargo.weight_kg > 100000:  # 100 tons
        raise ValidationError(
            "Cargo weight exceeds maximum limit (100,000 kg)",
            field="cargo.weight_kg"
        )
    
    if request.cargo.quantity <= 0:
        raise ValidationError(
            "Cargo quantity must be greater than 0",
            field="cargo.quantity"
        )
    
    if request.cargo.quantity > 10000:
        raise ValidationError(
            "Cargo quantity exceeds maximum limit (10,000)",
            field="cargo.quantity"
        )
    
    # Validate dimensions if provided
    if request.cargo.length_m and request.cargo.length_m <= 0:
        raise ValidationError(
            "Cargo length must be greater than 0",
            field="cargo.length_m"
        )
    
    if request.cargo.width_m and request.cargo.width_m <= 0:
        raise ValidationError(
            "Cargo width must be greater than 0",
            field="cargo.width_m"
        )
    
    if request.cargo.height_m and request.cargo.height_m <= 0:
        raise ValidationError(
            "Cargo height must be greater than 0",
            field="cargo.height_m"
        )
    
    # Validate preferred departure date if provided
    if request.preferred_departure_date:
        if request.preferred_departure_date < datetime.now():
            raise ValidationError(
                "Preferred departure date cannot be in the past",
                field="preferred_departure_date"
            )
    
    logger.debug("Quote request validation passed")


def validate_location(location: Optional[Location], field_name: str = "location") -> None:
    """
    Validate location data
    
    Args:
        location: Location to validate
        field_name: Field name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not location:
        raise ValidationError(f"{field_name} is required", field=field_name)
    
    if not location.country or not location.country.strip():
        raise ValidationError(
            f"{field_name} country is required",
            field=f"{field_name}.country"
        )
    
    if not location.city or not location.city.strip():
        raise ValidationError(
            f"{field_name} city is required",
            field=f"{field_name}.city"
        )


def validate_cargo_details(cargo: Optional[CargoDetails], field_name: str = "cargo") -> None:
    """
    Validate cargo details
    
    Args:
        cargo: Cargo details to validate
        field_name: Field name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not cargo:
        raise ValidationError(f"{field_name} is required", field=field_name)
    
    if cargo.weight_kg <= 0:
        raise ValidationError(
            f"{field_name} weight must be greater than 0",
            field=f"{field_name}.weight_kg"
        )
    
    if cargo.quantity <= 0:
        raise ValidationError(
            f"{field_name} quantity must be greater than 0",
            field=f"{field_name}.quantity"
        )

