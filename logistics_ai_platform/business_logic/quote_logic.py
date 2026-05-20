"""
Pure functions for quote business logic

This module contains pure business logic functions for quote generation.
All functions are side-effect free and easily testable.
"""

import logging
from typing import List, Optional
from datetime import datetime, timedelta
import uuid

from models.schemas import (
    QuoteRequest,
    QuoteOption,
    TransportationMode,
    Location,
    CargoDetails,
)

logger = logging.getLogger(__name__)


def generate_quote_id() -> str:
    """
    Generate a unique quote ID
    
    Returns:
        Unique quote identifier in format QXXXXXXXX
    """
    quote_id = f"Q{str(uuid.uuid4())[:8].upper()}"
    logger.debug(f"Generated quote ID: {quote_id}")
    return quote_id


def generate_request_id() -> str:
    """
    Generate a unique request ID
    
    Returns:
        Unique request identifier in format REQXXXXXXXX
    """
    request_id = f"REQ{str(uuid.uuid4())[:8].upper()}"
    logger.debug(f"Generated request ID: {request_id}")
    return request_id


def calculate_distance(origin: Location, destination: Location) -> float:
    """
    Calculate approximate distance between locations
    
    Args:
        origin: Origin location
        destination: Destination location
        
    Returns:
        Distance in kilometers
        
    Note:
        Uses geospatial calculation if available, otherwise
        returns a default estimation.
    """
    if not origin or not destination:
        logger.warning("Invalid locations provided, using default distance")
        return 1000.0
    
    try:
        from utils.geospatial import calculate_distance_km
        distance = calculate_distance_km(origin, destination)
        if distance is not None and distance > 0:
            logger.debug(f"Calculated distance: {distance} km")
            return distance
    except Exception as e:
        logger.warning(f"Error calculating distance: {e}, using fallback")
    
    # Fallback: simplified estimation
    logger.debug("Using fallback distance calculation")
    return 1000.0  # Default distance in km


def calculate_transit_days(
    distance: float,
    mode: TransportationMode
) -> int:
    """Calculate transit days based on distance and mode"""
    if mode == TransportationMode.AIR:
        return max(1, int(distance / 5000))
    elif mode == TransportationMode.MARITIME:
        return max(7, int(distance / 200))
    elif mode == TransportationMode.GROUND:
        return max(1, int(distance / 800))
    return 7  # Default


def calculate_price(
    cargo: CargoDetails,
    distance: float,
    mode: TransportationMode
) -> float:
    """
    Calculate price based on cargo, distance, and transportation mode
    
    Args:
        cargo: Cargo details (weight, volume, quantity)
        distance: Distance in kilometers
        mode: Transportation mode
        
    Returns:
        Calculated price in USD
        
    Note:
        Pricing logic:
        - Air: Weight-based pricing
        - Maritime: Volume or weight (whichever is higher)
        - Ground: Weight + distance-based pricing
    """
    if not cargo or cargo.weight_kg <= 0:
        logger.warning("Invalid cargo details, returning 0.0")
        return 0.0
    
    if distance < 0:
        logger.warning("Invalid distance, using 0")
        distance = 0.0
    
    price = 0.0
    
    if mode == TransportationMode.AIR:
        # Air freight: weight-based, premium pricing
        price = cargo.weight_kg * 5.0
    elif mode == TransportationMode.MARITIME:
        # Maritime: charge by volume or weight (whichever is higher)
        volume_price = (cargo.volume_m3 * 150.0) if cargo.volume_m3 else 0
        weight_price = cargo.weight_kg * 0.5
        price = max(volume_price, weight_price)
    elif mode == TransportationMode.GROUND:
        # Ground: weight + distance-based
        price = cargo.weight_kg * 1.5 + (distance * 0.1)
    else:
        logger.warning(f"Unknown transportation mode: {mode}, returning 0.0")
        return 0.0
    
    # Ensure minimum price
    price = max(price, 50.0)  # Minimum $50
    
    logger.debug(f"Calculated price: ${price:.2f} for {mode.value} mode")
    return round(price, 2)


def create_quote_option(
    option_id: str,
    mode: TransportationMode,
    cargo: CargoDetails,
    distance: float,
    preferred_departure: Optional[datetime] = None
) -> QuoteOption:
    """Create a quote option"""
    transit_days = calculate_transit_days(distance, mode)
    price = calculate_price(cargo, distance, mode)
    
    departure = preferred_departure or datetime.now() + timedelta(days=1 if mode != TransportationMode.MARITIME else 3)
    arrival = departure + timedelta(days=transit_days)
    
    carriers = {
        TransportationMode.AIR: "Air Freight Express",
        TransportationMode.MARITIME: "Ocean Freight Line",
        TransportationMode.GROUND: "Ground Transport Co",
    }
    
    service_levels = {
        TransportationMode.AIR: "Express",
        TransportationMode.MARITIME: "Standard",
        TransportationMode.GROUND: "Standard",
    }
    
    features_map = {
        TransportationMode.AIR: ["Fast delivery", "Real-time tracking", "Priority handling"],
        TransportationMode.MARITIME: ["Cost-effective", "Container tracking", "Port handling"],
        TransportationMode.GROUND: ["Door-to-door", "GPS tracking", "Flexible scheduling"],
    }
    
    return QuoteOption(
        quote_id=option_id,
        transportation_mode=mode,
        carrier=carriers[mode],
        estimated_departure=departure,
        estimated_arrival=arrival,
        transit_days=transit_days,
        price_usd=price,
        service_level=service_levels[mode],
        features=features_map[mode]
    )


def create_quote_options(
    request: QuoteRequest
) -> List[QuoteOption]:
    """Create quote options based on request"""
    distance = calculate_distance(request.origin, request.destination)
    options = []
    
    modes_to_generate = []
    if request.transportation_mode == TransportationMode.MULTIMODAL:
        modes_to_generate = [
            TransportationMode.AIR,
            TransportationMode.MARITIME,
            TransportationMode.GROUND
        ]
    else:
        modes_to_generate = [request.transportation_mode]
    
    for mode in modes_to_generate:
        option_id = f"OPT-{mode.value.upper()[:3]}-{str(uuid.uuid4())[:8].upper()}"
        option = create_quote_option(
            option_id=option_id,
            mode=mode,
            cargo=request.cargo,
            distance=distance,
            preferred_departure=request.preferred_departure_date
        )
        options.append(option)
    
    # Sort by price
    options.sort(key=lambda x: x.price_usd)
    return options

