"""Geospatial utilities for logistics"""

from typing import Optional
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from models.schemas import Location


def calculate_distance_km(origin: Location, destination: Location) -> Optional[float]:
    """Calculate distance between two locations in kilometers"""
    if not origin.latitude or not origin.longitude:
        return None
    
    if not destination.latitude or not destination.longitude:
        return None
    
    origin_coords = (origin.latitude, origin.longitude)
    dest_coords = (destination.latitude, destination.longitude)
    
    return geodesic(origin_coords, dest_coords).kilometers


def geocode_location(location: Location) -> Optional[Location]:
    """Geocode a location to get coordinates"""
    if location.latitude and location.longitude:
        return location  # Already has coordinates
    
    try:
        geolocator = Nominatim(user_agent="logistics_ai_platform")
        query = f"{location.city}, {location.country}"
        if location.address:
            query = f"{location.address}, {query}"
        
        geocoded = geolocator.geocode(query)
        
        if geocoded:
            location.latitude = geocoded.latitude
            location.longitude = geocoded.longitude
            return location
    except Exception:
        pass
    
    return None













