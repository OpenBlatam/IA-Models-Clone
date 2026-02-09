"""
External APIs Service - Integración con APIs externas
"""

import logging
import os
from typing import Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class ExternalAPIsService:
    """Servicio para integración con APIs externas"""
    
    def __init__(self):
        self.google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def get_location_details(
        self,
        location: str
    ) -> Dict[str, Any]:
        """Obtener detalles de ubicación usando Google Maps"""
        
        if not self.google_maps_api_key:
            logger.warning("Google Maps API key no configurada")
            return self._default_location_details(location)
        
        try:
            # Geocoding API
            geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "address": location,
                "key": self.google_maps_api_key
            }
            
            response = await self.client.get(geocode_url, params=params)
            data = response.json()
            
            if data.get("status") == "OK" and data.get("results"):
                result = data["results"][0]
                geometry = result.get("geometry", {})
                location_data = geometry.get("location", {})
                
                # Place Details API para más información
                place_id = result.get("place_id")
                place_details = {}
                
                if place_id:
                    place_url = "https://maps.googleapis.com/maps/api/place/details/json"
                    place_params = {
                        "place_id": place_id,
                        "fields": "name,rating,user_ratings_total,types,formatted_address",
                        "key": self.google_maps_api_key
                    }
                    
                    place_response = await self.client.get(place_url, params=place_params)
                    place_data = place_response.json()
                    
                    if place_data.get("status") == "OK":
                        place_details = place_data.get("result", {})
                
                return {
                    "location": location,
                    "coordinates": {
                        "lat": location_data.get("lat"),
                        "lng": location_data.get("lng")
                    },
                    "formatted_address": result.get("formatted_address"),
                    "place_details": place_details,
                    "types": result.get("types", []),
                    "source": "google_maps"
                }
            else:
                return self._default_location_details(location)
                
        except Exception as e:
            logger.error(f"Error obteniendo detalles de ubicación: {e}")
            return self._default_location_details(location)
    
    async def get_weather_info(
        self,
        location: str
    ) -> Dict[str, Any]:
        """Obtener información del clima"""
        
        if not self.weather_api_key:
            logger.warning("Weather API key no configurada")
            return self._default_weather_info()
        
        try:
            # Usar OpenWeatherMap como ejemplo
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": self.weather_api_key,
                "units": "metric"
            }
            
            response = await self.client.get(weather_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    "location": location,
                    "temperature": data.get("main", {}).get("temp"),
                    "humidity": data.get("main", {}).get("humidity"),
                    "weather": data.get("weather", [{}])[0].get("description"),
                    "wind_speed": data.get("wind", {}).get("speed"),
                    "source": "openweathermap"
                }
            else:
                return self._default_weather_info()
                
        except Exception as e:
            logger.error(f"Error obteniendo información del clima: {e}")
            return self._default_weather_info()
    
    async def get_nearby_places(
        self,
        location: str,
        place_type: str = "store"
    ) -> Dict[str, Any]:
        """Obtener lugares cercanos"""
        
        if not self.google_maps_api_key:
            return {"error": "Google Maps API key no configurada"}
        
        try:
            # Primero obtener coordenadas
            location_details = await self.get_location_details(location)
            if "coordinates" not in location_details:
                return {"error": "No se pudieron obtener coordenadas"}
            
            coords = location_details["coordinates"]
            
            # Nearby Search API
            nearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": f"{coords['lat']},{coords['lng']}",
                "radius": 1000,  # 1km
                "type": place_type,
                "key": self.google_maps_api_key
            }
            
            response = await self.client.get(nearby_url, params=params)
            data = response.json()
            
            if data.get("status") == "OK":
                places = data.get("results", [])
                return {
                    "location": location,
                    "nearby_places": [
                        {
                            "name": p.get("name"),
                            "rating": p.get("rating"),
                            "types": p.get("types", []),
                            "vicinity": p.get("vicinity")
                        }
                        for p in places[:10]  # Top 10
                    ],
                    "count": len(places),
                    "source": "google_maps"
                }
            else:
                return {"error": "No se encontraron lugares cercanos"}
                
        except Exception as e:
            logger.error(f"Error obteniendo lugares cercanos: {e}")
            return {"error": str(e)}
    
    def _default_location_details(self, location: str) -> Dict[str, Any]:
        """Detalles de ubicación por defecto"""
        return {
            "location": location,
            "coordinates": None,
            "formatted_address": location,
            "source": "default"
        }
    
    def _default_weather_info(self) -> Dict[str, Any]:
        """Información del clima por defecto"""
        return {
            "location": "Unknown",
            "temperature": None,
            "source": "default"
        }
    
    async def close(self):
        """Cerrar cliente HTTP"""
        await self.client.aclose()




