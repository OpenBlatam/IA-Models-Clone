"""
Servicio de Seguimiento de Ubicación - Sistema para evitar triggers geográficos
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class LocationType(str, Enum):
    """Tipos de ubicación"""
    TRIGGER_ZONE = "trigger_zone"
    SAFE_ZONE = "safe_zone"
    SUPPORT_LOCATION = "support_location"
    TREATMENT_CENTER = "treatment_center"


class LocationTrackingService:
    """Servicio de seguimiento de ubicación"""
    
    def __init__(self):
        """Inicializa el servicio de seguimiento de ubicación"""
        pass
    
    def add_location(
        self,
        user_id: str,
        location_type: str,
        name: str,
        latitude: float,
        longitude: float,
        radius_meters: int = 100,
        description: Optional[str] = None
    ) -> Dict:
        """
        Agrega una ubicación
        
        Args:
            user_id: ID del usuario
            location_type: Tipo de ubicación
            name: Nombre de la ubicación
            latitude: Latitud
            longitude: Longitud
            radius_meters: Radio en metros
            description: Descripción
        
        Returns:
            Ubicación agregada
        """
        location = {
            "id": f"location_{datetime.now().timestamp()}",
            "user_id": user_id,
            "location_type": location_type,
            "name": name,
            "latitude": latitude,
            "longitude": longitude,
            "radius_meters": radius_meters,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        return location
    
    def check_location_proximity(
        self,
        user_id: str,
        current_latitude: float,
        current_longitude: float
    ) -> Dict:
        """
        Verifica proximidad a ubicaciones registradas
        
        Args:
            user_id: ID del usuario
            current_latitude: Latitud actual
            current_longitude: Longitud actual
        
        Returns:
            Análisis de proximidad
        """
        proximity_analysis = {
            "user_id": user_id,
            "current_location": {
                "latitude": current_latitude,
                "longitude": current_longitude
            },
            "nearby_locations": [],
            "trigger_zones_nearby": [],
            "safe_zones_nearby": [],
            "warnings": [],
            "checked_at": datetime.now().isoformat()
        }
        
        # Lógica para verificar proximidad
        # En implementación real, esto calcularía distancias
        
        return proximity_analysis
    
    def get_location_history(
        self,
        user_id: str,
        days: int = 7
    ) -> List[Dict]:
        """
        Obtiene historial de ubicaciones
        
        Args:
            user_id: ID del usuario
            days: Número de días
        
        Returns:
            Lista de ubicaciones históricas
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def create_geofence_alert(
        self,
        user_id: str,
        location_id: str,
        alert_type: str = "warning"
    ) -> Dict:
        """
        Crea alerta de geofence
        
        Args:
            user_id: ID del usuario
            location_id: ID de la ubicación
            alert_type: Tipo de alerta
        
        Returns:
            Alerta creada
        """
        alert = {
            "id": f"geofence_alert_{datetime.now().timestamp()}",
            "user_id": user_id,
            "location_id": location_id,
            "alert_type": alert_type,
            "triggered_at": datetime.now().isoformat(),
            "message": self._generate_geofence_message(alert_type),
            "action_required": alert_type == "critical"
        }
        
        return alert
    
    def _generate_geofence_message(self, alert_type: str) -> str:
        """Genera mensaje de geofence"""
        messages = {
            "warning": "Estás cerca de una zona de riesgo. Considera cambiar de ubicación.",
            "critical": "⚠️ Estás en una zona de alto riesgo. Contacta tu sistema de apoyo."
        }
        
        return messages.get(alert_type, "Alerta de ubicación activada")

