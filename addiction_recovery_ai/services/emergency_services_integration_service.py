"""
Servicio de Integración con Servicios de Emergencia - Sistema completo de servicios de emergencia
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class EmergencyType(str, Enum):
    """Tipos de emergencia"""
    CRISIS = "crisis"
    RELAPSE = "relapse"
    MEDICAL = "medical"
    MENTAL_HEALTH = "mental_health"
    SUICIDAL = "suicidal"


class EmergencyServicesIntegrationService:
    """Servicio de integración con servicios de emergencia"""
    
    def __init__(self):
        """Inicializa el servicio de emergencia"""
        self.emergency_contacts = self._load_emergency_contacts()
    
    def trigger_emergency(
        self,
        user_id: str,
        emergency_type: str,
        emergency_data: Dict
    ) -> Dict:
        """
        Activa emergencia
        
        Args:
            user_id: ID del usuario
            emergency_type: Tipo de emergencia
            emergency_data: Datos de emergencia
        
        Returns:
            Emergencia activada
        """
        emergency = {
            "id": f"emergency_{datetime.now().timestamp()}",
            "user_id": user_id,
            "emergency_type": emergency_type,
            "emergency_data": emergency_data,
            "severity": emergency_data.get("severity", "moderate"),
            "triggered_at": datetime.now().isoformat(),
            "status": "active",
            "response_actions": self._determine_response_actions(emergency_type, emergency_data)
        }
        
        return emergency
    
    def get_emergency_resources(
        self,
        user_id: str,
        location: Optional[Dict] = None
    ) -> Dict:
        """
        Obtiene recursos de emergencia
        
        Args:
            user_id: ID del usuario
            location: Ubicación del usuario
        
        Returns:
            Recursos de emergencia
        """
        return {
            "user_id": user_id,
            "crisis_hotlines": self._get_crisis_hotlines(),
            "emergency_services": self._get_emergency_services(location),
            "mental_health_resources": self._get_mental_health_resources(location),
            "local_support": self._get_local_support(location),
            "generated_at": datetime.now().isoformat()
        }
    
    def log_emergency_response(
        self,
        emergency_id: str,
        response_data: Dict
    ) -> Dict:
        """
        Registra respuesta de emergencia
        
        Args:
            emergency_id: ID de emergencia
            response_data: Datos de respuesta
        
        Returns:
            Respuesta registrada
        """
        return {
            "emergency_id": emergency_id,
            "response_data": response_data,
            "response_time": response_data.get("response_time_minutes", 0),
            "outcome": response_data.get("outcome", "resolved"),
            "logged_at": datetime.now().isoformat()
        }
    
    def _load_emergency_contacts(self) -> List[Dict]:
        """Carga contactos de emergencia"""
        return [
            {
                "type": "crisis_hotline",
                "name": "Línea Nacional de Crisis",
                "phone": "988",
                "available": "24/7"
            },
            {
                "type": "suicide_prevention",
                "name": "Línea de Prevención de Suicidio",
                "phone": "988",
                "available": "24/7"
            }
        ]
    
    def _determine_response_actions(self, emergency_type: str, data: Dict) -> List[str]:
        """Determina acciones de respuesta"""
        actions = []
        
        if emergency_type == EmergencyType.SUICIDAL:
            actions.append("⚠️ EMERGENCIA: Contacta servicios de emergencia inmediatamente (911 o 988)")
            actions.append("No estás solo. Hay ayuda disponible")
        elif emergency_type == EmergencyType.CRISIS:
            actions.append("Contacta línea de crisis (988)")
            actions.append("Notifica a tu sistema de apoyo")
        elif emergency_type == EmergencyType.RELAPSE:
            actions.append("Contacta tu consejero o terapeuta")
            actions.append("Revisa tu plan de prevención de recaídas")
        
        return actions
    
    def _get_crisis_hotlines(self) -> List[Dict]:
        """Obtiene líneas de crisis"""
        return [
            {
                "name": "Línea Nacional de Crisis",
                "phone": "988",
                "available": "24/7",
                "description": "Línea de crisis para salud mental"
            }
        ]
    
    def _get_emergency_services(self, location: Optional[Dict]) -> List[Dict]:
        """Obtiene servicios de emergencia"""
        return [
            {
                "name": "Servicios de Emergencia",
                "phone": "911",
                "available": "24/7",
                "description": "Servicios de emergencia médica"
            }
        ]
    
    def _get_mental_health_resources(self, location: Optional[Dict]) -> List[Dict]:
        """Obtiene recursos de salud mental"""
        return [
            {
                "name": "Centro de Salud Mental Local",
                "description": "Servicios de salud mental locales"
            }
        ]
    
    def _get_local_support(self, location: Optional[Dict]) -> List[Dict]:
        """Obtiene apoyo local"""
        return [
            {
                "name": "Grupos de Apoyo Local",
                "description": "Grupos de apoyo en tu área"
            }
        ]

