"""
Servicio de Integración con Emergencias - Conexión con servicios de emergencia
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class EmergencyServiceType(str, Enum):
    """Tipos de servicios de emergencia"""
    POLICE = "police"
    AMBULANCE = "ambulance"
    FIRE = "fire"
    CRISIS_LINE = "crisis_line"
    SUICIDE_PREVENTION = "suicide_prevention"
    ADDICTION_HOTLINE = "addiction_hotline"


class EmergencyIntegrationService:
    """Servicio de integración con servicios de emergencia"""
    
    def __init__(self):
        """Inicializa el servicio de integración con emergencias"""
        self.emergency_services = self._load_emergency_services()
    
    def get_emergency_services(
        self,
        location: Optional[str] = None,
        service_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene servicios de emergencia disponibles
        
        Args:
            location: Ubicación (opcional, para servicios locales)
            service_type: Tipo de servicio (opcional)
        
        Returns:
            Lista de servicios de emergencia
        """
        services = self.emergency_services.copy()
        
        if service_type:
            services = [s for s in services if s.get("type") == service_type]
        
        return services
    
    def trigger_emergency_call(
        self,
        user_id: str,
        emergency_type: str,
        location: Optional[str] = None,
        severity: str = "high"
    ) -> Dict:
        """
        Activa una llamada de emergencia
        
        Args:
            user_id: ID del usuario
            emergency_type: Tipo de emergencia
            location: Ubicación (opcional)
            severity: Severidad
        
        Returns:
            Información de emergencia activada
        """
        emergency = {
            "user_id": user_id,
            "emergency_type": emergency_type,
            "severity": severity,
            "location": location,
            "triggered_at": datetime.now().isoformat(),
            "services_contacted": self._get_services_for_emergency(emergency_type),
            "status": "active"
        }
        
        return emergency
    
    def get_crisis_resources(
        self,
        location: Optional[str] = None
    ) -> Dict:
        """
        Obtiene recursos de crisis disponibles
        
        Args:
            location: Ubicación (opcional)
        
        Returns:
            Recursos de crisis
        """
        return {
            "location": location,
            "crisis_lines": [
                {
                    "name": "Línea Nacional de Prevención del Suicidio",
                    "phone": "988",
                    "available": "24/7",
                    "languages": ["español", "inglés"]
                },
                {
                    "name": "Línea de Crisis de Adicciones",
                    "phone": "1-800-662-4357",
                    "available": "24/7",
                    "languages": ["español", "inglés"]
                }
            ],
            "emergency_services": {
                "police": "911",
                "ambulance": "911",
                "fire": "911"
            },
            "local_resources": self._get_local_resources(location),
            "online_resources": [
                {
                    "name": "Crisis Text Line",
                    "service": "Text HOME to 741741",
                    "available": "24/7"
                }
            ]
        }
    
    def create_emergency_plan(
        self,
        user_id: str,
        emergency_contacts: List[Dict],
        medical_info: Optional[Dict] = None
    ) -> Dict:
        """
        Crea un plan de emergencia personalizado
        
        Args:
            user_id: ID del usuario
            emergency_contacts: Lista de contactos de emergencia
            medical_info: Información médica (opcional)
        
        Returns:
            Plan de emergencia
        """
        plan = {
            "user_id": user_id,
            "emergency_contacts": emergency_contacts,
            "medical_info": medical_info or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "active": True
        }
        
        return plan
    
    def _get_services_for_emergency(self, emergency_type: str) -> List[str]:
        """Obtiene servicios apropiados para tipo de emergencia"""
        service_mapping = {
            "suicidal": [EmergencyServiceType.SUICIDE_PREVENTION, EmergencyServiceType.CRISIS_LINE],
            "overdose": [EmergencyServiceType.AMBULANCE, EmergencyServiceType.ADDICTION_HOTLINE],
            "violence": [EmergencyServiceType.POLICE],
            "medical": [EmergencyServiceType.AMBULANCE]
        }
        
        return service_mapping.get(emergency_type, [EmergencyServiceType.CRISIS_LINE])
    
    def _get_local_resources(self, location: Optional[str]) -> List[Dict]:
        """Obtiene recursos locales basados en ubicación"""
        # En implementación real, esto buscaría recursos locales
        return []
    
    def _load_emergency_services(self) -> List[Dict]:
        """Carga servicios de emergencia"""
        return [
            {
                "type": EmergencyServiceType.SUICIDE_PREVENTION,
                "name": "Línea Nacional de Prevención del Suicidio",
                "phone": "988",
                "available": "24/7",
                "description": "Línea de crisis disponible las 24 horas"
            },
            {
                "type": EmergencyServiceType.ADDICTION_HOTLINE,
                "name": "SAMHSA National Helpline",
                "phone": "1-800-662-4357",
                "available": "24/7",
                "description": "Línea de ayuda para adicciones"
            },
            {
                "type": EmergencyServiceType.CRISIS_LINE,
                "name": "Crisis Text Line",
                "service": "Text HOME to 741741",
                "available": "24/7",
                "description": "Línea de crisis por texto"
            }
        ]

