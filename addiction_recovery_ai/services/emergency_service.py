"""
Servicio de emergencia - Manejo de situaciones críticas y contactos de emergencia
"""

from typing import Dict, List, Optional
from datetime import datetime
import json


class EmergencyService:
    """Servicio de manejo de emergencias"""
    
    def __init__(self):
        """Inicializa el servicio de emergencia"""
        self.crisis_resources = self._load_crisis_resources()
    
    def create_emergency_contact(
        self,
        user_id: str,
        name: str,
        relationship: str,
        phone: str,
        email: Optional[str] = None,
        is_primary: bool = False
    ) -> Dict:
        """
        Crea un contacto de emergencia
        
        Args:
            user_id: ID del usuario
            name: Nombre del contacto
            relationship: Relación con el usuario
            phone: Teléfono
            email: Email (opcional)
            is_primary: Si es contacto principal
        
        Returns:
            Contacto de emergencia creado
        """
        contact = {
            "user_id": user_id,
            "name": name,
            "relationship": relationship,
            "phone": phone,
            "email": email,
            "is_primary": is_primary,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        return contact
    
    def get_emergency_contacts(self, user_id: str) -> List[Dict]:
        """
        Obtiene contactos de emergencia del usuario
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Lista de contactos de emergencia
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def trigger_emergency_protocol(
        self,
        user_id: str,
        risk_level: str,
        situation: str
    ) -> Dict:
        """
        Activa protocolo de emergencia
        
        Args:
            user_id: ID del usuario
            risk_level: Nivel de riesgo
            situation: Descripción de la situación
        
        Returns:
            Protocolo de emergencia activado
        """
        protocol = {
            "user_id": user_id,
            "risk_level": risk_level,
            "situation": situation,
            "triggered_at": datetime.now().isoformat(),
            "actions": self._get_emergency_actions(risk_level),
            "resources": self._get_emergency_resources(risk_level),
            "contacts": self.get_emergency_contacts(user_id)
        }
        
        return protocol
    
    def get_crisis_resources(self, location: Optional[str] = None) -> List[Dict]:
        """
        Obtiene recursos de crisis disponibles
        
        Args:
            location: Ubicación (opcional, para recursos locales)
        
        Returns:
            Lista de recursos de crisis
        """
        resources = self.crisis_resources.copy()
        
        # Filtrar por ubicación si se proporciona
        if location:
            # En implementación real, filtrar por ubicación
            pass
        
        return resources
    
    def _get_emergency_actions(self, risk_level: str) -> List[str]:
        """Obtiene acciones de emergencia según nivel de riesgo"""
        if risk_level == "crítico":
            return [
                "1. Llama inmediatamente a una línea de crisis o 911",
                "2. Contacta a tu contacto de emergencia principal",
                "3. Ve a un lugar seguro",
                "4. No estés solo - busca compañía",
                "5. Usa técnicas de respiración profunda"
            ]
        elif risk_level == "alto":
            return [
                "1. Contacta a tu sistema de apoyo inmediatamente",
                "2. Implementa tu plan de emergencia",
                "3. Usa estrategias de afrontamiento",
                "4. Considera llamar a una línea de apoyo"
            ]
        else:
            return [
                "1. Contacta a tu sistema de apoyo",
                "2. Revisa tus estrategias de afrontamiento",
                "3. Practica técnicas de relajación"
            ]
    
    def _get_emergency_resources(self, risk_level: str) -> List[Dict]:
        """Obtiene recursos de emergencia"""
        resources = []
        
        if risk_level in ["alto", "crítico"]:
            resources.extend([
                {
                    "type": "crisis_line",
                    "name": "Línea Nacional de Prevención del Suicidio",
                    "phone": "988",
                    "available": "24/7",
                    "description": "Línea de crisis disponible las 24 horas"
                },
                {
                    "type": "emergency",
                    "name": "Emergencias",
                    "phone": "911",
                    "available": "24/7",
                    "description": "Servicios de emergencia"
                }
            ])
        
        resources.extend(self.crisis_resources)
        
        return resources
    
    def _load_crisis_resources(self) -> List[Dict]:
        """Carga recursos de crisis"""
        return [
            {
                "type": "support_group",
                "name": "Grupos de Apoyo",
                "description": "Busca grupos de 12 pasos o similares en tu área",
                "website": "https://www.aa.org"
            },
            {
                "type": "therapy",
                "name": "Terapia y Consejería",
                "description": "Considera buscar ayuda profesional",
                "website": "https://www.psychologytoday.com"
            },
            {
                "type": "online_support",
                "name": "Apoyo en Línea",
                "description": "Comunidades de apoyo en línea disponibles 24/7",
                "website": "https://www.reddit.com/r/stopdrinking"
            }
        ]

