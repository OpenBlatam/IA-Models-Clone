"""
Servicio de Terapia con Realidad Virtual/Aumentada - Sistema completo de VR/AR
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class TherapyType(str, Enum):
    """Tipos de terapia VR/AR"""
    EXPOSURE_THERAPY = "exposure_therapy"
    MINDFULNESS = "mindfulness"
    RELAXATION = "relaxation"
    SOCIAL_SKILLS = "social_skills"
    COGNITIVE_BEHAVIORAL = "cognitive_behavioral"


class VRARTherapyService:
    """Servicio de terapia con realidad virtual/aumentada"""
    
    def __init__(self):
        """Inicializa el servicio de VR/AR"""
        self.available_scenarios = self._load_scenarios()
    
    def create_therapy_session(
        self,
        user_id: str,
        therapy_type: str,
        scenario_id: Optional[str] = None,
        duration_minutes: int = 30
    ) -> Dict:
        """
        Crea sesión de terapia VR/AR
        
        Args:
            user_id: ID del usuario
            therapy_type: Tipo de terapia
            scenario_id: ID del escenario (opcional)
            duration_minutes: Duración en minutos
        
        Returns:
            Sesión de terapia creada
        """
        session = {
            "id": f"vr_session_{datetime.now().timestamp()}",
            "user_id": user_id,
            "therapy_type": therapy_type,
            "scenario_id": scenario_id or self._get_default_scenario(therapy_type),
            "duration_minutes": duration_minutes,
            "status": "scheduled",
            "created_at": datetime.now().isoformat(),
            "vr_environment": self._get_vr_environment(therapy_type)
        }
        
        return session
    
    def start_therapy_session(
        self,
        session_id: str,
        user_id: str
    ) -> Dict:
        """
        Inicia sesión de terapia
        
        Args:
            session_id: ID de la sesión
            user_id: ID del usuario
        
        Returns:
            Sesión iniciada
        """
        return {
            "session_id": session_id,
            "user_id": user_id,
            "status": "active",
            "started_at": datetime.now().isoformat(),
            "vr_connection": "established",
            "tracking_enabled": True
        }
    
    def record_session_data(
        self,
        session_id: str,
        user_id: str,
        session_data: Dict
    ) -> Dict:
        """
        Registra datos de sesión
        
        Args:
            session_id: ID de la sesión
            user_id: ID del usuario
            session_data: Datos de la sesión
        
        Returns:
            Datos registrados
        """
        return {
            "session_id": session_id,
            "user_id": user_id,
            "session_data": session_data,
            "recorded_at": datetime.now().isoformat(),
            "metrics": self._calculate_session_metrics(session_data),
            "status": "recorded"
        }
    
    def get_available_scenarios(
        self,
        therapy_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene escenarios disponibles
        
        Args:
            therapy_type: Filtrar por tipo de terapia (opcional)
        
        Returns:
            Lista de escenarios
        """
        if therapy_type:
            return [s for s in self.available_scenarios if s.get("therapy_type") == therapy_type]
        return self.available_scenarios
    
    def _load_scenarios(self) -> List[Dict]:
        """Carga escenarios VR/AR"""
        return [
            {
                "id": "beach_relaxation",
                "therapy_type": TherapyType.RELAXATION,
                "name": "Playa Tranquila",
                "description": "Ambiente relajante de playa",
                "duration_minutes": 20
            },
            {
                "id": "social_cafe",
                "therapy_type": TherapyType.SOCIAL_SKILLS,
                "name": "Café Social",
                "description": "Practica habilidades sociales en un café",
                "duration_minutes": 30
            },
            {
                "id": "mindfulness_garden",
                "therapy_type": TherapyType.MINDFULNESS,
                "name": "Jardín de Mindfulness",
                "description": "Jardín virtual para meditación",
                "duration_minutes": 15
            }
        ]
    
    def _get_default_scenario(self, therapy_type: str) -> str:
        """Obtiene escenario por defecto"""
        defaults = {
            TherapyType.RELAXATION: "beach_relaxation",
            TherapyType.SOCIAL_SKILLS: "social_cafe",
            TherapyType.MINDFULNESS: "mindfulness_garden"
        }
        return defaults.get(therapy_type, "beach_relaxation")
    
    def _get_vr_environment(self, therapy_type: str) -> Dict:
        """Obtiene ambiente VR"""
        return {
            "environment": "immersive",
            "quality": "high",
            "interactivity": "full"
        }
    
    def _calculate_session_metrics(self, session_data: Dict) -> Dict:
        """Calcula métricas de sesión"""
        return {
            "engagement_score": 8.5,
            "completion_rate": 1.0,
            "stress_reduction": 0.3
        }

