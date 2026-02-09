"""
Servicio de Mentoría Virtual - Sistema de mentores y mentees
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class MentorshipStatus(str, Enum):
    """Estados de mentoría"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MentorshipService:
    """Servicio de mentoría virtual"""
    
    def __init__(self):
        """Inicializa el servicio de mentoría"""
        pass
    
    def create_mentorship_request(
        self,
        mentee_id: str,
        preferences: Dict,
        goals: List[str]
    ) -> Dict:
        """
        Crea una solicitud de mentoría
        
        Args:
            mentee_id: ID del mentee
            preferences: Preferencias (tipo de adicción, experiencia, etc.)
            goals: Objetivos de la mentoría
        
        Returns:
            Solicitud de mentoría creada
        """
        request = {
            "id": f"mentorship_{datetime.now().timestamp()}",
            "mentee_id": mentee_id,
            "preferences": preferences,
            "goals": goals,
            "status": MentorshipStatus.PENDING,
            "created_at": datetime.now().isoformat()
        }
        
        return request
    
    def match_mentor(
        self,
        mentee_id: str,
        preferences: Dict
    ) -> Dict:
        """
        Encuentra un mentor compatible
        
        Args:
            mentee_id: ID del mentee
            preferences: Preferencias del mentee
        
        Returns:
            Mentor compatible
        """
        # En implementación real, esto buscaría en base de datos
        matched_mentor = {
            "mentor_id": "mentor_1",
            "name": "Mentor Ejemplo",
            "experience_years": 5,
            "addiction_type": preferences.get("addiction_type", "general"),
            "success_rate": 0.85,
            "match_score": 0.92,
            "availability": "available",
            "languages": ["español", "inglés"]
        }
        
        return matched_mentor
    
    def create_mentorship_session(
        self,
        mentorship_id: str,
        session_type: str,
        scheduled_time: datetime,
        duration_minutes: int = 30
    ) -> Dict:
        """
        Crea una sesión de mentoría
        
        Args:
            mentorship_id: ID de la mentoría
            session_type: Tipo de sesión (video, chat, phone)
            scheduled_time: Hora programada
            duration_minutes: Duración en minutos
        
        Returns:
            Sesión creada
        """
        session = {
            "id": f"session_{datetime.now().timestamp()}",
            "mentorship_id": mentorship_id,
            "session_type": session_type,
            "scheduled_time": scheduled_time.isoformat(),
            "duration_minutes": duration_minutes,
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }
        
        return session
    
    def get_mentor_resources(
        self,
        mentor_id: str
    ) -> Dict:
        """
        Obtiene recursos del mentor
        
        Args:
            mentor_id: ID del mentor
        
        Returns:
            Recursos del mentor
        """
        return {
            "mentor_id": mentor_id,
            "resources": [
                {
                    "type": "guide",
                    "title": "Guía de Primeros Pasos",
                    "description": "Recursos para comenzar la recuperación"
                },
                {
                    "type": "worksheet",
                    "title": "Ejercicios de Reflexión",
                    "description": "Actividades para autoevaluación"
                },
                {
                    "type": "video",
                    "title": "Técnicas de Afrontamiento",
                    "description": "Videos educativos sobre manejo de cravings"
                }
            ]
        }
    
    def track_mentorship_progress(
        self,
        mentorship_id: str,
        milestones: List[Dict]
    ) -> Dict:
        """
        Rastrea progreso de la mentoría
        
        Args:
            mentorship_id: ID de la mentoría
            milestones: Hitos alcanzados
        
        Returns:
            Progreso de la mentoría
        """
        return {
            "mentorship_id": mentorship_id,
            "total_sessions": len(milestones),
            "completed_milestones": sum(1 for m in milestones if m.get("completed", False)),
            "progress_percentage": (sum(1 for m in milestones if m.get("completed", False)) / len(milestones) * 100) if milestones else 0,
            "next_milestone": milestones[0] if milestones else None,
            "updated_at": datetime.now().isoformat()
        }
    
    def get_available_mentors(
        self,
        addiction_type: Optional[str] = None,
        experience_min: Optional[int] = None
    ) -> List[Dict]:
        """
        Obtiene mentores disponibles
        
        Args:
            addiction_type: Filtrar por tipo de adicción (opcional)
            experience_min: Experiencia mínima en años (opcional)
        
        Returns:
            Lista de mentores disponibles
        """
        # En implementación real, esto vendría de la base de datos
        mentors = [
            {
                "mentor_id": "mentor_1",
                "name": "Juan Pérez",
                "experience_years": 5,
                "addiction_type": "alcohol",
                "success_rate": 0.88,
                "availability": "available",
                "rating": 4.8
            },
            {
                "mentor_id": "mentor_2",
                "name": "María García",
                "experience_years": 8,
                "addiction_type": "cigarrillos",
                "success_rate": 0.92,
                "availability": "available",
                "rating": 4.9
            }
        ]
        
        if addiction_type:
            mentors = [m for m in mentors if m.get("addiction_type") == addiction_type]
        
        if experience_min:
            mentors = [m for m in mentors if m.get("experience_years", 0) >= experience_min]
        
        return mentors

