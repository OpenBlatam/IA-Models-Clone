"""
Servicio de Terapias Virtuales - Sesiones de terapia y consejería virtual
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class TherapyType(str, Enum):
    """Tipos de terapia"""
    INDIVIDUAL = "individual"
    GROUP = "group"
    COGNITIVE_BEHAVIORAL = "cognitive_behavioral"
    DIALECTICAL_BEHAVIORAL = "dialectical_behavioral"
    MOTIVATIONAL_INTERVIEWING = "motivational_interviewing"
    MINDFULNESS = "mindfulness"


class SessionStatus(str, Enum):
    """Estados de sesión"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    MISSED = "missed"


class VirtualTherapyService:
    """Servicio de terapias virtuales"""
    
    def __init__(self):
        """Inicializa el servicio de terapias virtuales"""
        pass
    
    def schedule_therapy_session(
        self,
        user_id: str,
        therapy_type: str,
        therapist_id: Optional[str] = None,
        scheduled_time: datetime = None,
        duration_minutes: int = 60,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Programa una sesión de terapia
        
        Args:
            user_id: ID del usuario
            therapy_type: Tipo de terapia
            therapist_id: ID del terapeuta (opcional)
            scheduled_time: Hora programada
            duration_minutes: Duración en minutos
            notes: Notas adicionales
        
        Returns:
            Sesión programada
        """
        if scheduled_time is None:
            scheduled_time = datetime.now() + timedelta(days=1)
        
        session = {
            "id": f"session_{datetime.now().timestamp()}",
            "user_id": user_id,
            "therapist_id": therapist_id,
            "therapy_type": therapy_type,
            "scheduled_time": scheduled_time.isoformat(),
            "duration_minutes": duration_minutes,
            "status": SessionStatus.SCHEDULED,
            "notes": notes,
            "created_at": datetime.now().isoformat()
        }
        
        return session
    
    def get_available_therapists(
        self,
        therapy_type: Optional[str] = None,
        specialization: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene terapeutas disponibles
        
        Args:
            therapy_type: Filtrar por tipo de terapia (opcional)
            specialization: Especialización (opcional)
        
        Returns:
            Lista de terapeutas disponibles
        """
        # En implementación real, esto vendría de la base de datos
        therapists = [
            {
                "therapist_id": "therapist_1",
                "name": "Dr. Ana Martínez",
                "specialization": "adicciones",
                "therapy_types": [TherapyType.COGNITIVE_BEHAVIORAL, TherapyType.INDIVIDUAL],
                "experience_years": 10,
                "rating": 4.9,
                "availability": "available",
                "languages": ["español", "inglés"]
            },
            {
                "therapist_id": "therapist_2",
                "name": "Dr. Carlos Rodríguez",
                "specialization": "adicciones",
                "therapy_types": [TherapyType.MOTIVATIONAL_INTERVIEWING, TherapyType.GROUP],
                "experience_years": 8,
                "rating": 4.8,
                "availability": "available",
                "languages": ["español"]
            }
        ]
        
        if therapy_type:
            therapists = [t for t in therapists if therapy_type in t.get("therapy_types", [])]
        
        return therapists
    
    def start_therapy_session(
        self,
        session_id: str
    ) -> Dict:
        """
        Inicia una sesión de terapia
        
        Args:
            session_id: ID de la sesión
        
        Returns:
            Sesión iniciada
        """
        return {
            "session_id": session_id,
            "status": SessionStatus.IN_PROGRESS,
            "started_at": datetime.now().isoformat(),
            "video_link": f"https://therapy.video/{session_id}",  # En implementación real
            "chat_enabled": True
        }
    
    def complete_therapy_session(
        self,
        session_id: str,
        notes: Optional[str] = None,
        homework: Optional[List[str]] = None
    ) -> Dict:
        """
        Completa una sesión de terapia
        
        Args:
            session_id: ID de la sesión
            notes: Notas de la sesión (opcional)
            homework: Tareas asignadas (opcional)
        
        Returns:
            Sesión completada
        """
        return {
            "session_id": session_id,
            "status": SessionStatus.COMPLETED,
            "completed_at": datetime.now().isoformat(),
            "notes": notes,
            "homework": homework or [],
            "duration_actual": 60  # En implementación real, calcular diferencia
        }
    
    def get_therapy_resources(
        self,
        therapy_type: str
    ) -> List[Dict]:
        """
        Obtiene recursos de terapia
        
        Args:
            therapy_type: Tipo de terapia
        
        Returns:
            Lista de recursos
        """
        resources = {
            TherapyType.COGNITIVE_BEHAVIORAL: [
                {
                    "type": "worksheet",
                    "title": "Identificación de Pensamientos Negativos",
                    "description": "Ejercicio para identificar y desafiar pensamientos negativos"
                },
                {
                    "type": "video",
                    "title": "Técnicas de CBT",
                    "description": "Video educativo sobre técnicas de terapia cognitivo-conductual"
                }
            ],
            TherapyType.MINDFULNESS: [
                {
                    "type": "audio",
                    "title": "Meditación Guiada",
                    "description": "Sesión de meditación guiada de 10 minutos"
                },
                {
                    "type": "exercise",
                    "title": "Ejercicio de Respiración",
                    "description": "Técnica de respiración para reducir ansiedad"
                }
            ]
        }
        
        return resources.get(therapy_type, [])
    
    def get_user_therapy_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Obtiene historial de terapia del usuario
        
        Args:
            user_id: ID del usuario
            limit: Límite de resultados
        
        Returns:
            Historial de terapia
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def create_group_therapy_session(
        self,
        therapy_type: str,
        max_participants: int = 8,
        scheduled_time: datetime = None,
        topic: Optional[str] = None
    ) -> Dict:
        """
        Crea una sesión de terapia grupal
        
        Args:
            therapy_type: Tipo de terapia
            max_participants: Máximo de participantes
            scheduled_time: Hora programada
            topic: Tema de la sesión (opcional)
        
        Returns:
            Sesión grupal creada
        """
        if scheduled_time is None:
            scheduled_time = datetime.now() + timedelta(days=1)
        
        return {
            "id": f"group_session_{datetime.now().timestamp()}",
            "therapy_type": therapy_type,
            "max_participants": max_participants,
            "current_participants": 0,
            "scheduled_time": scheduled_time.isoformat(),
            "topic": topic,
            "status": "open_for_registration",
            "created_at": datetime.now().isoformat()
        }

