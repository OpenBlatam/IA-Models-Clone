"""
Servicio de Desafíos y Competencias - Sistema de desafíos para motivación
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class ChallengeType(str, Enum):
    """Tipos de desafíos"""
    SOBRIETY = "sobriety"
    HEALTH = "health"
    SOCIAL = "social"
    PERSONAL = "personal"
    COMMUNITY = "community"


class ChallengeStatus(str, Enum):
    """Estados de desafío"""
    AVAILABLE = "available"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class ChallengeService:
    """Servicio de desafíos y competencias"""
    
    def __init__(self):
        """Inicializa el servicio de desafíos"""
        self.challenge_templates = self._load_challenge_templates()
    
    def create_challenge(
        self,
        user_id: str,
        challenge_type: str,
        title: str,
        description: str,
        duration_days: int,
        target_value: Optional[float] = None,
        reward_points: int = 0
    ) -> Dict:
        """
        Crea un nuevo desafío
        
        Args:
            user_id: ID del usuario
            challenge_type: Tipo de desafío
            title: Título
            description: Descripción
            duration_days: Duración en días
            target_value: Valor objetivo (opcional)
            reward_points: Puntos de recompensa
        
        Returns:
            Desafío creado
        """
        challenge = {
            "id": f"challenge_{datetime.now().timestamp()}",
            "user_id": user_id,
            "challenge_type": challenge_type,
            "title": title,
            "description": description,
            "duration_days": duration_days,
            "target_value": target_value,
            "current_value": 0,
            "reward_points": reward_points,
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() + timedelta(days=duration_days)).isoformat(),
            "status": ChallengeStatus.ACTIVE,
            "created_at": datetime.now().isoformat()
        }
        
        return challenge
    
    def get_available_challenges(
        self,
        user_id: str,
        challenge_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene desafíos disponibles para el usuario
        
        Args:
            user_id: ID del usuario
            challenge_type: Filtrar por tipo (opcional)
        
        Returns:
            Lista de desafíos disponibles
        """
        challenges = [
            {
                "id": "challenge_1",
                "challenge_type": ChallengeType.SOBRIETY,
                "title": "7 Días de Desafío",
                "description": "Completa 7 días consecutivos sin consumo",
                "duration_days": 7,
                "reward_points": 100,
                "difficulty": "easy"
            },
            {
                "id": "challenge_2",
                "challenge_type": ChallengeType.HEALTH,
                "title": "Ejercicio Semanal",
                "description": "Haz ejercicio al menos 3 veces esta semana",
                "duration_days": 7,
                "reward_points": 75,
                "difficulty": "medium"
            },
            {
                "id": "challenge_3",
                "challenge_type": ChallengeType.SOCIAL,
                "title": "Conexión Social",
                "description": "Contacta con tu sistema de apoyo 5 veces esta semana",
                "duration_days": 7,
                "reward_points": 50,
                "difficulty": "easy"
            }
        ]
        
        if challenge_type:
            challenges = [c for c in challenges if c.get("challenge_type") == challenge_type]
        
        return challenges
    
    def join_challenge(
        self,
        user_id: str,
        challenge_id: str
    ) -> Dict:
        """
        Une un usuario a un desafío
        
        Args:
            user_id: ID del usuario
            challenge_id: ID del desafío
        
        Returns:
            Participación en desafío
        """
        participation = {
            "user_id": user_id,
            "challenge_id": challenge_id,
            "joined_at": datetime.now().isoformat(),
            "status": ChallengeStatus.ACTIVE,
            "progress": 0
        }
        
        return participation
    
    def update_challenge_progress(
        self,
        challenge_id: str,
        user_id: str,
        progress_value: float
    ) -> Dict:
        """
        Actualiza progreso de un desafío
        
        Args:
            challenge_id: ID del desafío
            user_id: ID del usuario
            progress_value: Valor de progreso
        
        Returns:
            Progreso actualizado
        """
        return {
            "challenge_id": challenge_id,
            "user_id": user_id,
            "progress_value": progress_value,
            "updated_at": datetime.now().isoformat()
        }
    
    def complete_challenge(
        self,
        challenge_id: str,
        user_id: str
    ) -> Dict:
        """
        Completa un desafío
        
        Args:
            challenge_id: ID del desafío
            user_id: ID del usuario
        
        Returns:
            Desafío completado
        """
        return {
            "challenge_id": challenge_id,
            "user_id": user_id,
            "status": ChallengeStatus.COMPLETED,
            "completed_at": datetime.now().isoformat(),
            "reward_earned": True
        }
    
    def get_user_challenges(
        self,
        user_id: str,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene desafíos del usuario
        
        Args:
            user_id: ID del usuario
            status: Filtrar por estado (opcional)
        
        Returns:
            Lista de desafíos del usuario
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def create_community_challenge(
        self,
        challenge_type: str,
        title: str,
        description: str,
        duration_days: int,
        max_participants: int = 100
    ) -> Dict:
        """
        Crea un desafío comunitario
        
        Args:
            challenge_type: Tipo de desafío
            title: Título
            description: Descripción
            duration_days: Duración
            max_participants: Máximo de participantes
        
        Returns:
            Desafío comunitario creado
        """
        challenge = {
            "id": f"community_challenge_{datetime.now().timestamp()}",
            "challenge_type": challenge_type,
            "title": title,
            "description": description,
            "duration_days": duration_days,
            "max_participants": max_participants,
            "current_participants": 0,
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() + timedelta(days=duration_days)).isoformat(),
            "status": "open",
            "created_at": datetime.now().isoformat()
        }
        
        return challenge
    
    def _load_challenge_templates(self) -> List[Dict]:
        """Carga plantillas de desafíos"""
        return [
            {
                "type": ChallengeType.SOBRIETY,
                "title_template": "{days} Días de Desafío",
                "description_template": "Completa {days} días consecutivos sin consumo",
                "default_duration": 7
            },
            {
                "type": ChallengeType.HEALTH,
                "title_template": "Desafío de Ejercicio",
                "description_template": "Haz ejercicio {times} veces esta semana",
                "default_duration": 7
            }
        ]

