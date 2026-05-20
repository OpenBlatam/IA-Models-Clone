"""
Challenges Service - Sistema de desafíos y misiones
===================================================

Sistema de desafíos gamificados para mantener al usuario motivado.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ChallengeType(str, Enum):
    """Tipos de desafíos"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ACHIEVEMENT = "achievement"
    SPECIAL = "special"


class ChallengeStatus(str, Enum):
    """Estado del desafío"""
    LOCKED = "locked"
    AVAILABLE = "available"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class Challenge:
    """Representa un desafío"""
    id: str
    title: str
    description: str
    type: ChallengeType
    difficulty: str  # easy, medium, hard
    points_reward: int
    xp_reward: int
    badge_reward: Optional[str] = None
    requirements: Dict[str, Any] = field(default_factory=dict)
    status: ChallengeStatus = ChallengeStatus.AVAILABLE
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 a 1.0


class ChallengesService:
    """Servicio de desafíos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.user_challenges: Dict[str, List[Challenge]] = {}
        self.available_challenges = self._initialize_challenges()
        logger.info("ChallengesService initialized")
    
    def get_available_challenges(
        self,
        user_id: str,
        challenge_type: Optional[ChallengeType] = None
    ) -> List[Challenge]:
        """Obtener desafíos disponibles para el usuario"""
        if user_id not in self.user_challenges:
            self.user_challenges[user_id] = []
        
        # Filtrar por tipo si se especifica
        challenges = self.available_challenges
        if challenge_type:
            challenges = [c for c in challenges if c.type == challenge_type]
        
        # Actualizar estado de desafíos
        for challenge in challenges:
            challenge.status = self._determine_challenge_status(user_id, challenge)
        
        return challenges
    
    def start_challenge(self, user_id: str, challenge_id: str) -> Challenge:
        """Iniciar un desafío"""
        challenge = next(
            (c for c in self.available_challenges if c.id == challenge_id),
            None
        )
        
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} not found")
        
        if challenge.status != ChallengeStatus.AVAILABLE:
            raise ValueError(f"Challenge {challenge_id} is not available")
        
        challenge.status = ChallengeStatus.IN_PROGRESS
        challenge.started_at = datetime.now()
        
        # Establecer fecha de expiración según tipo
        if challenge.type == ChallengeType.DAILY:
            challenge.expires_at = datetime.now() + timedelta(days=1)
        elif challenge.type == ChallengeType.WEEKLY:
            challenge.expires_at = datetime.now() + timedelta(weeks=1)
        elif challenge.type == ChallengeType.MONTHLY:
            challenge.expires_at = datetime.now() + timedelta(days=30)
        
        if user_id not in self.user_challenges:
            self.user_challenges[user_id] = []
        
        # Agregar si no existe
        if not any(c.id == challenge_id for c in self.user_challenges[user_id]):
            self.user_challenges[user_id].append(challenge)
        
        logger.info(f"Challenge {challenge_id} started for user {user_id}")
        return challenge
    
    def update_challenge_progress(
        self,
        user_id: str,
        challenge_id: str,
        progress: float
    ) -> Challenge:
        """Actualizar progreso de un desafío"""
        challenge = self._get_user_challenge(user_id, challenge_id)
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} not found for user {user_id}")
        
        challenge.progress = min(max(progress, 0.0), 1.0)
        
        # Verificar si está completo
        if challenge.progress >= 1.0:
            challenge.status = ChallengeStatus.COMPLETED
            challenge.completed_at = datetime.now()
        
        return challenge
    
    def complete_challenge(
        self,
        user_id: str,
        challenge_id: str
    ) -> Dict[str, Any]:
        """Completar un desafío"""
        challenge = self._get_user_challenge(user_id, challenge_id)
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} not found")
        
        challenge.status = ChallengeStatus.COMPLETED
        challenge.completed_at = datetime.now()
        challenge.progress = 1.0
        
        rewards = {
            "points": challenge.points_reward,
            "xp": challenge.xp_reward,
            "badge": challenge.badge_reward,
        }
        
        logger.info(f"Challenge {challenge_id} completed by user {user_id}")
        return {
            "challenge_id": challenge_id,
            "rewards": rewards,
            "completed_at": challenge.completed_at.isoformat(),
        }
    
    def _initialize_challenges(self) -> List[Challenge]:
        """Inicializar desafíos disponibles"""
        challenges = []
        
        # Desafíos diarios
        challenges.append(Challenge(
            id="daily_login",
            title="Login Diario",
            description="Inicia sesión todos los días",
            type=ChallengeType.DAILY,
            difficulty="easy",
            points_reward=20,
            xp_reward=20,
            requirements={"action": "login", "count": 1}
        ))
        
        challenges.append(Challenge(
            id="daily_job_search",
            title="Búsqueda Diaria",
            description="Busca al menos 5 trabajos hoy",
            type=ChallengeType.DAILY,
            difficulty="medium",
            points_reward=50,
            xp_reward=50,
            requirements={"action": "search_jobs", "count": 5}
        ))
        
        challenges.append(Challenge(
            id="daily_step",
            title="Paso del Día",
            description="Completa un paso de tu roadmap",
            type=ChallengeType.DAILY,
            difficulty="medium",
            points_reward=75,
            xp_reward=75,
            requirements={"action": "complete_step", "count": 1}
        ))
        
        # Desafíos semanales
        challenges.append(Challenge(
            id="weekly_applications",
            title="Aplicaciones Semanales",
            description="Aplica a 10 trabajos esta semana",
            type=ChallengeType.WEEKLY,
            difficulty="hard",
            points_reward=500,
            xp_reward=500,
            badge_reward="application_master",
            requirements={"action": "apply_job", "count": 10}
        ))
        
        challenges.append(Challenge(
            id="weekly_skills",
            title="Aprendizaje Semanal",
            description="Aprende 3 nuevas habilidades esta semana",
            type=ChallengeType.WEEKLY,
            difficulty="hard",
            points_reward=400,
            xp_reward=400,
            requirements={"action": "learn_skill", "count": 3}
        ))
        
        # Desafíos mensuales
        challenges.append(Challenge(
            id="monthly_streak",
            title="Racha Mensual",
            description="Mantén una racha de 30 días",
            type=ChallengeType.MONTHLY,
            difficulty="hard",
            points_reward=2000,
            xp_reward=2000,
            badge_reward="streak_master",
            requirements={"action": "maintain_streak", "days": 30}
        ))
        
        # Logros especiales
        challenges.append(Challenge(
            id="first_application",
            title="Primera Aplicación",
            description="Envía tu primera aplicación",
            type=ChallengeType.ACHIEVEMENT,
            difficulty="easy",
            points_reward=100,
            xp_reward=100,
            badge_reward="first_application",
            requirements={"action": "apply_job", "count": 1}
        ))
        
        challenges.append(Challenge(
            id="profile_complete",
            title="Perfil Completo",
            description="Completa tu perfil al 100%",
            type=ChallengeType.ACHIEVEMENT,
            difficulty="medium",
            points_reward=200,
            xp_reward=200,
            badge_reward="profile_complete",
            requirements={"action": "complete_profile", "percentage": 100}
        ))
        
        return challenges
    
    def _determine_challenge_status(
        self,
        user_id: str,
        challenge: Challenge
    ) -> ChallengeStatus:
        """Determinar estado del desafío"""
        user_challenge = self._get_user_challenge(user_id, challenge.id)
        
        if user_challenge:
            if user_challenge.status == ChallengeStatus.COMPLETED:
                return ChallengeStatus.COMPLETED
            elif user_challenge.status == ChallengeStatus.IN_PROGRESS:
                # Verificar si expiró
                if user_challenge.expires_at and datetime.now() > user_challenge.expires_at:
                    return ChallengeStatus.EXPIRED
                return ChallengeStatus.IN_PROGRESS
        
        # Verificar prerrequisitos
        if challenge.requirements.get("prerequisite"):
            prereq = self._get_user_challenge(user_id, challenge.requirements["prerequisite"])
            if not prereq or prereq.status != ChallengeStatus.COMPLETED:
                return ChallengeStatus.LOCKED
        
        return ChallengeStatus.AVAILABLE
    
    def _get_user_challenge(
        self,
        user_id: str,
        challenge_id: str
    ) -> Optional[Challenge]:
        """Obtener desafío del usuario"""
        if user_id not in self.user_challenges:
            return None
        
        return next(
            (c for c in self.user_challenges[user_id] if c.id == challenge_id),
            None
        )
    
    def get_user_challenges(self, user_id: str) -> List[Challenge]:
        """Obtener todos los desafíos del usuario"""
        return self.user_challenges.get(user_id, [])




