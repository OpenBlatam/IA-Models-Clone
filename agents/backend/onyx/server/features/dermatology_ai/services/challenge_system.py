"""
Sistema de challenges
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import uuid


class ChallengeStatus(str, Enum):
    """Estado del challenge"""
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    LOCKED = "locked"


@dataclass
class Challenge:
    """Challenge"""
    id: str
    name: str
    description: str
    objective: str
    target_value: float
    reward_points: int
    start_date: str
    end_date: str
    status: ChallengeStatus = ChallengeStatus.ACTIVE
    category: str = "general"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "objective": self.objective,
            "target_value": self.target_value,
            "reward_points": self.reward_points,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status.value,
            "category": self.category
        }


@dataclass
class UserChallenge:
    """Challenge de usuario"""
    user_id: str
    challenge_id: str
    progress: float = 0.0
    status: ChallengeStatus = ChallengeStatus.ACTIVE
    started_at: str = None
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "challenge_id": self.challenge_id,
            "progress": self.progress,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class ChallengeSystem:
    """Sistema de challenges"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.challenges: Dict[str, Challenge] = {}
        self.user_challenges: Dict[str, List[UserChallenge]] = {}  # user_id -> [challenges]
        self._initialize_default_challenges()
    
    def _initialize_default_challenges(self):
        """Inicializa challenges por defecto"""
        now = datetime.now()
        
        challenges_data = [
            {
                "name": "Análisis Semanal",
                "description": "Completa 7 análisis esta semana",
                "objective": "analysis_count",
                "target_value": 7,
                "reward_points": 150,
                "start_date": now.isoformat(),
                "end_date": (now + timedelta(days=7)).isoformat(),
                "category": "consistency"
            },
            {
                "name": "Mejora tu Score",
                "description": "Mejora tu score en 5 puntos",
                "objective": "score_improvement",
                "target_value": 5,
                "reward_points": 100,
                "start_date": now.isoformat(),
                "end_date": (now + timedelta(days=30)).isoformat(),
                "category": "improvement"
            },
            {
                "name": "Rutina Completa",
                "description": "Sigue tu rutina recomendada por 14 días",
                "objective": "routine_consistency",
                "target_value": 14,
                "reward_points": 200,
                "start_date": now.isoformat(),
                "end_date": (now + timedelta(days=30)).isoformat(),
                "category": "routine"
            }
        ]
        
        for ch_data in challenges_data:
            ch_id = str(uuid.uuid4())
            challenge = Challenge(
                id=ch_id,
                name=ch_data["name"],
                description=ch_data["description"],
                objective=ch_data["objective"],
                target_value=ch_data["target_value"],
                reward_points=ch_data["reward_points"],
                start_date=ch_data["start_date"],
                end_date=ch_data["end_date"],
                category=ch_data["category"]
            )
            self.challenges[ch_id] = challenge
    
    def get_available_challenges(self, user_id: str) -> List[Challenge]:
        """Obtiene challenges disponibles para un usuario"""
        now = datetime.now()
        available = []
        
        for challenge in self.challenges.values():
            # Verificar si está activo
            start = datetime.fromisoformat(challenge.start_date)
            end = datetime.fromisoformat(challenge.end_date)
            
            if start <= now <= end:
                # Verificar si el usuario ya lo tiene
                user_challenges = self.user_challenges.get(user_id, [])
                if not any(uc.challenge_id == challenge.id for uc in user_challenges):
                    available.append(challenge)
        
        return available
    
    def start_challenge(self, user_id: str, challenge_id: str) -> bool:
        """
        Inicia un challenge
        
        Args:
            user_id: ID del usuario
            challenge_id: ID del challenge
            
        Returns:
            True si se inició correctamente
        """
        if challenge_id not in self.challenges:
            return False
        
        if user_id not in self.user_challenges:
            self.user_challenges[user_id] = []
        
        # Verificar si ya está activo
        if any(uc.challenge_id == challenge_id and uc.status == ChallengeStatus.ACTIVE
               for uc in self.user_challenges[user_id]):
            return False
        
        user_challenge = UserChallenge(
            user_id=user_id,
            challenge_id=challenge_id
        )
        
        self.user_challenges[user_id].append(user_challenge)
        return True
    
    def update_challenge_progress(self, user_id: str, challenge_id: str, progress: float):
        """
        Actualiza progreso de un challenge
        
        Args:
            user_id: ID del usuario
            challenge_id: ID del challenge
            progress: Progreso (0-100)
        """
        user_challenges = self.user_challenges.get(user_id, [])
        
        for uc in user_challenges:
            if uc.challenge_id == challenge_id and uc.status == ChallengeStatus.ACTIVE:
                uc.progress = min(100.0, max(0.0, progress))
                
                # Verificar si está completado
                challenge = self.challenges.get(challenge_id)
                if challenge and uc.progress >= 100.0:
                    uc.status = ChallengeStatus.COMPLETED
                    uc.completed_at = datetime.now().isoformat()
                
                break
    
    def get_user_challenges(self, user_id: str) -> List[Dict]:
        """Obtiene challenges de un usuario"""
        user_challenges = self.user_challenges.get(user_id, [])
        
        result = []
        for uc in user_challenges:
            challenge = self.challenges.get(uc.challenge_id)
            if challenge:
                result.append({
                    "challenge": challenge.to_dict(),
                    "user_progress": uc.to_dict()
                })
        
        return result






