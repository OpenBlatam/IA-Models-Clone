"""
Sistema de gamificación
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid


class AchievementType(str, Enum):
    """Tipos de logros"""
    ANALYSIS_COUNT = "analysis_count"
    STREAK = "streak"
    SCORE_IMPROVEMENT = "score_improvement"
    CONSISTENCY = "consistency"
    MILESTONE = "milestone"


@dataclass
class Achievement:
    """Logro"""
    id: str
    name: str
    description: str
    type: AchievementType
    icon: Optional[str] = None
    points: int = 0
    rarity: str = "common"  # common, rare, epic, legendary
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "icon": self.icon,
            "points": self.points,
            "rarity": self.rarity,
            "created_at": self.created_at
        }


@dataclass
class UserAchievement:
    """Logro de usuario"""
    user_id: str
    achievement_id: str
    unlocked_at: str = None
    progress: float = 100.0  # 0-100
    
    def __post_init__(self):
        if self.unlocked_at is None:
            self.unlocked_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "achievement_id": self.achievement_id,
            "unlocked_at": self.unlocked_at,
            "progress": self.progress
        }


class GamificationSystem:
    """Sistema de gamificación"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.achievements: Dict[str, Achievement] = {}
        self.user_achievements: Dict[str, List[str]] = {}  # user_id -> [achievement_ids]
        self.user_points: Dict[str, int] = {}  # user_id -> points
        self.user_streaks: Dict[str, int] = {}  # user_id -> streak_days
        self._initialize_achievements()
    
    def _initialize_achievements(self):
        """Inicializa logros predefinidos"""
        achievements_data = [
            {
                "name": "Primer Análisis",
                "description": "Completa tu primer análisis de piel",
                "type": AchievementType.ANALYSIS_COUNT,
                "points": 10,
                "rarity": "common"
            },
            {
                "name": "Analista Dedicado",
                "description": "Completa 10 análisis",
                "type": AchievementType.ANALYSIS_COUNT,
                "points": 50,
                "rarity": "rare"
            },
            {
                "name": "Racha de 7 Días",
                "description": "Analiza tu piel 7 días seguidos",
                "type": AchievementType.STREAK,
                "points": 100,
                "rarity": "epic"
            },
            {
                "name": "Mejora Continua",
                "description": "Mejora tu score en 10 puntos",
                "type": AchievementType.SCORE_IMPROVEMENT,
                "points": 75,
                "rarity": "rare"
            },
            {
                "name": "Piel Perfecta",
                "description": "Alcanza un score de 90 o más",
                "type": AchievementType.MILESTONE,
                "points": 200,
                "rarity": "legendary"
            }
        ]
        
        for ach_data in achievements_data:
            ach_id = str(uuid.uuid4())
            achievement = Achievement(
                id=ach_id,
                name=ach_data["name"],
                description=ach_data["description"],
                type=ach_data["type"],
                points=ach_data["points"],
                rarity=ach_data["rarity"]
            )
            self.achievements[ach_id] = achievement
    
    def check_achievements(self, user_id: str, event_type: str, data: Dict) -> List[Achievement]:
        """
        Verifica y desbloquea logros
        
        Args:
            user_id: ID del usuario
            event_type: Tipo de evento
            data: Datos del evento
            
        Returns:
            Lista de logros desbloqueados
        """
        unlocked = []
        
        # Verificar logros según tipo de evento
        if event_type == "analysis_completed":
            unlocked.extend(self._check_analysis_achievements(user_id, data))
        elif event_type == "streak_updated":
            unlocked.extend(self._check_streak_achievements(user_id, data))
        elif event_type == "score_improved":
            unlocked.extend(self._check_score_achievements(user_id, data))
        
        return unlocked
    
    def _check_analysis_achievements(self, user_id: str, data: Dict) -> List[Achievement]:
        """Verifica logros de análisis"""
        unlocked = []
        
        # Contar análisis del usuario
        analysis_count = data.get("total_analyses", 0)
        
        # Logro: Primer Análisis
        if analysis_count == 1:
            ach = self._get_achievement_by_name("Primer Análisis")
            if ach and self._unlock_achievement(user_id, ach.id):
                unlocked.append(ach)
        
        # Logro: Analista Dedicado
        if analysis_count == 10:
            ach = self._get_achievement_by_name("Analista Dedicado")
            if ach and self._unlock_achievement(user_id, ach.id):
                unlocked.append(ach)
        
        return unlocked
    
    def _check_streak_achievements(self, user_id: str, data: Dict) -> List[Achievement]:
        """Verifica logros de racha"""
        unlocked = []
        
        streak = data.get("streak_days", 0)
        
        if streak >= 7:
            ach = self._get_achievement_by_name("Racha de 7 Días")
            if ach and self._unlock_achievement(user_id, ach.id):
                unlocked.append(ach)
        
        return unlocked
    
    def _check_score_achievements(self, user_id: str, data: Dict) -> List[Achievement]:
        """Verifica logros de score"""
        unlocked = []
        
        score = data.get("score", 0)
        improvement = data.get("improvement", 0)
        
        # Logro: Piel Perfecta
        if score >= 90:
            ach = self._get_achievement_by_name("Piel Perfecta")
            if ach and self._unlock_achievement(user_id, ach.id):
                unlocked.append(ach)
        
        # Logro: Mejora Continua
        if improvement >= 10:
            ach = self._get_achievement_by_name("Mejora Continua")
            if ach and self._unlock_achievement(user_id, ach.id):
                unlocked.append(ach)
        
        return unlocked
    
    def _get_achievement_by_name(self, name: str) -> Optional[Achievement]:
        """Obtiene logro por nombre"""
        for ach in self.achievements.values():
            if ach.name == name:
                return ach
        return None
    
    def _unlock_achievement(self, user_id: str, achievement_id: str) -> bool:
        """Desbloquea un logro"""
        if user_id not in self.user_achievements:
            self.user_achievements[user_id] = []
        
        if achievement_id in self.user_achievements[user_id]:
            return False  # Ya desbloqueado
        
        self.user_achievements[user_id].append(achievement_id)
        
        # Agregar puntos
        achievement = self.achievements.get(achievement_id)
        if achievement:
            if user_id not in self.user_points:
                self.user_points[user_id] = 0
            self.user_points[user_id] += achievement.points
        
        return True
    
    def get_user_achievements(self, user_id: str) -> List[Achievement]:
        """Obtiene logros de un usuario"""
        achievement_ids = self.user_achievements.get(user_id, [])
        return [self.achievements[aid] for aid in achievement_ids if aid in self.achievements]
    
    def get_user_points(self, user_id: str) -> int:
        """Obtiene puntos de un usuario"""
        return self.user_points.get(user_id, 0)
    
    def get_user_level(self, user_id: str) -> int:
        """Calcula nivel del usuario basado en puntos"""
        points = self.get_user_points(user_id)
        # 100 puntos por nivel
        return (points // 100) + 1
    
    def get_leaderboard(self, limit: int = 100) -> List[Dict]:
        """Obtiene leaderboard"""
        users = [
            {"user_id": uid, "points": points, "level": self.get_user_level(uid)}
            for uid, points in self.user_points.items()
        ]
        
        users.sort(key=lambda x: x["points"], reverse=True)
        return users[:limit]
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Obtiene estadísticas de usuario"""
        return {
            "user_id": user_id,
            "points": self.get_user_points(user_id),
            "level": self.get_user_level(user_id),
            "achievements_count": len(self.user_achievements.get(user_id, [])),
            "streak_days": self.user_streaks.get(user_id, 0),
            "achievements": [a.to_dict() for a in self.get_user_achievements(user_id)]
        }






