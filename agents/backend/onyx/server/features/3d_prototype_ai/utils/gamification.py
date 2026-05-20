"""
Gamification - Sistema de gamificación
=======================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class AchievementType(str, Enum):
    """Tipos de logros"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


class Gamification:
    """Sistema de gamificación"""
    
    def __init__(self):
        self.user_points: Dict[str, int] = defaultdict(int)
        self.user_levels: Dict[str, int] = defaultdict(int)
        self.user_achievements: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.leaderboards: Dict[str, List[Dict[str, Any]]] = {}
        self.achievements_config: Dict[str, Dict[str, Any]] = {}
    
    def award_points(self, user_id: str, points: int, reason: str):
        """Otorga puntos a un usuario"""
        self.user_points[user_id] += points
        
        # Verificar si sube de nivel
        old_level = self.user_levels[user_id]
        new_level = self._calculate_level(self.user_points[user_id])
        
        if new_level > old_level:
            self.user_levels[user_id] = new_level
            logger.info(f"Usuario {user_id} subió al nivel {new_level}")
        
        logger.info(f"Puntos otorgados: {user_id} +{points} ({reason})")
    
    def _calculate_level(self, points: int) -> int:
        """Calcula nivel basado en puntos"""
        # Fórmula: nivel = sqrt(puntos / 100)
        import math
        return int(math.sqrt(points / 100)) + 1
    
    def unlock_achievement(self, user_id: str, achievement_id: str,
                          achievement_name: str, achievement_type: AchievementType,
                          description: str = ""):
        """Desbloquea un logro"""
        achievement = {
            "id": achievement_id,
            "name": achievement_name,
            "type": achievement_type.value,
            "description": description,
            "unlocked_at": datetime.now().isoformat()
        }
        
        # Verificar si ya lo tiene
        existing = [a for a in self.user_achievements[user_id] if a["id"] == achievement_id]
        if existing:
            return
        
        self.user_achievements[user_id].append(achievement)
        
        # Otorgar puntos por logro
        points_map = {
            AchievementType.BRONZE: 10,
            AchievementType.SILVER: 25,
            AchievementType.GOLD: 50,
            AchievementType.PLATINUM: 100,
            AchievementType.DIAMOND: 250
        }
        
        points = points_map.get(achievement_type, 10)
        self.award_points(user_id, points, f"Achievement: {achievement_name}")
        
        logger.info(f"Logro desbloqueado: {user_id} - {achievement_name}")
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de usuario"""
        return {
            "user_id": user_id,
            "points": self.user_points[user_id],
            "level": self.user_levels[user_id],
            "achievements": self.user_achievements[user_id],
            "achievements_count": len(self.user_achievements[user_id]),
            "next_level_points": self._get_next_level_points(self.user_levels[user_id])
        }
    
    def _get_next_level_points(self, current_level: int) -> int:
        """Obtiene puntos necesarios para siguiente nivel"""
        next_level = current_level + 1
        return (next_level ** 2) * 100
    
    def get_leaderboard(self, leaderboard_type: str = "points", limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene leaderboard"""
        if leaderboard_type == "points":
            sorted_users = sorted(
                self.user_points.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            leaderboard = [
                {
                    "user_id": user_id,
                    "points": points,
                    "level": self.user_levels[user_id],
                    "rank": idx + 1
                }
                for idx, (user_id, points) in enumerate(sorted_users[:limit])
            ]
        else:
            # Otros tipos de leaderboard
            leaderboard = []
        
        self.leaderboards[leaderboard_type] = leaderboard
        return leaderboard
    
    def check_achievements(self, user_id: str, action: str, context: Dict[str, Any]):
        """Verifica y desbloquea logros"""
        points = self.user_points[user_id]
        level = self.user_levels[user_id]
        achievements = len(self.user_achievements[user_id])
        
        # Logro: Primer prototipo
        if action == "prototype_created" and achievements == 0:
            self.unlock_achievement(
                user_id, "first_prototype", "Primer Prototipo",
                AchievementType.BRONZE, "Crea tu primer prototipo"
            )
        
        # Logro: Nivel 10
        if level >= 10 and not any(a["id"] == "level_10" for a in self.user_achievements[user_id]):
            self.unlock_achievement(
                user_id, "level_10", "Nivel 10",
                AchievementType.SILVER, "Alcanza el nivel 10"
            )
        
        # Logro: 1000 puntos
        if points >= 1000 and not any(a["id"] == "points_1000" for a in self.user_achievements[user_id]):
            self.unlock_achievement(
                user_id, "points_1000", "1000 Puntos",
                AchievementType.GOLD, "Acumula 1000 puntos"
            )




