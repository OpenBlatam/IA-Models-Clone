"""
Gamification Service - Sistema de gamificación
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AchievementType(str, Enum):
    """Tipos de logros"""
    DESIGN_CREATED = "design_created"
    ANALYSIS_COMPLETED = "analysis_completed"
    FIRST_EXPORT = "first_export"
    FEEDBACK_GIVEN = "feedback_given"
    COLLABORATION = "collaboration"
    MILESTONE = "milestone"


class GamificationService:
    """Servicio para gamificación"""
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.achievements: Dict[str, List[Dict[str, Any]]] = {}
        self.leaderboard: List[Dict[str, Any]] = []
    
    def initialize_user(self, user_id: str) -> Dict[str, Any]:
        """Inicializar perfil de usuario"""
        profile = {
            "user_id": user_id,
            "level": 1,
            "experience_points": 0,
            "badges": [],
            "achievements": [],
            "streak_days": 0,
            "last_activity": None,
            "total_designs": 0,
            "created_at": datetime.now().isoformat()
        }
        
        self.user_profiles[user_id] = profile
        
        return profile
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Obtener perfil de usuario"""
        if user_id not in self.user_profiles:
            return self.initialize_user(user_id)
        
        return self.user_profiles[user_id]
    
    def award_points(
        self,
        user_id: str,
        points: int,
        reason: str
    ) -> Dict[str, Any]:
        """Otorgar puntos de experiencia"""
        profile = self.get_user_profile(user_id)
        
        old_xp = profile["experience_points"]
        profile["experience_points"] += points
        new_xp = profile["experience_points"]
        
        # Calcular nivel (cada 1000 XP = 1 nivel)
        old_level = profile["level"]
        new_level = (new_xp // 1000) + 1
        profile["level"] = new_level
        
        # Verificar logros
        achievements_unlocked = self._check_achievements(user_id, profile)
        
        result = {
            "user_id": user_id,
            "points_awarded": points,
            "reason": reason,
            "old_xp": old_xp,
            "new_xp": new_xp,
            "old_level": old_level,
            "new_level": new_level,
            "leveled_up": new_level > old_level,
            "achievements_unlocked": achievements_unlocked
        }
        
        if new_level > old_level:
            result["level_up_message"] = f"¡Subiste al nivel {new_level}!"
        
        return result
    
    def _check_achievements(
        self,
        user_id: str,
        profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Verificar logros desbloqueados"""
        unlocked = []
        
        # Logro: Primer diseño
        if profile["total_designs"] == 1 and AchievementType.DESIGN_CREATED.value not in profile["achievements"]:
            achievement = self._create_achievement(AchievementType.DESIGN_CREATED, "Primer Diseño", "Creaste tu primer diseño")
            unlocked.append(achievement)
            profile["achievements"].append(AchievementType.DESIGN_CREATED.value)
        
        # Logro: 10 diseños
        if profile["total_designs"] == 10 and "10_designs" not in profile["achievements"]:
            achievement = self._create_achievement(AchievementType.MILESTONE, "Diseñador Prolífico", "Creaste 10 diseños")
            unlocked.append(achievement)
            profile["achievements"].append("10_designs")
        
        # Logro: Nivel 5
        if profile["level"] == 5 and "level_5" not in profile["achievements"]:
            achievement = self._create_achievement(AchievementType.MILESTONE, "Experto", "Alcanzaste el nivel 5")
            unlocked.append(achievement)
            profile["achievements"].append("level_5")
        
        if unlocked:
            if user_id not in self.achievements:
                self.achievements[user_id] = []
            self.achievements[user_id].extend(unlocked)
        
        return unlocked
    
    def _create_achievement(
        self,
        achievement_type: AchievementType,
        title: str,
        description: str
    ) -> Dict[str, Any]:
        """Crear logro"""
        return {
            "achievement_id": f"ach_{len(self.achievements.get('temp', [])) + 1}",
            "type": achievement_type.value,
            "title": title,
            "description": description,
            "unlocked_at": datetime.now().isoformat(),
            "points_reward": 100
        }
    
    def get_achievements(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener logros de usuario"""
        return self.achievements.get(user_id, [])
    
    def get_leaderboard(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Obtener leaderboard"""
        # Actualizar leaderboard
        self._update_leaderboard()
        
        return self.leaderboard[:limit]
    
    def _update_leaderboard(self):
        """Actualizar leaderboard"""
        leaderboard = []
        
        for user_id, profile in self.user_profiles.items():
            leaderboard.append({
                "user_id": user_id,
                "level": profile["level"],
                "experience_points": profile["experience_points"],
                "total_designs": profile["total_designs"],
                "achievements_count": len(profile["achievements"])
            })
        
        # Ordenar por XP
        leaderboard.sort(key=lambda x: x["experience_points"], reverse=True)
        
        # Agregar ranking
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
        
        self.leaderboard = leaderboard
    
    def get_badges(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener badges de usuario"""
        profile = self.get_user_profile(user_id)
        
        badges = []
        
        # Badge por nivel
        if profile["level"] >= 10:
            badges.append({"name": "Master Designer", "icon": "🏆", "unlocked": True})
        elif profile["level"] >= 5:
            badges.append({"name": "Expert Designer", "icon": "⭐", "unlocked": True})
        elif profile["level"] >= 2:
            badges.append({"name": "Designer", "icon": "🎨", "unlocked": True})
        
        # Badge por diseños
        if profile["total_designs"] >= 50:
            badges.append({"name": "Design Factory", "icon": "🏭", "unlocked": True})
        elif profile["total_designs"] >= 20:
            badges.append({"name": "Creative Pro", "icon": "✨", "unlocked": True})
        elif profile["total_designs"] >= 5:
            badges.append({"name": "Designer", "icon": "📐", "unlocked": True})
        
        return badges




