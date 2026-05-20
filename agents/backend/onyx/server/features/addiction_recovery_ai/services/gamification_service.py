"""
Servicio de gamificación - Sistema de puntos, logros y recompensas
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class AchievementType(str, Enum):
    """Tipos de logros"""
    MILESTONE = "milestone"
    STREAK = "streak"
    CONSISTENCY = "consistency"
    SUPPORT = "support"
    SPECIAL = "special"


class GamificationService:
    """Servicio de gamificación y recompensas"""
    
    def __init__(self):
        """Inicializa el servicio de gamificación"""
        self.achievements = self._load_achievements()
        self.point_system = self._setup_point_system()
    
    def calculate_points(
        self,
        days_sober: int,
        entries_count: int,
        milestones_achieved: int,
        coaching_sessions: int
    ) -> Dict:
        """
        Calcula puntos totales del usuario
        
        Args:
            days_sober: Días de sobriedad
            entries_count: Número de entradas registradas
            milestones_achieved: Hitos alcanzados
            coaching_sessions: Sesiones de coaching completadas
        
        Returns:
            Diccionario con puntos y desglose
        """
        points = {
            "total": 0,
            "breakdown": {
                "sober_days": days_sober * self.point_system["sober_day"],
                "entries": entries_count * self.point_system["entry"],
                "milestones": milestones_achieved * self.point_system["milestone"],
                "coaching": coaching_sessions * self.point_system["coaching_session"]
            },
            "level": 1,
            "level_name": "Principiante"
        }
        
        points["total"] = sum(points["breakdown"].values())
        points["level"], points["level_name"] = self._calculate_level(points["total"])
        
        return points
    
    def check_achievements(
        self,
        user_id: str,
        days_sober: int,
        current_streak: int,
        entries_count: int
    ) -> List[Dict]:
        """
        Verifica qué logros se han desbloqueado
        
        Args:
            user_id: ID del usuario
            days_sober: Días de sobriedad
            current_streak: Racha actual
            entries_count: Número de entradas
        
        Returns:
            Lista de logros desbloqueados
        """
        unlocked = []
        
        for achievement in self.achievements:
            if self._is_achievement_unlocked(achievement, days_sober, current_streak, entries_count):
                unlocked.append({
                    **achievement,
                    "unlocked_at": datetime.now().isoformat(),
                    "user_id": user_id
                })
        
        return unlocked
    
    def get_leaderboard(self, users_data: List[Dict], limit: int = 10) -> List[Dict]:
        """
        Genera tabla de clasificación
        
        Args:
            users_data: Lista de datos de usuarios
            limit: Límite de usuarios en el ranking
        
        Returns:
            Lista ordenada de usuarios
        """
        # Calcular puntos para cada usuario
        ranked_users = []
        for user_data in users_data:
            points = self.calculate_points(
                user_data.get("days_sober", 0),
                user_data.get("entries_count", 0),
                user_data.get("milestones_achieved", 0),
                user_data.get("coaching_sessions", 0)
            )
            ranked_users.append({
                "user_id": user_data.get("user_id"),
                "points": points["total"],
                "level": points["level"],
                "level_name": points["level_name"],
                "days_sober": user_data.get("days_sober", 0)
            })
        
        # Ordenar por puntos
        ranked_users.sort(key=lambda x: x["points"], reverse=True)
        
        # Agregar posición
        for i, user in enumerate(ranked_users[:limit], 1):
            user["rank"] = i
        
        return ranked_users[:limit]
    
    def get_rewards(self, level: int) -> List[Dict]:
        """
        Obtiene recompensas disponibles para un nivel
        
        Args:
            level: Nivel del usuario
        
        Returns:
            Lista de recompensas
        """
        rewards = []
        
        if level >= 1:
            rewards.append({
                "type": "badge",
                "name": "Iniciador",
                "description": "Has comenzado tu viaje de recuperación",
                "icon": "🌟"
            })
        
        if level >= 5:
            rewards.append({
                "type": "badge",
                "name": "Comprometido",
                "description": "Has demostrado compromiso constante",
                "icon": "💪"
            })
        
        if level >= 10:
            rewards.append({
                "type": "badge",
                "name": "Experto",
                "description": "Eres un experto en tu recuperación",
                "icon": "🏆"
            })
        
        return rewards
    
    def _calculate_level(self, total_points: int) -> tuple:
        """Calcula nivel basado en puntos totales"""
        levels = [
            (0, 1, "Principiante"),
            (100, 2, "Aprendiz"),
            (500, 3, "Intermedio"),
            (1000, 4, "Avanzado"),
            (2500, 5, "Experto"),
            (5000, 6, "Maestro"),
            (10000, 7, "Leyenda")
        ]
        
        for points_threshold, level, level_name in reversed(levels):
            if total_points >= points_threshold:
                return level, level_name
        
        return 1, "Principiante"
    
    def _is_achievement_unlocked(
        self,
        achievement: Dict,
        days_sober: int,
        current_streak: int,
        entries_count: int
    ) -> bool:
        """Verifica si un logro está desbloqueado"""
        achievement_type = achievement.get("type")
        requirement = achievement.get("requirement", {})
        
        if achievement_type == AchievementType.MILESTONE:
            required_days = requirement.get("days", 0)
            return days_sober >= required_days
        
        elif achievement_type == AchievementType.STREAK:
            required_streak = requirement.get("days", 0)
            return current_streak >= required_streak
        
        elif achievement_type == AchievementType.CONSISTENCY:
            required_entries = requirement.get("entries", 0)
            return entries_count >= required_entries
        
        return False
    
    def _load_achievements(self) -> List[Dict]:
        """Carga logros disponibles"""
        return [
            {
                "id": "first_day",
                "type": AchievementType.MILESTONE,
                "name": "Primer Día",
                "description": "Completa tu primer día de sobriedad",
                "requirement": {"days": 1},
                "points": 10,
                "icon": "🌱"
            },
            {
                "id": "first_week",
                "type": AchievementType.MILESTONE,
                "name": "Primera Semana",
                "description": "Completa 7 días de sobriedad",
                "requirement": {"days": 7},
                "points": 50,
                "icon": "⭐"
            },
            {
                "id": "first_month",
                "type": AchievementType.MILESTONE,
                "name": "Primer Mes",
                "description": "Alcanza 30 días de sobriedad",
                "requirement": {"days": 30},
                "points": 200,
                "icon": "🎯"
            },
            {
                "id": "streak_7",
                "type": AchievementType.STREAK,
                "name": "Racha de 7",
                "description": "Mantén 7 días consecutivos sin consumo",
                "requirement": {"days": 7},
                "points": 75,
                "icon": "🔥"
            },
            {
                "id": "streak_30",
                "type": AchievementType.STREAK,
                "name": "Racha de 30",
                "description": "Mantén 30 días consecutivos sin consumo",
                "requirement": {"days": 30},
                "points": 300,
                "icon": "🔥🔥"
            },
            {
                "id": "consistent_tracker",
                "type": AchievementType.CONSISTENCY,
                "name": "Rastreador Consistente",
                "description": "Registra 30 entradas diarias",
                "requirement": {"entries": 30},
                "points": 100,
                "icon": "📊"
            }
        ]
    
    def _setup_point_system(self) -> Dict:
        """Configura sistema de puntos"""
        return {
            "sober_day": 10,
            "entry": 5,
            "milestone": 50,
            "coaching_session": 25,
            "support_contact": 15
        }

