"""
Gamification Service - Sistema de gamificación
===============================================

Sistema completo de gamificación con puntos, niveles, badges, logros y leaderboards.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BadgeType(str, Enum):
    """Tipos de badges disponibles"""
    FIRST_STEP = "first_step"
    PROFILE_COMPLETE = "profile_complete"
    FIRST_APPLICATION = "first_application"
    STREAK_7_DAYS = "streak_7_days"
    STREAK_30_DAYS = "streak_30_days"
    SKILL_LEARNED = "skill_learned"
    NETWORKING_MASTER = "networking_master"
    INTERVIEW_READY = "interview_ready"
    JOB_OFFER = "job_offer"
    MENTOR = "mentor"
    COMMUNITY_HELPER = "community_helper"


class AchievementType(str, Enum):
    """Tipos de logros"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"
    LEGEND = "legend"


@dataclass
class Badge:
    """Representa un badge"""
    id: str
    name: str
    description: str
    icon: str
    rarity: str  # common, rare, epic, legendary
    points: int
    unlocked_at: Optional[datetime] = None


@dataclass
class Achievement:
    """Representa un logro"""
    id: str
    name: str
    description: str
    icon: str
    points: int
    progress: float = 0.0  # 0.0 a 1.0
    unlocked_at: Optional[datetime] = None


@dataclass
class UserStats:
    """Estadísticas del usuario"""
    user_id: str
    total_points: int = 0
    current_level: int = 1
    experience_points: int = 0
    badges: List[Badge] = field(default_factory=list)
    achievements: List[Achievement] = field(default_factory=list)
    current_streak: int = 0
    longest_streak: int = 0
    last_activity: Optional[datetime] = None
    jobs_applied: int = 0
    jobs_saved: int = 0
    skills_learned: int = 0
    networking_contacts: int = 0


class GamificationService:
    """Servicio de gamificación"""
    
    # Experiencia necesaria por nivel
    XP_PER_LEVEL = {
        1: 100,
        2: 250,
        3: 500,
        4: 1000,
        5: 2000,
        6: 3500,
        7: 5500,
        8: 8000,
        9: 12000,
        10: 20000,
    }
    
    # Puntos por acción
    POINTS_ACTIONS = {
        "complete_profile": 50,
        "complete_step": 25,
        "apply_job": 100,
        "save_job": 10,
        "learn_skill": 75,
        "network_contact": 30,
        "complete_challenge": 150,
        "help_community": 50,
        "daily_login": 20,
    }
    
    def __init__(self):
        """Inicializar servicio de gamificación"""
        self.user_stats: Dict[str, UserStats] = {}
        logger.info("GamificationService initialized")
    
    def get_user_stats(self, user_id: str) -> UserStats:
        """Obtener estadísticas del usuario"""
        if user_id not in self.user_stats:
            self.user_stats[user_id] = UserStats(user_id=user_id)
        return self.user_stats[user_id]
    
    def add_points(self, user_id: str, action: str, amount: Optional[int] = None) -> Dict[str, Any]:
        """Agregar puntos al usuario por una acción"""
        stats = self.get_user_stats(user_id)
        
        points = amount or self.POINTS_ACTIONS.get(action, 10)
        stats.total_points += points
        stats.experience_points += points
        
        # Verificar si subió de nivel
        level_up = False
        new_level = self._calculate_level(stats.experience_points)
        if new_level > stats.current_level:
            level_up = True
            stats.current_level = new_level
            logger.info(f"User {user_id} leveled up to {new_level}!")
        
        # Actualizar última actividad
        stats.last_activity = datetime.now()
        
        # Verificar streak
        self._update_streak(stats)
        
        # Verificar badges y achievements
        self._check_badges_and_achievements(stats, action)
        
        return {
            "points_added": points,
            "total_points": stats.total_points,
            "current_level": stats.current_level,
            "experience_points": stats.experience_points,
            "level_up": level_up,
            "next_level_xp": self._get_xp_for_next_level(stats.current_level),
        }
    
    def _calculate_level(self, xp: int) -> int:
        """Calcular nivel basado en experiencia"""
        level = 1
        total_xp = 0
        
        for lvl, required_xp in sorted(self.XP_PER_LEVEL.items()):
            if xp >= total_xp + required_xp:
                total_xp += required_xp
                level = lvl + 1
            else:
                break
        
        return min(level, 10)  # Máximo nivel 10
    
    def _get_xp_for_next_level(self, current_level: int) -> int:
        """Obtener XP necesaria para el siguiente nivel"""
        if current_level >= 10:
            return 0
        return self.XP_PER_LEVEL.get(current_level + 1, 0)
    
    def _update_streak(self, stats: UserStats):
        """Actualizar racha de días consecutivos"""
        now = datetime.now()
        
        if stats.last_activity is None:
            stats.current_streak = 1
            return
        
        days_diff = (now.date() - stats.last_activity.date()).days
        
        if days_diff == 0:
            # Ya se registró actividad hoy
            return
        elif days_diff == 1:
            # Día consecutivo
            stats.current_streak += 1
        else:
            # Racha rota
            stats.current_streak = 1
        
        if stats.current_streak > stats.longest_streak:
            stats.longest_streak = stats.current_streak
    
    def _check_badges_and_achievements(self, stats: UserStats, action: str):
        """Verificar y otorgar badges y achievements"""
        # Badge por primera acción
        if action == "complete_step" and not any(b.id == BadgeType.FIRST_STEP for b in stats.badges):
            badge = Badge(
                id=BadgeType.FIRST_STEP,
                name="Primer Paso",
                description="Completaste tu primer paso en el camino",
                icon="🎯",
                rarity="common",
                points=25,
                unlocked_at=datetime.now()
            )
            stats.badges.append(badge)
            stats.total_points += badge.points
        
        # Badge por perfil completo
        if action == "complete_profile" and not any(b.id == BadgeType.PROFILE_COMPLETE for b in stats.badges):
            badge = Badge(
                id=BadgeType.PROFILE_COMPLETE,
                name="Perfil Completo",
                description="Completaste tu perfil al 100%",
                icon="✅",
                rarity="common",
                points=50,
                unlocked_at=datetime.now()
            )
            stats.badges.append(badge)
        
        # Badge por primera aplicación
        if action == "apply_job" and stats.jobs_applied == 1:
            badge = Badge(
                id=BadgeType.FIRST_APPLICATION,
                name="Primera Aplicación",
                description="Enviaste tu primera aplicación",
                icon="📝",
                rarity="common",
                points=100,
                unlocked_at=datetime.now()
            )
            stats.badges.append(badge)
        
        # Badge por racha de 7 días
        if stats.current_streak == 7 and not any(b.id == BadgeType.STREAK_7_DAYS for b in stats.badges):
            badge = Badge(
                id=BadgeType.STREAK_7_DAYS,
                name="Semana de Dedicación",
                description="7 días consecutivos activo",
                icon="🔥",
                rarity="rare",
                points=200,
                unlocked_at=datetime.now()
            )
            stats.badges.append(badge)
        
        # Badge por racha de 30 días
        if stats.current_streak == 30 and not any(b.id == BadgeType.STREAK_30_DAYS for b in stats.badges):
            badge = Badge(
                id=BadgeType.STREAK_30_DAYS,
                name="Mes de Excelencia",
                description="30 días consecutivos activo",
                icon="⭐",
                rarity="epic",
                points=1000,
                unlocked_at=datetime.now()
            )
            stats.badges.append(badge)
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener leaderboard de usuarios"""
        sorted_users = sorted(
            self.user_stats.values(),
            key=lambda x: x.total_points,
            reverse=True
        )
        
        return [
            {
                "user_id": stats.user_id,
                "total_points": stats.total_points,
                "level": stats.current_level,
                "badges_count": len(stats.badges),
                "streak": stats.current_streak,
            }
            for stats in sorted_users[:limit]
        ]
    
    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Obtener progreso completo del usuario"""
        stats = self.get_user_stats(user_id)
        
        return {
            "user_id": user_id,
            "level": stats.current_level,
            "total_points": stats.total_points,
            "experience_points": stats.experience_points,
            "next_level_xp": self._get_xp_for_next_level(stats.current_level),
            "xp_progress": self._calculate_xp_progress(stats),
            "current_streak": stats.current_streak,
            "longest_streak": stats.longest_streak,
            "badges": [
                {
                    "id": badge.id,
                    "name": badge.name,
                    "description": badge.description,
                    "icon": badge.icon,
                    "rarity": badge.rarity,
                    "unlocked_at": badge.unlocked_at.isoformat() if badge.unlocked_at else None,
                }
                for badge in stats.badges
            ],
            "achievements": [
                {
                    "id": achievement.id,
                    "name": achievement.name,
                    "description": achievement.description,
                    "progress": achievement.progress,
                    "unlocked": achievement.unlocked_at is not None,
                }
                for achievement in stats.achievements
            ],
            "stats": {
                "jobs_applied": stats.jobs_applied,
                "jobs_saved": stats.jobs_saved,
                "skills_learned": stats.skills_learned,
                "networking_contacts": stats.networking_contacts,
            }
        }
    
    def _calculate_xp_progress(self, stats: UserStats) -> float:
        """Calcular progreso hacia el siguiente nivel (0.0 a 1.0)"""
        if stats.current_level >= 10:
            return 1.0
        
        current_level_xp = sum(
            self.XP_PER_LEVEL.get(i, 0)
            for i in range(1, stats.current_level)
        )
        
        next_level_xp = self._get_xp_for_next_level(stats.current_level)
        current_xp_in_level = stats.experience_points - current_level_xp
        
        if next_level_xp == 0:
            return 1.0
        
        return min(current_xp_in_level / next_level_xp, 1.0)




