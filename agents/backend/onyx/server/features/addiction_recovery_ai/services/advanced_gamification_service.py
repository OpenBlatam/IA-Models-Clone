"""
Servicio de Gamificación Avanzada - Sistema completo de gamificación
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class AchievementRarity(str, Enum):
    """Rareza de logros"""
    COMMON = "common"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"


class AdvancedGamificationService:
    """Servicio de gamificación avanzada"""
    
    def __init__(self):
        """Inicializa el servicio de gamificación"""
        self.achievements = self._load_achievements()
    
    def award_achievement(
        self,
        user_id: str,
        achievement_id: str,
        progress_data: Optional[Dict] = None
    ) -> Dict:
        """
        Otorga un logro
        
        Args:
            user_id: ID del usuario
            achievement_id: ID del logro
            progress_data: Datos de progreso
        
        Returns:
            Logro otorgado
        """
        achievement = next((a for a in self.achievements if a.get("id") == achievement_id), None)
        
        if not achievement:
            return {
                "error": "Achievement not found"
            }
        
        awarded = {
            "user_id": user_id,
            "achievement_id": achievement_id,
            "achievement_name": achievement.get("name"),
            "description": achievement.get("description"),
            "rarity": achievement.get("rarity"),
            "points": achievement.get("points", 0),
            "unlocked_at": datetime.now().isoformat(),
            "progress_data": progress_data
        }
        
        return awarded
    
    def get_user_achievements(
        self,
        user_id: str,
        rarity: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene logros del usuario
        
        Args:
            user_id: ID del usuario
            rarity: Filtrar por rareza (opcional)
        
        Returns:
            Lista de logros
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def calculate_user_level(
        self,
        user_id: str,
        total_points: int
    ) -> Dict:
        """
        Calcula nivel del usuario
        
        Args:
            user_id: ID del usuario
            total_points: Puntos totales
        
        Returns:
            Información de nivel
        """
        # Sistema de niveles basado en puntos
        level = min(100, (total_points // 100) + 1)
        points_for_next = (level * 100) - total_points
        
        return {
            "user_id": user_id,
            "level": level,
            "total_points": total_points,
            "points_for_next_level": points_for_next,
            "progress_to_next": ((total_points % 100) / 100) * 100,
            "title": self._get_level_title(level)
        }
    
    def get_leaderboard(
        self,
        category: str = "overall",
        limit: int = 100
    ) -> List[Dict]:
        """
        Obtiene tabla de clasificación
        
        Args:
            category: Categoría (overall, weekly, monthly)
            limit: Límite de resultados
        
        Returns:
            Tabla de clasificación
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def create_custom_badge(
        self,
        user_id: str,
        badge_name: str,
        badge_description: str,
        icon: Optional[str] = None
    ) -> Dict:
        """
        Crea badge personalizado
        
        Args:
            user_id: ID del usuario
            badge_name: Nombre del badge
            badge_description: Descripción
            icon: Icono (opcional)
        
        Returns:
            Badge creado
        """
        badge = {
            "id": f"badge_{datetime.now().timestamp()}",
            "user_id": user_id,
            "name": badge_name,
            "description": badge_description,
            "icon": icon or "🏆",
            "created_at": datetime.now().isoformat(),
            "custom": True
        }
        
        return badge
    
    def _load_achievements(self) -> List[Dict]:
        """Carga logros disponibles"""
        return [
            {
                "id": "first_day",
                "name": "Primer Día",
                "description": "Completa tu primer día de sobriedad",
                "rarity": AchievementRarity.COMMON,
                "points": 10
            },
            {
                "id": "week_warrior",
                "name": "Guerrero de la Semana",
                "description": "Completa 7 días consecutivos",
                "rarity": AchievementRarity.RARE,
                "points": 50
            },
            {
                "id": "month_master",
                "name": "Maestro del Mes",
                "description": "Completa 30 días consecutivos",
                "rarity": AchievementRarity.EPIC,
                "points": 200
            },
            {
                "id": "century_club",
                "name": "Club del Siglo",
                "description": "Completa 100 días consecutivos",
                "rarity": AchievementRarity.LEGENDARY,
                "points": 1000
            }
        ]
    
    def _get_level_title(self, level: int) -> str:
        """Obtiene título de nivel"""
        if level < 10:
            return "Principiante"
        elif level < 25:
            return "Aprendiz"
        elif level < 50:
            return "Experto"
        elif level < 75:
            return "Maestro"
        else:
            return "Leyenda"

