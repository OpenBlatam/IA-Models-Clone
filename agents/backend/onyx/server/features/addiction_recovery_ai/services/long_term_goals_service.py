"""
Servicio de Seguimiento de Objetivos a Largo Plazo - Sistema completo de objetivos
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class GoalStatus(str, Enum):
    """Estados de objetivo"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class LongTermGoalsService:
    """Servicio de seguimiento de objetivos a largo plazo"""
    
    def __init__(self):
        """Inicializa el servicio de objetivos"""
        pass
    
    def create_long_term_goal(
        self,
        user_id: str,
        title: str,
        description: str,
        target_date: str,
        milestones: Optional[List[Dict]] = None,
        category: str = "recovery"
    ) -> Dict:
        """
        Crea un objetivo a largo plazo
        
        Args:
            user_id: ID del usuario
            title: Título del objetivo
            description: Descripción
            target_date: Fecha objetivo
            milestones: Hitos intermedios (opcional)
            category: Categoría del objetivo
        
        Returns:
            Objetivo creado
        """
        goal = {
            "id": f"goal_{datetime.now().timestamp()}",
            "user_id": user_id,
            "title": title,
            "description": description,
            "target_date": target_date,
            "milestones": milestones or [],
            "category": category,
            "status": GoalStatus.PLANNED,
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        return goal
    
    def update_goal_progress(
        self,
        goal_id: str,
        user_id: str,
        progress: float,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Actualiza progreso de objetivo
        
        Args:
            goal_id: ID del objetivo
            user_id: ID del usuario
            progress: Progreso (0-100)
            notes: Notas adicionales
        
        Returns:
            Progreso actualizado
        """
        return {
            "goal_id": goal_id,
            "user_id": user_id,
            "progress": min(100, max(0, progress)),
            "notes": notes,
            "updated_at": datetime.now().isoformat(),
            "status": self._determine_status(progress)
        }
    
    def get_goal_analytics(
        self,
        goal_id: str,
        user_id: str
    ) -> Dict:
        """
        Obtiene analíticas de objetivo
        
        Args:
            goal_id: ID del objetivo
            user_id: ID del usuario
        
        Returns:
            Analíticas del objetivo
        """
        return {
            "goal_id": goal_id,
            "user_id": user_id,
            "current_progress": 0.0,
            "projected_completion": None,
            "on_track": True,
            "days_remaining": 0,
            "milestones_completed": 0,
            "velocity": 0.0,
            "generated_at": datetime.now().isoformat()
        }
    
    def suggest_goals(
        self,
        user_id: str,
        recovery_stage: str,
        days_sober: int
    ) -> List[Dict]:
        """
        Sugiere objetivos basados en etapa de recuperación
        
        Args:
            user_id: ID del usuario
            recovery_stage: Etapa de recuperación
            days_sober: Días de sobriedad
        
        Returns:
            Lista de objetivos sugeridos
        """
        suggestions = []
        
        if days_sober < 30:
            suggestions.append({
                "title": "Completar 30 Días",
                "description": "Alcanza tu primer mes de sobriedad",
                "target_days": 30,
                "priority": "high"
            })
        elif days_sober < 90:
            suggestions.append({
                "title": "Completar 90 Días",
                "description": "Alcanza los 3 meses de sobriedad",
                "target_days": 90,
                "priority": "high"
            })
        
        return suggestions
    
    def _determine_status(self, progress: float) -> str:
        """Determina estado basado en progreso"""
        if progress >= 100:
            return GoalStatus.COMPLETED
        elif progress >= 75:
            return GoalStatus.ON_TRACK
        elif progress >= 50:
            return GoalStatus.IN_PROGRESS
        else:
            return GoalStatus.AT_RISK

