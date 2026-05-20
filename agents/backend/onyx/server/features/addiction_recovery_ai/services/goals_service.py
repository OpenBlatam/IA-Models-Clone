"""
Servicio de Metas y Objetivos - Gestión avanzada de metas
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class GoalStatus(str, Enum):
    """Estados de metas"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GoalType(str, Enum):
    """Tipos de metas"""
    SOBRIETY = "sobriety"
    HEALTH = "health"
    PERSONAL = "personal"
    SOCIAL = "social"
    FINANCIAL = "financial"


class GoalsService:
    """Servicio de gestión de metas y objetivos"""
    
    def __init__(self):
        """Inicializa el servicio de metas"""
        pass
    
    def create_goal(
        self,
        user_id: str,
        goal_type: str,
        title: str,
        description: str,
        target_date: str,
        target_value: Optional[float] = None,
        milestones: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Crea una nueva meta
        
        Args:
            user_id: ID del usuario
            goal_type: Tipo de meta
            title: Título
            description: Descripción
            target_date: Fecha objetivo
            target_value: Valor objetivo (opcional)
            milestones: Hitos intermedios (opcional)
        
        Returns:
            Meta creada
        """
        goal = {
            "id": f"goal_{datetime.now().timestamp()}",
            "user_id": user_id,
            "goal_type": goal_type,
            "title": title,
            "description": description,
            "target_date": target_date,
            "target_value": target_value,
            "current_value": 0,
            "progress_percentage": 0,
            "status": GoalStatus.PENDING,
            "milestones": milestones or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        return goal
    
    def update_goal_progress(
        self,
        goal_id: str,
        current_value: float,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Actualiza progreso de una meta
        
        Args:
            goal_id: ID de la meta
            current_value: Valor actual
            notes: Notas adicionales
        
        Returns:
            Meta actualizada
        """
        # En implementación real, esto actualizaría la base de datos
        return {
            "goal_id": goal_id,
            "current_value": current_value,
            "updated_at": datetime.now().isoformat(),
            "notes": notes
        }
    
    def get_user_goals(
        self,
        user_id: str,
        status: Optional[str] = None,
        goal_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene metas del usuario
        
        Args:
            user_id: ID del usuario
            status: Filtrar por estado (opcional)
            goal_type: Filtrar por tipo (opcional)
        
        Returns:
            Lista de metas
        """
        # En implementación real, esto vendría de la base de datos
        goals = []
        
        # Ejemplos de metas
        example_goals = [
            {
                "id": "goal_1",
                "user_id": user_id,
                "goal_type": GoalType.SOBRIETY,
                "title": "30 días de sobriedad",
                "description": "Alcanzar 30 días consecutivos sin consumo",
                "target_date": (datetime.now() + timedelta(days=30)).isoformat(),
                "target_value": 30,
                "current_value": 0,
                "progress_percentage": 0,
                "status": GoalStatus.IN_PROGRESS
            },
            {
                "id": "goal_2",
                "user_id": user_id,
                "goal_type": GoalType.HEALTH,
                "title": "Ejercicio regular",
                "description": "Hacer ejercicio 3 veces por semana",
                "target_date": (datetime.now() + timedelta(days=90)).isoformat(),
                "status": GoalStatus.PENDING
            }
        ]
        
        if status:
            goals = [g for g in example_goals if g.get("status") == status]
        else:
            goals = example_goals
        
        if goal_type:
            goals = [g for g in goals if g.get("goal_type") == goal_type]
        
        return goals
    
    def complete_goal(self, goal_id: str, completion_notes: Optional[str] = None) -> Dict:
        """
        Marca una meta como completada
        
        Args:
            goal_id: ID de la meta
            completion_notes: Notas de finalización
        
        Returns:
            Meta completada
        """
        return {
            "goal_id": goal_id,
            "status": GoalStatus.COMPLETED,
            "completed_at": datetime.now().isoformat(),
            "completion_notes": completion_notes
        }
    
    def get_goal_suggestions(
        self,
        user_id: str,
        days_sober: int,
        addiction_type: str
    ) -> List[Dict]:
        """
        Obtiene sugerencias de metas basadas en progreso
        
        Args:
            user_id: ID del usuario
            days_sober: Días de sobriedad
            addiction_type: Tipo de adicción
        
        Returns:
            Lista de sugerencias de metas
        """
        suggestions = []
        
        # Metas de sobriedad
        if days_sober < 7:
            suggestions.append({
                "goal_type": GoalType.SOBRIETY,
                "title": "Primera semana",
                "description": "Completar 7 días de sobriedad",
                "target_days": 7,
                "priority": "high"
            })
        elif days_sober < 30:
            suggestions.append({
                "goal_type": GoalType.SOBRIETY,
                "title": "Primer mes",
                "description": "Alcanzar 30 días de sobriedad",
                "target_days": 30,
                "priority": "high"
            })
        
        # Metas de salud
        suggestions.append({
            "goal_type": GoalType.HEALTH,
            "title": "Ejercicio regular",
            "description": "Hacer ejercicio al menos 3 veces por semana",
            "priority": "medium"
        })
        
        # Metas financieras (si aplica)
        suggestions.append({
            "goal_type": GoalType.FINANCIAL,
            "title": "Ahorro mensual",
            "description": "Ahorrar el dinero que antes gastabas en la adicción",
            "priority": "medium"
        })
        
        # Metas sociales
        suggestions.append({
            "goal_type": GoalType.SOCIAL,
            "title": "Construir red de apoyo",
            "description": "Asistir a grupos de apoyo o construir relaciones saludables",
            "priority": "high"
        })
        
        return suggestions
    
    def calculate_goal_progress(self, goal: Dict) -> float:
        """
        Calcula porcentaje de progreso de una meta
        
        Args:
            goal: Meta
        
        Returns:
            Porcentaje de progreso (0-100)
        """
        if not goal.get("target_value"):
            return 0.0
        
        current = goal.get("current_value", 0)
        target = goal.get("target_value", 1)
        
        progress = (current / target) * 100 if target > 0 else 0
        return min(100, max(0, progress))
    
    def get_goals_statistics(self, user_id: str, goals: List[Dict]) -> Dict:
        """
        Obtiene estadísticas de metas
        
        Args:
            user_id: ID del usuario
            goals: Lista de metas
        
        Returns:
            Estadísticas
        """
        total = len(goals)
        completed = sum(1 for g in goals if g.get("status") == GoalStatus.COMPLETED)
        in_progress = sum(1 for g in goals if g.get("status") == GoalStatus.IN_PROGRESS)
        pending = sum(1 for g in goals if g.get("status") == GoalStatus.PENDING)
        
        avg_progress = 0
        if goals:
            progress_values = [self.calculate_goal_progress(g) for g in goals]
            avg_progress = sum(progress_values) / len(progress_values) if progress_values else 0
        
        return {
            "user_id": user_id,
            "total_goals": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "completion_rate": (completed / total * 100) if total > 0 else 0,
            "average_progress": round(avg_progress, 2)
        }

