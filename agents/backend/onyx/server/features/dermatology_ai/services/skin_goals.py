"""
Sistema de objetivos de piel
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid


class GoalStatus(str, Enum):
    """Estado del objetivo"""
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class SkinGoal:
    """Objetivo de piel"""
    id: str
    user_id: str
    title: str
    description: str
    target_metric: str  # "hydration", "texture", "overall_score"
    target_value: float
    current_value: float
    deadline: Optional[str] = None
    status: GoalStatus = GoalStatus.ACTIVE
    progress_percentage: float = 0.0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self._update_progress()
    
    def _update_progress(self):
        """Actualiza progreso"""
        if self.target_value > 0:
            self.progress_percentage = min(100.0, (self.current_value / self.target_value) * 100)
        else:
            self.progress_percentage = 0.0
    
    def update_current_value(self, value: float):
        """Actualiza valor actual"""
        self.current_value = value
        self._update_progress()
        
        # Verificar si se completó
        if self.progress_percentage >= 100.0 and self.status == GoalStatus.ACTIVE:
            self.status = GoalStatus.COMPLETED
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "target_metric": self.target_metric,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "deadline": self.deadline,
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "created_at": self.created_at
        }


class SkinGoalsManager:
    """Gestor de objetivos de piel"""
    
    def __init__(self):
        """Inicializa el gestor"""
        self.goals: Dict[str, List[SkinGoal]] = {}  # user_id -> [goals]
    
    def create_goal(self, user_id: str, title: str, description: str,
                    target_metric: str, target_value: float,
                    current_value: float = 0.0, deadline: Optional[str] = None) -> SkinGoal:
        """Crea un nuevo objetivo"""
        goal = SkinGoal(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            description=description,
            target_metric=target_metric,
            target_value=target_value,
            current_value=current_value,
            deadline=deadline
        )
        
        if user_id not in self.goals:
            self.goals[user_id] = []
        
        self.goals[user_id].append(goal)
        return goal
    
    def update_goal_progress(self, user_id: str, goal_id: str,
                            current_value: float) -> bool:
        """Actualiza progreso de un objetivo"""
        user_goals = self.goals.get(user_id, [])
        
        for goal in user_goals:
            if goal.id == goal_id:
                goal.update_current_value(current_value)
                return True
        
        return False
    
    def get_user_goals(self, user_id: str, status: Optional[GoalStatus] = None) -> List[SkinGoal]:
        """Obtiene objetivos del usuario"""
        user_goals = self.goals.get(user_id, [])
        
        if status:
            user_goals = [g for g in user_goals if g.status == status]
        
        # Ordenar por fecha de creación
        user_goals.sort(key=lambda x: x.created_at, reverse=True)
        
        return user_goals
    
    def complete_goal(self, user_id: str, goal_id: str) -> bool:
        """Marca objetivo como completado"""
        user_goals = self.goals.get(user_id, [])
        
        for goal in user_goals:
            if goal.id == goal_id:
                goal.status = GoalStatus.COMPLETED
                goal.progress_percentage = 100.0
                return True
        
        return False
    
    def pause_goal(self, user_id: str, goal_id: str) -> bool:
        """Pausa un objetivo"""
        user_goals = self.goals.get(user_id, [])
        
        for goal in user_goals:
            if goal.id == goal_id:
                goal.status = GoalStatus.PAUSED
                return True
        
        return False
    
    def get_goals_statistics(self, user_id: str) -> Dict:
        """Obtiene estadísticas de objetivos"""
        user_goals = self.goals.get(user_id, [])
        
        total = len(user_goals)
        active = len([g for g in user_goals if g.status == GoalStatus.ACTIVE])
        completed = len([g for g in user_goals if g.status == GoalStatus.COMPLETED])
        paused = len([g for g in user_goals if g.status == GoalStatus.PAUSED])
        
        avg_progress = 0.0
        if user_goals:
            avg_progress = sum(g.progress_percentage for g in user_goals) / len(user_goals)
        
        return {
            "total_goals": total,
            "active_goals": active,
            "completed_goals": completed,
            "paused_goals": paused,
            "average_progress": avg_progress,
            "completion_rate": (completed / total * 100) if total > 0 else 0.0
        }






