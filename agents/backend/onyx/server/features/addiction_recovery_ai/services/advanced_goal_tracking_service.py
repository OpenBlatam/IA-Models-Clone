"""
Servicio de Seguimiento de Objetivos Avanzado - Sistema completo de objetivos
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedGoalTrackingService:
    """Servicio de seguimiento de objetivos avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de objetivos"""
        pass
    
    def create_advanced_goal(
        self,
        user_id: str,
        goal_data: Dict
    ) -> Dict:
        """
        Crea objetivo avanzado
        
        Args:
            user_id: ID del usuario
            goal_data: Datos del objetivo
        
        Returns:
            Objetivo creado
        """
        goal = {
            "id": f"goal_{datetime.now().timestamp()}",
            "user_id": user_id,
            "goal_data": goal_data,
            "goal_type": goal_data.get("goal_type", "recovery"),
            "title": goal_data.get("title", ""),
            "description": goal_data.get("description", ""),
            "target_date": goal_data.get("target_date"),
            "milestones": goal_data.get("milestones", []),
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "progress": 0.0
        }
        
        return goal
    
    def track_goal_progress(
        self,
        goal_id: str,
        user_id: str,
        progress_update: Dict
    ) -> Dict:
        """
        Rastrea progreso de objetivo
        
        Args:
            goal_id: ID del objetivo
            user_id: ID del usuario
            progress_update: Actualización de progreso
        
        Returns:
            Progreso actualizado
        """
        return {
            "goal_id": goal_id,
            "user_id": user_id,
            "progress_update": progress_update,
            "new_progress": progress_update.get("progress", 0),
            "updated_at": datetime.now().isoformat(),
            "milestones_achieved": self._check_milestones(goal_id, progress_update)
        }
    
    def analyze_goal_performance(
        self,
        user_id: str,
        goals: List[Dict]
    ) -> Dict:
        """
        Analiza rendimiento de objetivos
        
        Args:
            user_id: ID del usuario
            goals: Lista de objetivos
        
        Returns:
            Análisis de rendimiento
        """
        if not goals:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "total_goals": len(goals),
            "active_goals": sum(1 for g in goals if g.get("status") == "active"),
            "completed_goals": sum(1 for g in goals if g.get("status") == "completed"),
            "average_progress": self._calculate_average_progress(goals),
            "goal_completion_rate": self._calculate_completion_rate(goals),
            "top_performing_goals": self._identify_top_goals(goals),
            "goals_needing_attention": self._identify_struggling_goals(goals),
            "recommendations": self._generate_goal_recommendations(goals),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_goal_achievement(
        self,
        goal_id: str,
        user_id: str,
        current_progress: Dict
    ) -> Dict:
        """
        Predice logro de objetivo
        
        Args:
            goal_id: ID del objetivo
            user_id: ID del usuario
            current_progress: Progreso actual
        
        Returns:
            Predicción de logro
        """
        achievement_probability = self._calculate_achievement_probability(current_progress)
        
        return {
            "goal_id": goal_id,
            "user_id": user_id,
            "achievement_probability": round(achievement_probability, 3),
            "estimated_completion_date": self._estimate_completion_date(current_progress),
            "confidence": 0.75,
            "recommendations": self._generate_achievement_recommendations(achievement_probability),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _check_milestones(self, goal_id: str, progress: Dict) -> List[str]:
        """Verifica hitos alcanzados"""
        milestones = []
        
        current_progress = progress.get("progress", 0)
        
        if current_progress >= 0.25:
            milestones.append("25% completado")
        if current_progress >= 0.5:
            milestones.append("50% completado")
        if current_progress >= 0.75:
            milestones.append("75% completado")
        
        return milestones
    
    def _calculate_average_progress(self, goals: List[Dict]) -> float:
        """Calcula progreso promedio"""
        if not goals:
            return 0.0
        
        progresses = [g.get("progress", 0) for g in goals]
        return round(statistics.mean(progresses), 2)
    
    def _calculate_completion_rate(self, goals: List[Dict]) -> float:
        """Calcula tasa de completación"""
        if not goals:
            return 0.0
        
        completed = sum(1 for g in goals if g.get("status") == "completed")
        return round((completed / len(goals) * 100), 2)
    
    def _identify_top_goals(self, goals: List[Dict]) -> List[Dict]:
        """Identifica objetivos con mejor rendimiento"""
        sorted_goals = sorted(goals, key=lambda x: x.get("progress", 0), reverse=True)
        return sorted_goals[:3]
    
    def _identify_struggling_goals(self, goals: List[Dict]) -> List[Dict]:
        """Identifica objetivos que necesitan atención"""
        struggling = [g for g in goals if g.get("progress", 0) < 0.3 and g.get("status") == "active"]
        return struggling
    
    def _generate_goal_recommendations(self, goals: List[Dict]) -> List[str]:
        """Genera recomendaciones de objetivos"""
        recommendations = []
        
        struggling = self._identify_struggling_goals(goals)
        if struggling:
            recommendations.append("Algunos objetivos necesitan más atención. Considera ajustar estrategias")
        
        return recommendations
    
    def _calculate_achievement_probability(self, progress: Dict) -> float:
        """Calcula probabilidad de logro"""
        current_progress = progress.get("progress", 0)
        days_remaining = progress.get("days_remaining", 30)
        
        if current_progress >= 0.8:
            return 0.9
        elif current_progress >= 0.5:
            return 0.7
        elif current_progress >= 0.3:
            return 0.5
        else:
            return 0.3
    
    def _estimate_completion_date(self, progress: Dict) -> str:
        """Estima fecha de completación"""
        current_progress = progress.get("progress", 0)
        days_remaining = progress.get("days_remaining", 30)
        
        if current_progress > 0:
            estimated_days = days_remaining * (1 - current_progress) / current_progress
        else:
            estimated_days = days_remaining
        
        estimated_date = datetime.now() + timedelta(days=int(estimated_days))
        return estimated_date.isoformat()
    
    def _generate_achievement_recommendations(self, probability: float) -> List[str]:
        """Genera recomendaciones de logro"""
        recommendations = []
        
        if probability < 0.6:
            recommendations.append("Aumenta el esfuerzo para alcanzar este objetivo")
            recommendations.append("Considera ajustar el objetivo si es necesario")
        
        return recommendations

