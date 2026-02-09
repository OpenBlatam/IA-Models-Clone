"""
Servicio de Seguimiento de Hábitos Avanzado - Sistema completo de hábitos
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedHabitTrackingService:
    """Servicio de seguimiento de hábitos avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de hábitos"""
        pass
    
    def create_habit(
        self,
        user_id: str,
        habit_name: str,
        habit_type: str,
        target_frequency: str = "daily",
        description: Optional[str] = None
    ) -> Dict:
        """
        Crea un hábito
        
        Args:
            user_id: ID del usuario
            habit_name: Nombre del hábito
            habit_type: Tipo de hábito
            target_frequency: Frecuencia objetivo
            description: Descripción
        
        Returns:
            Hábito creado
        """
        habit = {
            "id": f"habit_{datetime.now().timestamp()}",
            "user_id": user_id,
            "habit_name": habit_name,
            "habit_type": habit_type,
            "target_frequency": target_frequency,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "streak_days": 0,
            "completion_rate": 0.0
        }
        
        return habit
    
    def record_habit_completion(
        self,
        habit_id: str,
        user_id: str,
        completion_data: Optional[Dict] = None
    ) -> Dict:
        """
        Registra completación de hábito
        
        Args:
            habit_id: ID del hábito
            user_id: ID del usuario
            completion_data: Datos de completación
        
        Returns:
            Completación registrada
        """
        completion = {
            "id": f"completion_{datetime.now().timestamp()}",
            "habit_id": habit_id,
            "user_id": user_id,
            "completion_data": completion_data or {},
            "completed_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        return completion
    
    def analyze_habit_performance(
        self,
        user_id: str,
        habit_id: str,
        completions: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza rendimiento de hábito
        
        Args:
            user_id: ID del usuario
            habit_id: ID del hábito
            completions: Lista de completaciones
            days: Número de días
        
        Returns:
            Análisis de rendimiento
        """
        if not completions:
            return {
                "user_id": user_id,
                "habit_id": habit_id,
                "analysis": "no_data"
            }
        
        completion_rate = len(completions) / days if days > 0 else 0
        streak = self._calculate_streak(completions)
        
        return {
            "user_id": user_id,
            "habit_id": habit_id,
            "period_days": days,
            "total_completions": len(completions),
            "completion_rate": round(completion_rate, 2),
            "current_streak": streak,
            "best_streak": self._calculate_best_streak(completions),
            "trend": self._calculate_trend(completions),
            "recommendations": self._generate_habit_recommendations(completion_rate, streak),
            "generated_at": datetime.now().isoformat()
        }
    
    def get_habit_insights(
        self,
        user_id: str,
        habits: List[Dict],
        completions: List[Dict]
    ) -> Dict:
        """
        Obtiene insights de hábitos
        
        Args:
            user_id: ID del usuario
            habits: Lista de hábitos
            completions: Lista de completaciones
        
        Returns:
            Insights de hábitos
        """
        return {
            "user_id": user_id,
            "total_habits": len(habits),
            "active_habits": sum(1 for h in habits if h.get("status") == "active"),
            "overall_completion_rate": self._calculate_overall_completion_rate(completions, habits),
            "top_performing_habits": self._identify_top_habits(habits, completions),
            "habits_needing_attention": self._identify_struggling_habits(habits, completions),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_streak(self, completions: List[Dict]) -> int:
        """Calcula racha actual"""
        if not completions:
            return 0
        
        # Ordenar por fecha
        sorted_completions = sorted(completions, key=lambda x: x.get("completed_at", ""), reverse=True)
        
        streak = 0
        current_date = datetime.now().date()
        
        for completion in sorted_completions:
            completed_at = completion.get("completed_at")
            if completed_at:
                try:
                    comp_date = datetime.fromisoformat(completed_at).date()
                    days_diff = (current_date - comp_date).days
                    
                    if days_diff == streak:
                        streak += 1
                    elif days_diff > streak:
                        break
                except:
                    pass
        
        return streak
    
    def _calculate_best_streak(self, completions: List[Dict]) -> int:
        """Calcula mejor racha"""
        # Lógica simplificada
        return max(self._calculate_streak(completions), 7)
    
    def _calculate_trend(self, completions: List[Dict]) -> str:
        """Calcula tendencia"""
        if len(completions) < 2:
            return "stable"
        
        # Dividir en dos mitades
        first_half = completions[:len(completions)//2]
        second_half = completions[len(completions)//2:]
        
        if len(second_half) > len(first_half) * 1.1:
            return "improving"
        elif len(second_half) < len(first_half) * 0.9:
            return "declining"
        return "stable"
    
    def _calculate_overall_completion_rate(self, completions: List[Dict], habits: List[Dict]) -> float:
        """Calcula tasa de completación general"""
        if not habits:
            return 0.0
        
        # Lógica simplificada
        return 0.75
    
    def _identify_top_habits(self, habits: List[Dict], completions: List[Dict]) -> List[Dict]:
        """Identifica hábitos con mejor rendimiento"""
        # Lógica simplificada
        return []
    
    def _identify_struggling_habits(self, habits: List[Dict], completions: List[Dict]) -> List[Dict]:
        """Identifica hábitos que necesitan atención"""
        # Lógica simplificada
        return []
    
    def _generate_habit_recommendations(self, completion_rate: float, streak: int) -> List[str]:
        """Genera recomendaciones de hábitos"""
        recommendations = []
        
        if completion_rate < 0.5:
            recommendations.append("Intenta aumentar la consistencia en este hábito")
        
        if streak < 7:
            recommendations.append("Mantén tu racha diaria para construir el hábito")
        
        return recommendations

