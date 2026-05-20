"""
Servicio de Análisis de Hábitos Avanzado - Sistema completo de análisis de hábitos
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class AdvancedHabitAnalysisService:
    """Servicio de análisis de hábitos avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de hábitos"""
        pass
    
    def analyze_habits(
        self,
        user_id: str,
        habits: List[Dict]
    ) -> Dict:
        """
        Analiza hábitos
        
        Args:
            user_id: ID del usuario
            habits: Lista de hábitos
        
        Returns:
            Análisis de hábitos
        """
        if not habits:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "analysis_id": f"habits_{datetime.now().timestamp()}",
            "total_habits": len(habits),
            "habit_categories": self._categorize_habits(habits),
            "consistency_scores": self._calculate_consistency(habits),
            "strength_habits": self._identify_strong_habits(habits),
            "weak_habits": self._identify_weak_habits(habits),
            "recommendations": self._generate_habit_recommendations(habits),
            "generated_at": datetime.now().isoformat()
        }
    
    def track_habit_progress(
        self,
        user_id: str,
        habit_id: str,
        progress_data: List[Dict]
    ) -> Dict:
        """
        Rastrea progreso de hábito
        
        Args:
            user_id: ID del usuario
            habit_id: ID del hábito
            progress_data: Datos de progreso
        
        Returns:
            Análisis de progreso
        """
        return {
            "user_id": user_id,
            "habit_id": habit_id,
            "total_entries": len(progress_data),
            "completion_rate": self._calculate_completion_rate(progress_data),
            "streak": self._calculate_streak(progress_data),
            "trend": self._analyze_progress_trend(progress_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def _categorize_habits(self, habits: List[Dict]) -> Dict:
        """Categoriza hábitos"""
        categories = defaultdict(int)
        
        for habit in habits:
            category = habit.get("category", "general")
            categories[category] += 1
        
        return dict(categories)
    
    def _calculate_consistency(self, habits: List[Dict]) -> Dict:
        """Calcula consistencia"""
        consistency_scores = {}
        
        for habit in habits:
            habit_id = habit.get("id", "")
            completion_rate = habit.get("completion_rate", 0)
            consistency_scores[habit_id] = round(completion_rate, 2)
        
        return consistency_scores
    
    def _identify_strong_habits(self, habits: List[Dict]) -> List[Dict]:
        """Identifica hábitos fuertes"""
        strong = [h for h in habits if h.get("completion_rate", 0) >= 80]
        return sorted(strong, key=lambda x: x.get("completion_rate", 0), reverse=True)[:5]
    
    def _identify_weak_habits(self, habits: List[Dict]) -> List[Dict]:
        """Identifica hábitos débiles"""
        weak = [h for h in habits if h.get("completion_rate", 0) < 50]
        return sorted(weak, key=lambda x: x.get("completion_rate", 0))[:5]
    
    def _generate_habit_recommendations(self, habits: List[Dict]) -> List[str]:
        """Genera recomendaciones de hábitos"""
        recommendations = []
        
        weak_habits = self._identify_weak_habits(habits)
        if weak_habits:
            recommendations.append("Algunos hábitos necesitan más atención. Considera ajustar tu estrategia")
        
        return recommendations
    
    def _calculate_completion_rate(self, progress_data: List[Dict]) -> float:
        """Calcula tasa de completación"""
        if not progress_data:
            return 0.0
        
        completed = sum(1 for p in progress_data if p.get("completed", False))
        return round((completed / len(progress_data)) * 100, 2)
    
    def _calculate_streak(self, progress_data: List[Dict]) -> int:
        """Calcula racha"""
        streak = 0
        
        for entry in reversed(progress_data):
            if entry.get("completed", False):
                streak += 1
            else:
                break
        
        return streak
    
    def _analyze_progress_trend(self, progress_data: List[Dict]) -> str:
        """Analiza tendencia de progreso"""
        if len(progress_data) < 2:
            return "stable"
        
        first_half = progress_data[:len(progress_data)//2]
        second_half = progress_data[len(progress_data)//2:]
        
        first_rate = self._calculate_completion_rate(first_half)
        second_rate = self._calculate_completion_rate(second_half)
        
        if second_rate > first_rate * 1.1:
            return "improving"
        elif second_rate < first_rate * 0.9:
            return "declining"
        return "stable"

