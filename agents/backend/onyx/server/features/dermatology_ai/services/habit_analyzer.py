"""
Sistema de análisis de hábitos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics


@dataclass
class HabitPattern:
    """Patrón de hábito"""
    habit_type: str
    frequency: float  # veces por semana
    consistency: float  # 0-1
    trend: str  # "improving", "declining", "stable"
    impact_score: float  # 0-100
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "habit_type": self.habit_type,
            "frequency": self.frequency,
            "consistency": self.consistency,
            "trend": self.trend,
            "impact_score": self.impact_score,
            "recommendations": self.recommendations
        }


@dataclass
class HabitAnalysis:
    """Análisis de hábitos"""
    user_id: str
    patterns: List[HabitPattern]
    overall_score: float
    top_habits: List[str]
    improvement_areas: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "patterns": [p.to_dict() for p in self.patterns],
            "overall_score": self.overall_score,
            "top_habits": self.top_habits,
            "improvement_areas": self.improvement_areas,
            "created_at": self.created_at
        }


class HabitAnalyzer:
    """Sistema de análisis de hábitos"""
    
    def __init__(self):
        """Inicializa el analizador"""
        self.habit_data: Dict[str, List[Dict]] = {}  # user_id -> [habit_records]
    
    def record_habit(self, user_id: str, habit_type: str, date: str,
                         completed: bool, notes: Optional[str] = None):
        """Registra un hábito"""
        record = {
            "habit_type": habit_type,
            "date": date,
            "completed": completed,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        
        if user_id not in self.habit_data:
            self.habit_data[user_id] = []
        
        self.habit_data[user_id].append(record)
    
    def analyze_habits(self, user_id: str, days: int = 30) -> HabitAnalysis:
        """Analiza hábitos del usuario"""
        user_records = self.habit_data.get(user_id, [])
        
        # Filtrar por período
        cutoff = datetime.now() - timedelta(days=days)
        recent_records = [
            r for r in user_records
            if datetime.fromisoformat(r["date"]) >= cutoff
        ]
        
        # Agrupar por tipo de hábito
        habit_types = set(r["habit_type"] for r in recent_records)
        patterns = []
        
        for habit_type in habit_types:
            habit_records = [r for r in recent_records if r["habit_type"] == habit_type]
            
            # Calcular frecuencia
            completed_count = sum(1 for r in habit_records if r["completed"])
            frequency = (completed_count / days) * 7  # veces por semana
            
            # Calcular consistencia
            total_days = len(set(r["date"] for r in habit_records))
            consistency = completed_count / total_days if total_days > 0 else 0.0
            
            # Determinar tendencia
            trend = self._determine_trend(habit_records)
            
            # Calcular impacto
            impact_score = self._calculate_impact(habit_type, frequency, consistency)
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(habit_type, frequency, consistency, trend)
            
            pattern = HabitPattern(
                habit_type=habit_type,
                frequency=frequency,
                consistency=consistency,
                trend=trend,
                impact_score=impact_score,
                recommendations=recommendations
            )
            patterns.append(pattern)
        
        # Calcular score general
        overall_score = statistics.mean([p.impact_score for p in patterns]) if patterns else 0.0
        
        # Top hábitos
        top_habits = sorted(patterns, key=lambda x: x.impact_score, reverse=True)[:3]
        top_habits_names = [h.habit_type for h in top_habits]
        
        # Áreas de mejora
        improvement_areas = [
            p.habit_type for p in patterns
            if p.consistency < 0.5 or p.frequency < 1.0
        ]
        
        return HabitAnalysis(
            user_id=user_id,
            patterns=patterns,
            overall_score=overall_score,
            top_habits=top_habits_names,
            improvement_areas=improvement_areas
        )
    
    def _determine_trend(self, records: List[Dict]) -> str:
        """Determina tendencia del hábito"""
        if len(records) < 7:
            return "stable"
        
        # Dividir en dos períodos
        mid_point = len(records) // 2
        first_half = records[:mid_point]
        second_half = records[mid_point:]
        
        first_completion = sum(1 for r in first_half if r["completed"]) / len(first_half)
        second_completion = sum(1 for r in second_half if r["completed"]) / len(second_half)
        
        if second_completion > first_completion * 1.1:
            return "improving"
        elif second_completion < first_completion * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _calculate_impact(self, habit_type: str, frequency: float, consistency: float) -> float:
        """Calcula impacto del hábito"""
        base_impact = {
            "skincare_routine": 30.0,
            "sunscreen_application": 25.0,
            "water_intake": 15.0,
            "sleep": 20.0,
            "exercise": 10.0
        }
        
        base = base_impact.get(habit_type, 10.0)
        impact = base * consistency * min(frequency / 7.0, 1.0)
        
        return float(min(100.0, impact))
    
    def _generate_recommendations(self, habit_type: str, frequency: float,
                                 consistency: float, trend: str) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        if consistency < 0.5:
            recommendations.append(f"Mejora la consistencia de {habit_type}")
        
        if frequency < 1.0:
            recommendations.append(f"Aumenta la frecuencia de {habit_type}")
        
        if trend == "declining":
            recommendations.append(f"El hábito {habit_type} está empeorando. Revisa tu rutina.")
        
        return recommendations






