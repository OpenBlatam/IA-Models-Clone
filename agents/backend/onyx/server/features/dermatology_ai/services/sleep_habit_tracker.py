"""
Sistema de seguimiento de hábitos de sueño
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, date, timedelta
import uuid


@dataclass
class SleepRecord:
    """Registro de sueño"""
    id: str
    user_id: str
    sleep_date: str
    bedtime: str  # HH:MM format
    wake_time: str  # HH:MM format
    sleep_duration_hours: float
    sleep_quality: str  # "poor", "fair", "good", "excellent"
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "sleep_date": self.sleep_date,
            "bedtime": self.bedtime,
            "wake_time": self.wake_time,
            "sleep_duration_hours": self.sleep_duration_hours,
            "sleep_quality": self.sleep_quality,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class SleepAnalysis:
    """Análisis de hábitos de sueño"""
    user_id: str
    average_sleep_hours: float
    average_sleep_quality: str
    sleep_consistency: str  # "consistent", "irregular", "very_irregular"
    recommendations: List[str]
    skin_impact: Dict
    days_analyzed: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "average_sleep_hours": self.average_sleep_hours,
            "average_sleep_quality": self.average_sleep_quality,
            "sleep_consistency": self.sleep_consistency,
            "recommendations": self.recommendations,
            "skin_impact": self.skin_impact,
            "days_analyzed": self.days_analyzed
        }


class SleepHabitTracker:
    """Sistema de seguimiento de hábitos de sueño"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.records: Dict[str, List[SleepRecord]] = {}  # user_id -> [records]
    
    def add_sleep_record(self, user_id: str, sleep_date: str, bedtime: str,
                        wake_time: str, sleep_duration_hours: float,
                        sleep_quality: str, notes: Optional[str] = None) -> SleepRecord:
        """Agrega registro de sueño"""
        record = SleepRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            sleep_date=sleep_date,
            bedtime=bedtime,
            wake_time=wake_time,
            sleep_duration_hours=sleep_duration_hours,
            sleep_quality=sleep_quality,
            notes=notes
        )
        
        if user_id not in self.records:
            self.records[user_id] = []
        
        self.records[user_id].append(record)
        return record
    
    def analyze_sleep_habits(self, user_id: str, days: int = 30) -> SleepAnalysis:
        """Analiza hábitos de sueño"""
        user_records = self.records.get(user_id, [])
        
        if not user_records:
            return SleepAnalysis(
                user_id=user_id,
                average_sleep_hours=0.0,
                average_sleep_quality="unknown",
                sleep_consistency="unknown",
                recommendations=["Agrega registros de sueño para análisis"],
                skin_impact={},
                days_analyzed=0
            )
        
        # Filtrar por días
        cutoff = datetime.now().date() - timedelta(days=days)
        recent_records = [
            r for r in user_records
            if datetime.fromisoformat(r.sleep_date).date() >= cutoff
        ]
        
        if not recent_records:
            return SleepAnalysis(
                user_id=user_id,
                average_sleep_hours=0.0,
                average_sleep_quality="unknown",
                sleep_consistency="unknown",
                recommendations=["No hay registros recientes"],
                skin_impact={},
                days_analyzed=0
            )
        
        # Calcular promedio de horas
        avg_hours = sum(r.sleep_duration_hours for r in recent_records) / len(recent_records)
        
        # Calcular calidad promedio
        quality_scores = {"poor": 1, "fair": 2, "good": 3, "excellent": 4}
        avg_quality_score = sum(quality_scores.get(r.sleep_quality, 2) for r in recent_records) / len(recent_records)
        
        if avg_quality_score >= 3.5:
            avg_quality = "excellent"
        elif avg_quality_score >= 2.5:
            avg_quality = "good"
        elif avg_quality_score >= 1.5:
            avg_quality = "fair"
        else:
            avg_quality = "poor"
        
        # Calcular consistencia
        hour_variance = sum((r.sleep_duration_hours - avg_hours) ** 2 for r in recent_records) / len(recent_records)
        
        if hour_variance < 1.0:
            consistency = "consistent"
        elif hour_variance < 2.0:
            consistency = "irregular"
        else:
            consistency = "very_irregular"
        
        # Recomendaciones
        recommendations = []
        
        if avg_hours < 6:
            recommendations.append("Duermes menos de 6 horas. Esto afecta la regeneración de la piel.")
            recommendations.append("Intenta dormir 7-8 horas diarias para mejor salud de la piel")
        
        if avg_quality in ["poor", "fair"]:
            recommendations.append("Mejora la calidad de tu sueño con rutinas antes de dormir")
            recommendations.append("Considera productos reparadores nocturnos")
        
        if consistency in ["irregular", "very_irregular"]:
            recommendations.append("Mantén un horario de sueño consistente")
            recommendations.append("La irregularidad afecta los ritmos circadianos de la piel")
        
        # Impacto en la piel
        skin_impact = {
            "regeneration_score": min(100, avg_hours * 10 + (quality_scores.get(avg_quality, 2) * 5)),
            "dark_circles_risk": "high" if avg_hours < 6 else "medium" if avg_hours < 7 else "low",
            "skin_repair": "optimal" if avg_hours >= 7 and avg_quality in ["good", "excellent"] else "suboptimal"
        }
        
        return SleepAnalysis(
            user_id=user_id,
            average_sleep_hours=avg_hours,
            average_sleep_quality=avg_quality,
            sleep_consistency=consistency,
            recommendations=recommendations,
            skin_impact=skin_impact,
            days_analyzed=len(recent_records)
        )

