"""
Servicio de Métricas Avanzadas - Sistema completo de métricas y KPIs
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics
from collections import defaultdict


class AdvancedMetricsService:
    """Servicio de métricas avanzadas"""
    
    def __init__(self):
        """Inicializa el servicio de métricas"""
        pass
    
    def calculate_recovery_kpis(
        self,
        user_id: str,
        data: List[Dict]
    ) -> Dict:
        """
        Calcula KPIs de recuperación
        
        Args:
            user_id: ID del usuario
            data: Datos históricos
        
        Returns:
            KPIs de recuperación
        """
        if not data:
            return {
                "user_id": user_id,
                "error": "No data available"
            }
        
        kpis = {
            "user_id": user_id,
            "sobriety": {
                "current_streak": max([d.get("days_sober", 0) for d in data]),
                "longest_streak": max([d.get("days_sober", 0) for d in data]),
                "total_days": len(data),
                "consistency_rate": self._calculate_consistency(data)
            },
            "engagement": {
                "check_in_rate": self._calculate_check_in_rate(data),
                "average_mood": round(statistics.mean([d.get("mood_score", 5) for d in data]), 2),
                "activity_level": self._calculate_activity_level(data)
            },
            "wellness": {
                "average_sleep": round(statistics.mean([d.get("sleep_hours", 7) for d in data]), 2) if any(d.get("sleep_hours") for d in data) else 0,
                "exercise_frequency": self._calculate_exercise_frequency(data),
                "stress_management": self._calculate_stress_management(data)
            },
            "calculated_at": datetime.now().isoformat()
        }
        
        return kpis
    
    def generate_metrics_dashboard(
        self,
        user_id: str,
        period: str = "30d"
    ) -> Dict:
        """
        Genera dashboard de métricas
        
        Args:
            user_id: ID del usuario
            period: Período (7d, 30d, 90d, all)
        
        Returns:
            Dashboard de métricas
        """
        dashboard = {
            "user_id": user_id,
            "period": period,
            "overview": {
                "total_days": 0,
                "current_streak": 0,
                "success_rate": 0.0
            },
            "trends": {
                "mood_trend": "stable",
                "cravings_trend": "stable",
                "activity_trend": "stable"
            },
            "comparisons": {
                "vs_last_period": {},
                "vs_baseline": {}
            },
            "alerts": [],
            "generated_at": datetime.now().isoformat()
        }
        
        return dashboard
    
    def calculate_engagement_score(
        self,
        user_id: str,
        activities: List[Dict]
    ) -> Dict:
        """
        Calcula puntuación de engagement
        
        Args:
            user_id: ID del usuario
            activities: Lista de actividades
        
        Returns:
            Puntuación de engagement
        """
        if not activities:
            return {
                "user_id": user_id,
                "score": 0,
                "level": "low"
            }
        
        # Calcular score basado en frecuencia y variedad
        activity_count = len(activities)
        unique_activities = len(set(a.get("type", "") for a in activities))
        
        score = min(100, (activity_count * 2) + (unique_activities * 10))
        
        return {
            "user_id": user_id,
            "score": round(score, 2),
            "level": self._get_engagement_level(score),
            "activity_count": activity_count,
            "unique_activities": unique_activities,
            "calculated_at": datetime.now().isoformat()
        }
    
    def _calculate_consistency(self, data: List[Dict]) -> float:
        """Calcula tasa de consistencia"""
        if not data:
            return 0.0
        
        # Asumir que si hay entrada, es consistente
        expected_days = (datetime.now() - datetime.fromisoformat(data[0].get("date", datetime.now().isoformat()))).days + 1
        actual_days = len(data)
        
        return round((actual_days / expected_days) * 100, 2) if expected_days > 0 else 0.0
    
    def _calculate_check_in_rate(self, data: List[Dict]) -> float:
        """Calcula tasa de check-ins"""
        check_ins = sum(1 for d in data if d.get("check_in_completed", False))
        total = len(data)
        
        return round((check_ins / total) * 100, 2) if total > 0 else 0.0
    
    def _calculate_activity_level(self, data: List[Dict]) -> str:
        """Calcula nivel de actividad"""
        activities = sum(1 for d in data if d.get("activities_count", 0) > 0)
        total = len(data)
        
        rate = activities / total if total > 0 else 0
        
        if rate >= 0.7:
            return "high"
        elif rate >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_exercise_frequency(self, data: List[Dict]) -> float:
        """Calcula frecuencia de ejercicio"""
        exercise_days = sum(1 for d in data if d.get("exercise_done", False))
        total = len(data)
        
        return round((exercise_days / total) * 100, 2) if total > 0 else 0.0
    
    def _calculate_stress_management(self, data: List[Dict]) -> float:
        """Calcula manejo de estrés"""
        if not data:
            return 0.0
        
        stress_scores = [d.get("stress_level", 5) for d in data]
        avg_stress = statistics.mean(stress_scores)
        
        # Invertir: menor estrés = mejor manejo
        management_score = 10 - avg_stress
        
        return round(max(0, min(10, management_score)), 2)
    
    def _get_engagement_level(self, score: float) -> str:
        """Obtiene nivel de engagement"""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "moderate"
        else:
            return "low"

