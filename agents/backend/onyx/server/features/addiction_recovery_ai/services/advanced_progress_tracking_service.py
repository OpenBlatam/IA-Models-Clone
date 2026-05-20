"""
Servicio de Seguimiento Visual Avanzado de Progreso - Sistema completo de visualización de progreso
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedProgressTrackingService:
    """Servicio de seguimiento visual avanzado de progreso"""
    
    def __init__(self):
        """Inicializa el servicio de seguimiento avanzado"""
        pass
    
    def generate_progress_visualization(
        self,
        user_id: str,
        data: List[Dict],
        visualization_type: str = "comprehensive"
    ) -> Dict:
        """
        Genera visualización avanzada de progreso
        
        Args:
            user_id: ID del usuario
            data: Datos históricos
            visualization_type: Tipo de visualización
        
        Returns:
            Visualización de progreso
        """
        if not data:
            return {
                "user_id": user_id,
                "error": "No data available"
            }
        
        # Calcular métricas clave
        days_sober = [entry.get("days_sober", 0) for entry in data]
        mood_scores = [entry.get("mood_score", 5) for entry in data]
        cravings = [entry.get("cravings_level", 3) for entry in data]
        
        visualization = {
            "user_id": user_id,
            "visualization_type": visualization_type,
            "charts": {
                "progress_timeline": self._generate_timeline_chart(data),
                "mood_trend": self._generate_trend_chart(mood_scores, "Mood"),
                "cravings_trend": self._generate_trend_chart(cravings, "Cravings"),
                "heatmap": self._generate_heatmap(data),
                "radar": self._generate_radar_chart(data)
            },
            "metrics": {
                "current_streak": max(days_sober) if days_sober else 0,
                "average_mood": round(statistics.mean(mood_scores), 2) if mood_scores else 0,
                "average_cravings": round(statistics.mean(cravings), 2) if cravings else 0,
                "improvement_rate": self._calculate_improvement_rate(data)
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return visualization
    
    def generate_comparison_view(
        self,
        user_id: str,
        period1_data: List[Dict],
        period2_data: List[Dict]
    ) -> Dict:
        """
        Genera vista comparativa entre períodos
        
        Args:
            user_id: ID del usuario
            period1_data: Datos del período 1
            period2_data: Datos del período 2
        
        Returns:
            Vista comparativa
        """
        comparison = {
            "user_id": user_id,
            "period1": self._analyze_period(period1_data),
            "period2": self._analyze_period(period2_data),
            "improvements": self._calculate_improvements(period1_data, period2_data),
            "generated_at": datetime.now().isoformat()
        }
        
        return comparison
    
    def generate_milestone_timeline(
        self,
        user_id: str,
        milestones: List[Dict]
    ) -> Dict:
        """
        Genera línea de tiempo de hitos
        
        Args:
            user_id: ID del usuario
            milestones: Lista de hitos
        
        Returns:
            Línea de tiempo de hitos
        """
        timeline = {
            "user_id": user_id,
            "milestones": sorted(milestones, key=lambda x: x.get("date", "")),
            "total_milestones": len(milestones),
            "upcoming_milestones": self._get_upcoming_milestones(milestones),
            "generated_at": datetime.now().isoformat()
        }
        
        return timeline
    
    def _generate_timeline_chart(self, data: List[Dict]) -> Dict:
        """Genera gráfico de línea de tiempo"""
        return {
            "type": "line",
            "data": {
                "labels": [entry.get("date", "") for entry in data],
                "datasets": [{
                    "label": "Progreso",
                    "data": [entry.get("days_sober", 0) for entry in data]
                }]
            }
        }
    
    def _generate_trend_chart(self, values: List[float], label: str) -> Dict:
        """Genera gráfico de tendencia"""
        return {
            "type": "line",
            "data": {
                "labels": list(range(len(values))),
                "datasets": [{
                    "label": label,
                    "data": values,
                    "fill": True
                }]
            }
        }
    
    def _generate_heatmap(self, data: List[Dict]) -> Dict:
        """Genera mapa de calor de actividad"""
        return {
            "type": "heatmap",
            "data": {
                "days": [entry.get("date", "") for entry in data],
                "intensity": [entry.get("activity_level", 0) for entry in data]
            }
        }
    
    def _generate_radar_chart(self, data: List[Dict]) -> Dict:
        """Genera gráfico de radar"""
        return {
            "type": "radar",
            "data": {
                "labels": ["Mood", "Energy", "Support", "Exercise", "Sleep"],
                "datasets": [{
                    "label": "Current",
                    "data": [5, 5, 5, 5, 5]
                }]
            }
        }
    
    def _calculate_improvement_rate(self, data: List[Dict]) -> float:
        """Calcula tasa de mejora"""
        if len(data) < 2:
            return 0.0
        
        first_mood = data[0].get("mood_score", 5)
        last_mood = data[-1].get("mood_score", 5)
        
        improvement = ((last_mood - first_mood) / first_mood) * 100 if first_mood > 0 else 0
        return round(improvement, 2)
    
    def _analyze_period(self, data: List[Dict]) -> Dict:
        """Analiza un período"""
        if not data:
            return {}
        
        mood_scores = [entry.get("mood_score", 5) for entry in data]
        return {
            "days": len(data),
            "average_mood": round(statistics.mean(mood_scores), 2) if mood_scores else 0,
            "best_day": max(data, key=lambda x: x.get("mood_score", 0)) if data else None
        }
    
    def _calculate_improvements(self, period1: List[Dict], period2: List[Dict]) -> List[str]:
        """Calcula mejoras entre períodos"""
        improvements = []
        
        if period1 and period2:
            avg1 = statistics.mean([e.get("mood_score", 5) for e in period1])
            avg2 = statistics.mean([e.get("mood_score", 5) for e in period2])
            
            if avg2 > avg1:
                improvements.append(f"Mejora en estado de ánimo: {round(avg2 - avg1, 2)} puntos")
        
        return improvements
    
    def _get_upcoming_milestones(self, milestones: List[Dict]) -> List[Dict]:
        """Obtiene próximos hitos"""
        today = datetime.now().date()
        upcoming = [m for m in milestones if datetime.fromisoformat(m.get("date", "")).date() > today]
        return sorted(upcoming, key=lambda x: x.get("date", ""))[:5]

