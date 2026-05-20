"""
Sistema de analytics y métricas del sistema
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class AnalyticsEngine:
    """Motor de analytics para análisis de datos"""
    
    def __init__(self, db_manager):
        """
        Inicializa el motor de analytics
        
        Args:
            db_manager: Instancia de DatabaseManager
        """
        self.db = db_manager
    
    def get_user_insights(self, user_id: str, days: int = 30) -> Dict:
        """
        Obtiene insights para un usuario
        
        Args:
            user_id: ID del usuario
            days: Días de historial a analizar
            
        Returns:
            Diccionario con insights
        """
        # Obtener historial
        history = self.db.get_user_history(user_id, limit=1000)
        
        if not history:
            return {"message": "No hay datos suficientes"}
        
        # Filtrar por fecha
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in history
            if datetime.fromisoformat(h["timestamp"]) >= cutoff_date
        ]
        
        if not recent_history:
            return {"message": "No hay datos recientes"}
        
        # Calcular métricas
        scores = [h["quality_scores"].get("overall_score", 0) for h in recent_history]
        
        # Tendencias
        if len(scores) >= 2:
            trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
            trend_percentage = ((scores[-1] - scores[0]) / scores[0] * 100) if scores[0] > 0 else 0
        else:
            trend = "insufficient_data"
            trend_percentage = 0
        
        # Condiciones más comunes
        all_conditions = []
        for h in recent_history:
            all_conditions.extend([c.get("name") for c in h.get("conditions", [])])
        
        condition_counts = defaultdict(int)
        for condition in all_conditions:
            condition_counts[condition] += 1
        
        most_common_conditions = sorted(
            condition_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Áreas que más necesitan mejora
        priority_areas = defaultdict(int)
        for h in recent_history:
            for priority in h.get("recommendations_priority", []):
                priority_areas[priority] += 1
        
        top_priorities = sorted(
            priority_areas.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Tipos de piel más frecuentes
        skin_types = [h.get("skin_type", "unknown") for h in recent_history]
        most_common_skin_type = max(set(skin_types), key=skin_types.count) if skin_types else "unknown"
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_analyses": len(recent_history),
            "statistics": {
                "average_score": round(statistics.mean(scores), 2),
                "median_score": round(statistics.median(scores), 2),
                "min_score": round(min(scores), 2),
                "max_score": round(max(scores), 2),
                "std_deviation": round(statistics.stdev(scores) if len(scores) > 1 else 0, 2)
            },
            "trend": {
                "direction": trend,
                "percentage_change": round(trend_percentage, 2),
                "current_score": round(scores[-1], 2) if scores else 0,
                "initial_score": round(scores[0], 2) if scores else 0
            },
            "most_common_conditions": [
                {"name": name, "frequency": count}
                for name, count in most_common_conditions
            ],
            "top_priority_areas": [
                {"area": area, "frequency": count}
                for area, count in top_priorities
            ],
            "most_common_skin_type": most_common_skin_type,
            "recommendations": self._generate_insight_recommendations(
                trend, scores, most_common_conditions, top_priorities
            )
        }
    
    def get_system_analytics(self, days: int = 30) -> Dict:
        """
        Obtiene analytics del sistema completo
        
        Args:
            days: Días de historial a analizar
            
        Returns:
            Diccionario con analytics del sistema
        """
        # Obtener todas las estadísticas
        stats = self.db.get_statistics()
        
        # Analytics adicionales
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return {
            "period_days": days,
            "total_analyses": stats.get("total_analyses", 0),
            "average_score": stats.get("average_score", 0),
            "score_range": {
                "min": stats.get("min_score", 0),
                "max": stats.get("max_score", 0)
            },
            "generated_at": datetime.now().isoformat()
        }
    
    def get_progress_report(self, user_id: str, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict:
        """
        Genera reporte de progreso para un usuario
        
        Args:
            user_id: ID del usuario
            start_date: Fecha de inicio (ISO format)
            end_date: Fecha de fin (ISO format)
            
        Returns:
            Diccionario con reporte de progreso
        """
        history = self.db.get_user_history(user_id, limit=1000)
        
        if not history:
            return {"message": "No hay datos disponibles"}
        
        # Filtrar por fechas si se proporcionan
        if start_date:
            start = datetime.fromisoformat(start_date)
            history = [h for h in history if datetime.fromisoformat(h["timestamp"]) >= start]
        
        if end_date:
            end = datetime.fromisoformat(end_date)
            history = [h for h in history if datetime.fromisoformat(h["timestamp"]) <= end]
        
        if not history:
            return {"message": "No hay datos en el rango especificado"}
        
        # Ordenar por fecha
        history.sort(key=lambda x: x["timestamp"])
        
        # Análisis de progreso por métrica
        metrics_progress = {}
        metric_names = ["overall_score", "texture_score", "hydration_score", 
                       "elasticity_score", "pigmentation_score", "pore_size_score",
                       "wrinkles_score", "redness_score", "dark_spots_score"]
        
        for metric in metric_names:
            values = [h["quality_scores"].get(metric, 0) for h in history]
            if values:
                metrics_progress[metric] = {
                    "initial": round(values[0], 2),
                    "final": round(values[-1], 2),
                    "change": round(values[-1] - values[0], 2),
                    "percentage_change": round(((values[-1] - values[0]) / values[0] * 100) if values[0] > 0 else 0, 2),
                    "average": round(statistics.mean(values), 2),
                    "trend": "improving" if values[-1] > values[0] else "declining" if values[-1] < values[0] else "stable"
                }
        
        return {
            "user_id": user_id,
            "period": {
                "start": history[0]["timestamp"],
                "end": history[-1]["timestamp"],
                "days": (datetime.fromisoformat(history[-1]["timestamp"]) - 
                        datetime.fromisoformat(history[0]["timestamp"])).days
            },
            "total_analyses": len(history),
            "metrics_progress": metrics_progress,
            "overall_improvement": round(
                metrics_progress.get("overall_score", {}).get("change", 0), 2
            )
        }
    
    def _generate_insight_recommendations(self, trend: str, scores: List[float],
                                        conditions: List, priorities: List) -> List[str]:
        """Genera recomendaciones basadas en insights"""
        recommendations = []
        
        if trend == "declining":
            recommendations.append(
                "Tu score general está disminuyendo. Considera revisar tu rutina de skincare."
            )
        elif trend == "improving":
            recommendations.append(
                "¡Excelente! Estás viendo mejoras en tu piel. Continúa con tu rutina actual."
            )
        
        if scores and scores[-1] < 50:
            recommendations.append(
                "Tu score general es bajo. Te recomendamos consultar con un dermatólogo."
            )
        
        if conditions:
            recommendations.append(
                f"Has tenido {len(conditions)} condiciones detectadas. "
                "Considera productos específicos para estas condiciones."
            )
        
        if priorities:
            top_priority = priorities[0][0] if priorities else None
            if top_priority:
                recommendations.append(
                    f"El área que más necesita atención es: {top_priority}. "
                    "Enfócate en productos para esta área."
                )
        
        return recommendations






