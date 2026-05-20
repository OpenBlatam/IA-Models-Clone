"""
Quality Tracker
===============

Tracking de calidad de generación.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityTracker:
    """Tracker de calidad."""
    
    def __init__(self, history_size: int = 100):
        """
        Inicializar tracker.
        
        Args:
            history_size: Tamaño del historial
        """
        self.history_size = history_size
        self.quality_history: deque = deque(maxlen=history_size)
        self.feedback_history: deque = deque(maxlen=history_size)
        self._logger = logger
    
    def record_quality(
        self,
        manual_id: int,
        metrics: Dict[str, float],
        user_rating: Optional[float] = None
    ):
        """
        Registrar calidad de manual.
        
        Args:
            manual_id: ID del manual
            metrics: Métricas de calidad
            user_rating: Rating del usuario
        """
        entry = {
            "manual_id": manual_id,
            "metrics": metrics,
            "user_rating": user_rating,
            "timestamp": datetime.now().isoformat()
        }
        
        self.quality_history.append(entry)
        
        if user_rating:
            self.feedback_history.append({
                "manual_id": manual_id,
                "rating": user_rating,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de calidad."""
        if not self.quality_history:
            return {
                "total_manuals": 0,
                "avg_rating": 0.0,
                "metrics": {}
            }
        
        ratings = [
            entry["user_rating"]
            for entry in self.quality_history
            if entry.get("user_rating") is not None
        ]
        
        # Agregar métricas
        all_metrics = {}
        for entry in self.quality_history:
            for metric_name, value in entry.get("metrics", {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calcular promedios
        avg_metrics = {
            name: sum(values) / len(values)
            for name, values in all_metrics.items()
        }
        
        return {
            "total_manuals": len(self.quality_history),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0.0,
            "total_ratings": len(ratings),
            "metrics": avg_metrics,
            "recent_trend": self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calcular tendencia reciente."""
        if len(self.feedback_history) < 10:
            return "insufficient_data"
        
        recent = list(self.feedback_history)[-10:]
        older = list(self.feedback_history)[-20:-10] if len(self.feedback_history) >= 20 else []
        
        if not older:
            return "insufficient_data"
        
        recent_avg = sum(r["rating"] for r in recent) / len(recent)
        older_avg = sum(r["rating"] for r in older) / len(older)
        
        if recent_avg > older_avg + 0.2:
            return "improving"
        elif recent_avg < older_avg - 0.2:
            return "declining"
        else:
            return "stable"




