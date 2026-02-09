"""
Satisfaction Analyzer - Sistema de análisis de satisfacción
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SatisfactionMetric:
    """Métrica de satisfacción"""
    content_id: str
    user_id: Optional[str]
    satisfaction_score: float
    metric_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


class SatisfactionAnalyzer:
    """Analizador de satisfacción"""

    def __init__(self):
        """Inicializar analizador"""
        self.satisfaction_metrics: List[SatisfactionMetric] = []
        self.content_scores: Dict[str, List[float]] = defaultdict(list)

    def record_satisfaction(
        self,
        content_id: str,
        satisfaction_score: float,
        user_id: Optional[str] = None,
        metric_type: str = "overall",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar métrica de satisfacción.

        Args:
            content_id: ID del contenido
            satisfaction_score: Score de satisfacción (0-1)
            user_id: ID del usuario (opcional)
            metric_type: Tipo de métrica
            metadata: Metadatos adicionales
        """
        metric = SatisfactionMetric(
            content_id=content_id,
            user_id=user_id,
            satisfaction_score=satisfaction_score,
            metric_type=metric_type,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.satisfaction_metrics.append(metric)
        self.content_scores[content_id].append(satisfaction_score)
        
        logger.debug(f"Métrica de satisfacción registrada: {content_id} - {satisfaction_score}")

    def analyze_content_satisfaction(
        self,
        content_id: str,
        period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analizar satisfacción de un contenido.

        Args:
            content_id: ID del contenido
            period_days: Período en días (opcional)

        Returns:
            Análisis de satisfacción
        """
        # Filtrar métricas
        metrics = [
            m for m in self.satisfaction_metrics
            if m.content_id == content_id
        ]
        
        if period_days:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)
            metrics = [m for m in metrics if m.timestamp >= cutoff_date]
        
        if not metrics:
            return {"error": "No hay métricas de satisfacción para este contenido"}
        
        # Calcular estadísticas
        scores = [m.satisfaction_score for m in metrics]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Distribución por tipo de métrica
        metric_types = defaultdict(list)
        for m in metrics:
            metric_types[m.metric_type].append(m.satisfaction_score)
        
        type_averages = {
            metric_type: sum(scores) / len(scores)
            for metric_type, scores in metric_types.items()
        }
        
        # Tendencias (últimos 7 días vs anteriores)
        if len(metrics) >= 2:
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_metrics = [m for m in metrics if m.timestamp >= recent_cutoff]
            older_metrics = [m for m in metrics if m.timestamp < recent_cutoff]
            
            if recent_metrics and older_metrics:
                recent_avg = sum(m.satisfaction_score for m in recent_metrics) / len(recent_metrics)
                older_avg = sum(m.satisfaction_score for m in older_metrics) / len(older_metrics)
                trend = recent_avg - older_avg
            else:
                trend = 0.0
        else:
            trend = 0.0
        
        return {
            "content_id": content_id,
            "total_metrics": len(metrics),
            "average_satisfaction": avg_score,
            "min_satisfaction": min_score,
            "max_satisfaction": max_score,
            "satisfaction_by_type": type_averages,
            "trend": trend,
            "trend_direction": "improving" if trend > 0.05 else "declining" if trend < -0.05 else "stable",
            "period_days": period_days
        }

    def get_overall_satisfaction(
        self,
        period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Obtener satisfacción general.

        Args:
            period_days: Período en días (opcional)

        Returns:
            Satisfacción general
        """
        metrics = self.satisfaction_metrics
        
        if period_days:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)
            metrics = [m for m in metrics if m.timestamp >= cutoff_date]
        
        if not metrics:
            return {"error": "No hay métricas de satisfacción disponibles"}
        
        scores = [m.satisfaction_score for m in metrics]
        avg_score = sum(scores) / len(scores)
        
        # Distribución de scores
        score_ranges = {
            "excellent": sum(1 for s in scores if s >= 0.8),
            "good": sum(1 for s in scores if 0.6 <= s < 0.8),
            "fair": sum(1 for s in scores if 0.4 <= s < 0.6),
            "poor": sum(1 for s in scores if s < 0.4)
        }
        
        # Top contenidos
        content_averages = {}
        for content_id, content_scores in self.content_scores.items():
            if content_scores:
                content_averages[content_id] = sum(content_scores) / len(content_scores)
        
        top_content = sorted(
            content_averages.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "average_satisfaction": avg_score,
            "total_metrics": len(metrics),
            "score_distribution": score_ranges,
            "top_content": [
                {"content_id": content_id, "average_satisfaction": score}
                for content_id, score in top_content
            ],
            "period_days": period_days
        }

    def get_satisfaction_recommendations(
        self,
        content_id: str
    ) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones basadas en satisfacción.

        Args:
            content_id: ID del contenido

        Returns:
            Recomendaciones
        """
        analysis = self.analyze_content_satisfaction(content_id)
        
        if "error" in analysis:
            return []
        
        recommendations = []
        
        avg_satisfaction = analysis.get("average_satisfaction", 0.5)
        trend = analysis.get("trend", 0.0)
        
        if avg_satisfaction < 0.5:
            recommendations.append({
                "priority": "high",
                "category": "satisfaction",
                "title": "Satisfacción baja",
                "description": f"La satisfacción promedio es {avg_satisfaction:.2f}",
                "suggestion": "Revisa el contenido y considera mejoras significativas"
            })
        
        if trend < -0.1:
            recommendations.append({
                "priority": "high",
                "category": "trend",
                "title": "Tendencia negativa",
                "description": "La satisfacción está disminuyendo",
                "suggestion": "Investiga las causas de la disminución y toma medidas correctivas"
            })
        
        # Analizar por tipo de métrica
        satisfaction_by_type = analysis.get("satisfaction_by_type", {})
        for metric_type, score in satisfaction_by_type.items():
            if score < 0.5:
                recommendations.append({
                    "priority": "medium",
                    "category": metric_type,
                    "title": f"Satisfacción baja en {metric_type}",
                    "description": f"Score: {score:.2f}",
                    "suggestion": f"Mejora específica en el área de {metric_type}"
                })
        
        return recommendations






