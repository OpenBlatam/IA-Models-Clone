"""
Recommendation Engine
======================

Motor de recomendaciones.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Recomendación."""
    recommendation_id: str
    type: str
    title: str
    description: str
    confidence: float  # 0-1
    action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RecommendationEngine:
    """
    Motor de recomendaciones.
    
    Genera recomendaciones basadas en datos y patrones.
    """
    
    def __init__(self):
        """Inicializar motor de recomendaciones."""
        self.recommendations: List[Recommendation] = []
        self.recommendation_history: List[Dict[str, Any]] = []
        self.max_recommendations = 1000
    
    def generate_recommendation(
        self,
        recommendation_type: str,
        title: str,
        description: str,
        confidence: float,
        action: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Recommendation:
        """
        Generar recomendación.
        
        Args:
            recommendation_type: Tipo de recomendación
            title: Título
            description: Descripción
            confidence: Confianza (0-1)
            action: Acción recomendada (opcional)
            metadata: Metadata adicional
            
        Returns:
            Recomendación creada
        """
        recommendation_id = f"rec_{len(self.recommendations)}"
        
        recommendation = Recommendation(
            recommendation_id=recommendation_id,
            type=recommendation_type,
            title=title,
            description=description,
            confidence=confidence,
            action=action,
            metadata=metadata or {}
        )
        
        self.recommendations.append(recommendation)
        
        # Limitar tamaño
        if len(self.recommendations) > self.max_recommendations:
            self.recommendations = self.recommendations[-self.max_recommendations:]
        
        logger.info(f"Generated recommendation: {title} (confidence: {confidence})")
        
        return recommendation
    
    def get_recommendations(
        self,
        recommendation_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[Recommendation]:
        """
        Obtener recomendaciones.
        
        Args:
            recommendation_type: Filtrar por tipo
            min_confidence: Confianza mínima
            limit: Límite de resultados
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = self.recommendations
        
        if recommendation_type:
            recommendations = [r for r in recommendations if r.type == recommendation_type]
        
        recommendations = [r for r in recommendations if r.confidence >= min_confidence]
        
        # Ordenar por confianza
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations[:limit]
    
    def generate_performance_recommendations(self) -> List[Recommendation]:
        """Generar recomendaciones de performance."""
        from ..performance import get_performance_monitor
        from .analytics import get_analytics_engine
        
        recommendations = []
        
        # Obtener métricas
        monitor = get_performance_monitor()
        metrics = monitor.get_performance_metrics()
        
        # Recomendación: CPU alto
        if metrics.get("cpu_usage", 0) > 80:
            recommendations.append(self.generate_recommendation(
                recommendation_type="performance",
                title="High CPU Usage",
                description=f"CPU usage is {metrics['cpu_usage']:.1f}%",
                confidence=0.8,
                action="Consider reducing workload or scaling up",
                metadata={"cpu_usage": metrics["cpu_usage"]}
            ))
        
        # Recomendación: Memoria alta
        if metrics.get("memory_usage", {}).get("percent", 0) > 80:
            recommendations.append(self.generate_recommendation(
                recommendation_type="performance",
                title="High Memory Usage",
                description=f"Memory usage is {metrics['memory_usage'].get('percent', 0):.1f}%",
                confidence=0.8,
                action="Consider optimizing memory usage or increasing resources",
                metadata={"memory_usage": metrics["memory_usage"]}
            ))
        
        # Recomendación: Response time alto
        if metrics.get("response_time_p95", 0) > 1.0:
            recommendations.append(self.generate_recommendation(
                recommendation_type="performance",
                title="Slow Response Time",
                description=f"P95 response time is {metrics['response_time_p95']:.2f}s",
                confidence=0.9,
                action="Consider optimizing algorithms or caching",
                metadata={"response_time_p95": metrics["response_time_p95"]}
            ))
        
        return recommendations
    
    def generate_optimization_recommendations(self) -> List[Recommendation]:
        """Generar recomendaciones de optimización."""
        from .analytics import get_analytics_engine
        
        recommendations = []
        engine = get_analytics_engine()
        
        # Obtener estadísticas
        stats = engine.get_operation_statistics()
        
        # Recomendación: Cache hit rate bajo
        if stats.get("cache_hit_rate", 1.0) < 0.5:
            recommendations.append(self.generate_recommendation(
                recommendation_type="optimization",
                title="Low Cache Hit Rate",
                description="Cache hit rate is below optimal",
                confidence=0.7,
                action="Consider increasing cache size or improving cache strategy",
                metadata={"cache_hit_rate": stats.get("cache_hit_rate", 0)}
            ))
        
        return recommendations
    
    def get_recommendation_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de recomendaciones."""
        if not self.recommendations:
            return {
                "total_recommendations": 0,
                "by_type": {},
                "average_confidence": 0.0
            }
        
        by_type = {}
        for rec in self.recommendations:
            by_type[rec.type] = by_type.get(rec.type, 0) + 1
        
        avg_confidence = sum(r.confidence for r in self.recommendations) / len(self.recommendations)
        
        return {
            "total_recommendations": len(self.recommendations),
            "by_type": by_type,
            "average_confidence": avg_confidence
        }


# Instancia global
_recommendation_engine: Optional[RecommendationEngine] = None


def get_recommendation_engine() -> RecommendationEngine:
    """Obtener instancia global del motor de recomendaciones."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine






