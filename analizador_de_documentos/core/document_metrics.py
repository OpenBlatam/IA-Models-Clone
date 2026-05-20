"""
Document Metrics - Dashboard de Métricas
========================================

Sistema de métricas y dashboard para análisis de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetric:
    """Métrica de documento."""
    metric_name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsDashboard:
    """Dashboard de métricas."""
    period: str  # 'daily', 'weekly', 'monthly'
    start_date: datetime
    end_date: datetime
    total_documents: int
    total_analyses: int
    average_quality_score: float
    average_grammar_score: float
    average_processing_time: float
    top_categories: List[Dict[str, Any]] = field(default_factory=list)
    trends: Dict[str, Any] = field(default_factory=dict)
    metrics: List[DocumentMetric] = field(default_factory=list)


class MetricsCollector:
    """Recolector de métricas."""
    
    def __init__(self, analyzer):
        """Inicializar recolector."""
        self.analyzer = analyzer
        self.metrics_history: List[DocumentMetric] = []
        self.analysis_history: List[Dict[str, Any]] = []
    
    def record_analysis(
        self,
        document_id: str,
        analysis_result: Any,
        processing_time: float,
        quality_score: Optional[float] = None,
        grammar_score: Optional[float] = None
    ):
        """Registrar análisis para métricas."""
        self.analysis_history.append({
            "document_id": document_id,
            "timestamp": datetime.now(),
            "processing_time": processing_time,
            "quality_score": quality_score,
            "grammar_score": grammar_score,
            "has_classification": hasattr(analysis_result, 'classification'),
            "has_summary": hasattr(analysis_result, 'summary') and bool(analysis_result.summary),
            "keyword_count": len(analysis_result.keywords) if hasattr(analysis_result, 'keywords') else 0
        })
    
    async def generate_dashboard(
        self,
        period: str = "daily",
        days: int = 7
    ) -> MetricsDashboard:
        """
        Generar dashboard de métricas.
        
        Args:
            period: Período ('daily', 'weekly', 'monthly')
            days: Número de días a analizar
        
        Returns:
            MetricsDashboard con estadísticas
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filtrar análisis del período
        period_analyses = [
            a for a in self.analysis_history
            if start_date <= a["timestamp"] <= end_date
        ]
        
        if not period_analyses:
            return MetricsDashboard(
                period=period,
                start_date=start_date,
                end_date=end_date,
                total_documents=0,
                total_analyses=0,
                average_quality_score=0.0,
                average_grammar_score=0.0,
                average_processing_time=0.0
            )
        
        # Calcular métricas
        total_documents = len(set(a["document_id"] for a in period_analyses))
        total_analyses = len(period_analyses)
        
        quality_scores = [a["quality_score"] for a in period_analyses if a["quality_score"]]
        grammar_scores = [a["grammar_score"] for a in period_analyses if a["grammar_score"]]
        processing_times = [a["processing_time"] for a in period_analyses]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        avg_grammar = sum(grammar_scores) / len(grammar_scores) if grammar_scores else 0.0
        avg_processing = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        # Calcular tendencias
        trends = self._calculate_trends(period_analyses, start_date, end_date)
        
        # Top categorías (simulado)
        top_categories = self._calculate_top_categories(period_analyses)
        
        # Métricas adicionales
        metrics = self._generate_additional_metrics(period_analyses)
        
        return MetricsDashboard(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_documents=total_documents,
            total_analyses=total_analyses,
            average_quality_score=avg_quality,
            average_grammar_score=avg_grammar,
            average_processing_time=avg_processing,
            top_categories=top_categories,
            trends=trends,
            metrics=metrics
        )
    
    def _calculate_trends(
        self,
        analyses: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calcular tendencias."""
        # Dividir en períodos
        period_days = (end_date - start_date).days
        if period_days == 0:
            return {}
        
        # Agrupar por día
        daily_quality = defaultdict(list)
        daily_grammar = defaultdict(list)
        daily_processing = defaultdict(list)
        
        for analysis in analyses:
            day = analysis["timestamp"].date()
            if analysis["quality_score"]:
                daily_quality[day].append(analysis["quality_score"])
            if analysis["grammar_score"]:
                daily_grammar[day].append(analysis["grammar_score"])
            daily_processing[day].append(analysis["processing_time"])
        
        # Calcular promedios diarios
        quality_trend = {
            str(day): sum(scores) / len(scores)
            for day, scores in daily_quality.items()
        }
        
        grammar_trend = {
            str(day): sum(scores) / len(scores)
            for day, scores in daily_grammar.items()
        }
        
        processing_trend = {
            str(day): sum(times) / len(times)
            for day, times in daily_processing.items()
        }
        
        # Calcular dirección de tendencia
        if quality_trend:
            quality_values = list(quality_trend.values())
            quality_direction = "up" if quality_values[-1] > quality_values[0] else "down"
        else:
            quality_direction = "stable"
        
        return {
            "quality": {
                "daily": quality_trend,
                "direction": quality_direction
            },
            "grammar": {
                "daily": grammar_trend,
                "direction": "up" if grammar_trend and list(grammar_trend.values())[-1] > list(grammar_trend.values())[0] else "down"
            },
            "processing_time": {
                "daily": processing_trend,
                "direction": "down" if processing_trend and list(processing_trend.values())[-1] < list(processing_trend.values())[0] else "up"
            }
        }
    
    def _calculate_top_categories(
        self,
        analyses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calcular categorías más comunes."""
        # Simulado - en producción usar clasificaciones reales
        return [
            {"category": "report", "count": len(analyses) // 2, "percentage": 50.0},
            {"category": "article", "count": len(analyses) // 4, "percentage": 25.0},
            {"category": "other", "count": len(analyses) // 4, "percentage": 25.0}
        ]
    
    def _generate_additional_metrics(
        self,
        analyses: List[Dict[str, Any]]
    ) -> List[DocumentMetric]:
        """Generar métricas adicionales."""
        metrics = []
        
        # Métricas básicas
        total_analyses = len(analyses)
        documents_with_summary = sum(1 for a in analyses if a["has_summary"])
        documents_with_classification = sum(1 for a in analyses if a["has_classification"])
        avg_keywords = sum(a["keyword_count"] for a in analyses) / total_analyses if total_analyses > 0 else 0
        
        metrics.append(DocumentMetric(
            metric_name="documents_with_summary_percentage",
            value=(documents_with_summary / total_analyses * 100) if total_analyses > 0 else 0,
            unit="%"
        ))
        
        metrics.append(DocumentMetric(
            metric_name="documents_with_classification_percentage",
            value=(documents_with_classification / total_analyses * 100) if total_analyses > 0 else 0,
            unit="%"
        ))
        
        metrics.append(DocumentMetric(
            metric_name="average_keywords_per_document",
            value=avg_keywords,
            unit="keywords"
        ))
        
        return metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas generales."""
        if not self.analysis_history:
            return {
                "total_analyses": 0,
                "total_documents": 0
            }
        
        return {
            "total_analyses": len(self.analysis_history),
            "total_documents": len(set(a["document_id"] for a in self.analysis_history)),
            "average_processing_time": sum(a["processing_time"] for a in self.analysis_history) / len(self.analysis_history),
            "date_range": {
                "first": min(a["timestamp"] for a in self.analysis_history).isoformat(),
                "last": max(a["timestamp"] for a in self.analysis_history).isoformat()
            }
        }


__all__ = [
    "MetricsCollector",
    "MetricsDashboard",
    "DocumentMetric"
]
















