"""
ROI Analyzer - Sistema de análisis de ROI (Return on Investment)
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ROIMetric:
    """Métrica de ROI"""
    metric_name: str
    value: float
    cost: float
    revenue: float
    roi: float
    timestamp: datetime


class ROIAnalyzer:
    """Analizador de ROI"""

    def __init__(self):
        """Inicializar analizador"""
        self.roi_history: List[ROIMetric] = []
        self.content_investments: Dict[str, Dict[str, Any]] = {}

    def record_investment(
        self,
        content_id: str,
        cost: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar inversión en contenido.

        Args:
            content_id: ID del contenido
            cost: Costo de producción
            metadata: Metadatos adicionales
        """
        self.content_investments[content_id] = {
            "cost": cost,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.debug(f"Inversión registrada para contenido {content_id}: ${cost}")

    def record_revenue(
        self,
        content_id: str,
        revenue: float,
        metric_name: str = "revenue"
    ):
        """
        Registrar ingresos de contenido.

        Args:
            content_id: ID del contenido
            revenue: Ingresos generados
            metric_name: Nombre de la métrica
        """
        if content_id not in self.content_investments:
            logger.warning(f"Contenido {content_id} no tiene inversión registrada")
            return
        
        investment = self.content_investments[content_id]
        cost = investment["cost"]
        
        roi = ((revenue - cost) / cost * 100) if cost > 0 else 0
        
        metric = ROIMetric(
            metric_name=metric_name,
            value=revenue,
            cost=cost,
            revenue=revenue,
            roi=roi,
            timestamp=datetime.utcnow()
        )
        
        self.roi_history.append(metric)
        logger.debug(f"ROI calculado para {content_id}: {roi}%")

    def calculate_roi(
        self,
        content_id: str
    ) -> Dict[str, Any]:
        """
        Calcular ROI de un contenido.

        Args:
            content_id: ID del contenido

        Returns:
            Análisis de ROI
        """
        if content_id not in self.content_investments:
            return {"error": "Contenido no encontrado"}
        
        investment = self.content_investments[content_id]
        cost = investment["cost"]
        
        # Buscar métricas de ROI para este contenido
        content_metrics = [
            m for m in self.roi_history
            if m.metric_name.startswith(content_id)
        ]
        
        if not content_metrics:
            return {
                "content_id": content_id,
                "cost": cost,
                "revenue": 0,
                "roi": -100,  # Pérdida total
                "status": "no_revenue",
                "message": "No hay ingresos registrados"
            }
        
        total_revenue = sum(m.revenue for m in content_metrics)
        total_roi = ((total_revenue - cost) / cost * 100) if cost > 0 else 0
        
        return {
            "content_id": content_id,
            "cost": cost,
            "revenue": total_revenue,
            "roi": total_roi,
            "profit": total_revenue - cost,
            "status": "positive" if total_roi > 0 else "negative",
            "metrics_count": len(content_metrics)
        }

    def analyze_portfolio_roi(self) -> Dict[str, Any]:
        """
        Analizar ROI del portafolio completo.

        Returns:
            Análisis de ROI del portafolio
        """
        if not self.content_investments:
            return {"error": "No hay inversiones registradas"}
        
        total_cost = sum(inv["cost"] for inv in self.content_investments.values())
        total_revenue = sum(m.revenue for m in self.roi_history)
        total_roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        # ROI por contenido
        content_rois = []
        for content_id in self.content_investments:
            roi_data = self.calculate_roi(content_id)
            if "error" not in roi_data:
                content_rois.append(roi_data)
        
        # Ordenar por ROI
        content_rois.sort(key=lambda x: x.get("roi", 0), reverse=True)
        
        return {
            "total_cost": total_cost,
            "total_revenue": total_revenue,
            "total_roi": total_roi,
            "total_profit": total_revenue - total_cost,
            "content_count": len(self.content_investments),
            "top_performers": content_rois[:5],
            "worst_performers": content_rois[-5:] if len(content_rois) >= 5 else content_rois
        }

    def get_roi_recommendations(self) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones basadas en ROI.

        Returns:
            Recomendaciones
        """
        recommendations = []
        
        portfolio = self.analyze_portfolio_roi()
        if "error" in portfolio:
            return recommendations
        
        total_roi = portfolio.get("total_roi", 0)
        
        if total_roi < 0:
            recommendations.append({
                "priority": "high",
                "category": "roi",
                "title": "ROI negativo del portafolio",
                "description": f"El ROI total es {total_roi:.2f}%",
                "suggestion": "Revisa los contenidos con peor desempeño y considera optimizarlos o eliminarlos"
            })
        
        worst_performers = portfolio.get("worst_performers", [])
        if worst_performers:
            worst = worst_performers[0]
            if worst.get("roi", 0) < -50:
                recommendations.append({
                    "priority": "high",
                    "category": "content",
                    "title": f"Contenido con ROI muy negativo: {worst.get('content_id')}",
                    "description": f"ROI: {worst.get('roi', 0):.2f}%",
                    "suggestion": "Considera eliminar o reescribir este contenido"
                })
        
        top_performers = portfolio.get("top_performers", [])
        if top_performers:
            best = top_performers[0]
            if best.get("roi", 0) > 100:
                recommendations.append({
                    "priority": "medium",
                    "category": "optimization",
                    "title": f"Contenido de alto rendimiento: {best.get('content_id')}",
                    "description": f"ROI: {best.get('roi', 0):.2f}%",
                    "suggestion": "Crea más contenido similar a este"
                })
        
        return recommendations






