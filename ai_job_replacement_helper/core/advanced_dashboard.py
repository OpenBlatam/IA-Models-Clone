"""
Advanced Dashboard Service - Dashboard avanzado con analytics
==============================================================

Sistema de dashboard interactivo con métricas avanzadas y visualizaciones.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DashboardWidget:
    """Widget del dashboard"""
    id: str
    widget_type: str  # "metric", "chart", "table", "list"
    title: str
    data: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: Optional[int] = None  # segundos


@dataclass
class Dashboard:
    """Dashboard completo"""
    id: str
    user_id: str
    name: str
    widgets: List[DashboardWidget]
    layout: str = "grid"  # grid, custom
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class DashboardMetrics:
    """Métricas del dashboard"""
    total_applications: int
    applications_this_week: int
    interview_rate: float
    offer_rate: float
    average_response_time: float  # días
    skills_learned: int
    assessments_completed: int
    network_growth: int
    portfolio_views: int
    job_alerts_active: int
    streak_days: int
    current_level: int
    total_points: int


class AdvancedDashboardService:
    """Servicio de dashboard avanzado"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.dashboards: Dict[str, Dashboard] = {}
        logger.info("AdvancedDashboardService initialized")
    
    def create_dashboard(
        self,
        user_id: str,
        name: str,
        layout: str = "grid"
    ) -> Dashboard:
        """Crear nuevo dashboard"""
        dashboard_id = f"dashboard_{user_id}_{int(datetime.now().timestamp())}"
        
        dashboard = Dashboard(
            id=dashboard_id,
            user_id=user_id,
            name=name,
            widgets=[],
            layout=layout,
        )
        
        self.dashboards[dashboard_id] = dashboard
        
        logger.info(f"Dashboard created: {dashboard_id}")
        return dashboard
    
    def add_widget(
        self,
        dashboard_id: str,
        widget_type: str,
        title: str,
        data: Dict[str, Any],
        position: Optional[Dict[str, int]] = None
    ) -> DashboardWidget:
        """Agregar widget al dashboard"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        widget_id = f"widget_{len(dashboard.widgets)}"
        
        if not position:
            position = {
                "x": len(dashboard.widgets) % 3,
                "y": len(dashboard.widgets) // 3,
                "width": 1,
                "height": 1,
            }
        
        widget = DashboardWidget(
            id=widget_id,
            widget_type=widget_type,
            title=title,
            data=data,
            position=position,
        )
        
        dashboard.widgets.append(widget)
        dashboard.updated_at = datetime.now()
        
        return widget
    
    def get_user_metrics(self, user_id: str) -> DashboardMetrics:
        """Obtener métricas del usuario"""
        # En producción, esto consultaría datos reales de otros servicios
        # Por ahora, simulamos
        
        return DashboardMetrics(
            total_applications=25,
            applications_this_week=5,
            interview_rate=0.2,  # 20%
            offer_rate=0.08,  # 8%
            average_response_time=3.5,  # días
            skills_learned=12,
            assessments_completed=8,
            network_growth=15,  # nuevos contactos este mes
            portfolio_views=120,
            job_alerts_active=3,
            streak_days=7,
            current_level=5,
            total_points=2500,
        )
    
    def generate_insights(self, user_id: str) -> List[Dict[str, Any]]:
        """Generar insights del usuario"""
        metrics = self.get_user_metrics(user_id)
        
        insights = []
        
        # Insight de aplicaciones
        if metrics.applications_this_week > 0:
            insights.append({
                "type": "success",
                "title": "Actividad de aplicaciones",
                "message": f"Has aplicado a {metrics.applications_this_week} trabajos esta semana. ¡Sigue así!",
                "action": "Ver aplicaciones",
            })
        
        # Insight de entrevistas
        if metrics.interview_rate > 0.15:
            insights.append({
                "type": "info",
                "title": "Tasa de entrevistas alta",
                "message": f"Tu tasa de entrevistas es {metrics.interview_rate*100:.0f}%. Excelente!",
                "action": "Preparar para entrevistas",
            })
        elif metrics.interview_rate < 0.1:
            insights.append({
                "type": "warning",
                "title": "Mejora tu tasa de entrevistas",
                "message": "Considera optimizar tu CV y personalizar tus aplicaciones",
                "action": "Optimizar CV",
            })
        
        # Insight de skills
        if metrics.skills_learned > 0:
            insights.append({
                "type": "success",
                "title": "Aprendizaje continuo",
                "message": f"Has aprendido {metrics.skills_learned} nuevas habilidades. ¡Excelente progreso!",
                "action": "Ver habilidades",
            })
        
        # Insight de streak
        if metrics.streak_days >= 7:
            insights.append({
                "type": "success",
                "title": "Racha activa",
                "message": f"Llevas {metrics.streak_days} días consecutivos activo. ¡Mantén el ritmo!",
                "action": "Ver logros",
            })
        
        return insights
    
    def get_trend_data(
        self,
        user_id: str,
        metric: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Obtener datos de tendencia"""
        # En producción, esto consultaría datos históricos reales
        # Por ahora, generamos datos simulados
        
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, 0, -1)]
        
        trend_data = {
            "metric": metric,
            "dates": dates,
            "values": [10 + i * 0.5 + (i % 3) * 2 for i in range(days)],
            "trend": "increasing",  # increasing, decreasing, stable
            "growth_rate": 5.2,  # porcentaje
        }
        
        return trend_data
    
    def get_comparison_data(
        self,
        user_id: str,
        metric: str
    ) -> Dict[str, Any]:
        """Obtener datos de comparación con otros usuarios"""
        user_metrics = self.get_user_metrics(user_id)
        
        # En producción, esto compararía con datos agregados de otros usuarios
        comparison = {
            "metric": metric,
            "user_value": getattr(user_metrics, metric, 0),
            "average_value": getattr(user_metrics, metric, 0) * 0.9,  # Simulado
            "percentile": 75,  # Simulado
            "comparison": "above_average",  # above_average, average, below_average
        }
        
        return comparison




