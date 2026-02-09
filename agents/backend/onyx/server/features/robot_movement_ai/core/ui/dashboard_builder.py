"""
Dashboard Builder System
=========================

Sistema de construcción de dashboards.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class WidgetType(Enum):
    """Tipo de widget."""
    METRIC = "metric"
    CHART = "chart"
    TABLE = "table"
    TEXT = "text"
    IMAGE = "image"


@dataclass
class DashboardWidget:
    """Widget de dashboard."""
    widget_id: str
    widget_type: WidgetType
    title: str
    data_source: str
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    size: Dict[str, int] = field(default_factory=lambda: {"width": 4, "height": 3})
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard."""
    dashboard_id: str
    name: str
    description: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout: str = "grid"  # grid, freeform
    refresh_interval: Optional[int] = None  # segundos


class DashboardBuilder:
    """
    Constructor de dashboards.
    
    Construye dashboards personalizables.
    """
    
    def __init__(self):
        """Inicializar constructor."""
        self.dashboards: Dict[str, Dashboard] = {}
    
    def create_dashboard(
        self,
        dashboard_id: str,
        name: str,
        description: str = "",
        layout: str = "grid",
        refresh_interval: Optional[int] = None
    ) -> Dashboard:
        """
        Crear nuevo dashboard.
        
        Args:
            dashboard_id: ID único del dashboard
            name: Nombre del dashboard
            description: Descripción
            layout: Layout (grid, freeform)
            refresh_interval: Intervalo de actualización
            
        Returns:
            Dashboard creado
        """
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            layout=layout,
            refresh_interval=refresh_interval
        )
        
        self.dashboards[dashboard_id] = dashboard
        logger.info(f"Created dashboard: {name} ({dashboard_id})")
        
        return dashboard
    
    def add_widget(
        self,
        dashboard: Dashboard,
        widget_id: str,
        widget_type: WidgetType,
        title: str,
        data_source: str,
        position: Optional[Dict[str, int]] = None,
        size: Optional[Dict[str, int]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> DashboardWidget:
        """
        Agregar widget al dashboard.
        
        Args:
            dashboard: Dashboard
            widget_id: ID único del widget
            widget_type: Tipo de widget
            title: Título
            data_source: Fuente de datos
            position: Posición (x, y)
            size: Tamaño (width, height)
            config: Configuración adicional
            
        Returns:
            Widget creado
        """
        widget = DashboardWidget(
            widget_id=widget_id,
            widget_type=widget_type,
            title=title,
            data_source=data_source,
            position=position or {"x": 0, "y": 0},
            size=size or {"width": 4, "height": 3},
            config=config or {}
        )
        
        dashboard.widgets.append(widget)
        return widget
    
    def generate_dashboard_config(self, dashboard: Dashboard) -> Dict[str, Any]:
        """
        Generar configuración de dashboard.
        
        Args:
            dashboard: Dashboard
            
        Returns:
            Configuración del dashboard
        """
        return {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "layout": dashboard.layout,
            "refresh_interval": dashboard.refresh_interval,
            "widgets": [
                {
                    "widget_id": w.widget_id,
                    "type": w.widget_type.value,
                    "title": w.title,
                    "data_source": w.data_source,
                    "position": w.position,
                    "size": w.size,
                    "config": w.config
                }
                for w in dashboard.widgets
            ]
        }
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Obtener dashboard por ID."""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dashboard]:
        """Listar todos los dashboards."""
        return list(self.dashboards.values())


# Instancia global
_dashboard_builder: Optional[DashboardBuilder] = None


def get_dashboard_builder() -> DashboardBuilder:
    """Obtener instancia global del constructor de dashboards."""
    global _dashboard_builder
    if _dashboard_builder is None:
        _dashboard_builder = DashboardBuilder()
    return _dashboard_builder






