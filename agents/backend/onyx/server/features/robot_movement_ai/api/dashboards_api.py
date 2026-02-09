"""
Dashboards API Endpoints
========================

Endpoints para dashboards.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..core.dashboard_builder import get_dashboard_builder, WidgetType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dashboards", tags=["dashboards"])


@router.post("/create")
async def create_dashboard(
    dashboard_id: str,
    name: str,
    description: str = "",
    layout: str = Query("grid", regex="^(grid|freeform)$"),
    refresh_interval: Optional[int] = None
) -> Dict[str, Any]:
    """Crear nuevo dashboard."""
    try:
        builder = get_dashboard_builder()
        dashboard = builder.create_dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            layout=layout,
            refresh_interval=refresh_interval
        )
        return {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description
        }
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dashboard_id}/widgets")
async def add_widget(
    dashboard_id: str,
    widget_id: str,
    widget_type: str,
    title: str,
    data_source: str,
    position: Optional[Dict[str, int]] = None,
    size: Optional[Dict[str, int]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Agregar widget al dashboard."""
    try:
        builder = get_dashboard_builder()
        dashboard = builder.get_dashboard(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        try:
            widget_type_enum = WidgetType(widget_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid widget type: {widget_type}"
            )
        
        widget = builder.add_widget(
            dashboard,
            widget_id=widget_id,
            widget_type=widget_type_enum,
            title=title,
            data_source=data_source,
            position=position,
            size=size,
            config=config
        )
        
        return {
            "widget_id": widget.widget_id,
            "type": widget.widget_type.value,
            "title": widget.title
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding widget: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dashboard_id}")
async def get_dashboard(dashboard_id: str) -> Dict[str, Any]:
    """Obtener configuración de dashboard."""
    try:
        builder = get_dashboard_builder()
        dashboard = builder.get_dashboard(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        config = builder.generate_dashboard_config(dashboard)
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_dashboards() -> Dict[str, Any]:
    """Listar todos los dashboards."""
    try:
        builder = get_dashboard_builder()
        dashboards = builder.list_dashboards()
        return {
            "dashboards": [
                {
                    "dashboard_id": d.dashboard_id,
                    "name": d.name,
                    "description": d.description,
                    "widgets_count": len(d.widgets)
                }
                for d in dashboards
            ],
            "count": len(dashboards)
        }
    except Exception as e:
        logger.error(f"Error listing dashboards: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






