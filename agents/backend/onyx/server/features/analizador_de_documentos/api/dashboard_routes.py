"""
Rutas para Dashboard
====================

Endpoints para dashboard web.
"""

import logging
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from ..utils.metrics import get_performance_monitor
from ..utils.dashboard_generator import DashboardGenerator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/dashboard",
    tags=["Dashboard"]
)


@router.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Obtener dashboard HTML"""
    try:
        monitor = get_performance_monitor()
        if monitor:
            stats = monitor.get_all_stats()
            
            dashboard_html = DashboardGenerator.generate_dashboard(
                stats,
                title="Analizador de Documentos - Dashboard"
            )
            
            return HTMLResponse(content=dashboard_html)
        else:
            return HTMLResponse(content="<h1>Dashboard no disponible</h1>")
    except Exception as e:
        logger.error(f"Error generando dashboard: {e}")
        return HTMLResponse(content=f"<h1>Error: {e}</h1>")
















