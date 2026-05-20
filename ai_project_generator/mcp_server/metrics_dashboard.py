"""
MCP Metrics Dashboard - Dashboard de métricas
==============================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)


class MetricsDashboard:
    """
    Dashboard de métricas para MCP
    
    Proporciona endpoints y vistas para visualizar métricas.
    """
    
    def __init__(self, observability: Optional[Any] = None):
        """
        Args:
            observability: Instancia de MCPObservability
        """
        self.observability = observability
        self.router = APIRouter(prefix="/mcp/v1/metrics", tags=["metrics"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Configura rutas del dashboard"""
        
        @self.router.get("/")
        async def metrics_dashboard() -> HTMLResponse:
            """Dashboard HTML de métricas"""
            html = self._generate_dashboard_html()
            return HTMLResponse(content=html)
        
        @self.router.get("/api/summary")
        async def metrics_summary() -> Dict[str, Any]:
            """Resumen de métricas en JSON"""
            if self.observability:
                return self.observability.get_metrics_summary()
            return {"error": "Observability not available"}
        
        @self.router.get("/api/prometheus")
        async def prometheus_metrics() -> str:
            """Métricas en formato Prometheus"""
            # Implementar exportación Prometheus
            return "# Prometheus metrics\n"
    
    def _generate_dashboard_html(self) -> str:
        """Genera HTML del dashboard"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>MCP Metrics Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }
        .metric-label { color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>MCP Metrics Dashboard</h1>
    <div id="metrics"></div>
    <script>
        async function loadMetrics() {
            const response = await fetch('/mcp/v1/metrics/api/summary');
            const data = await response.json();
            document.getElementById('metrics').innerHTML = 
                '<div class="metric-card">' +
                '<div class="metric-value">' + JSON.stringify(data, null, 2) + '</div>' +
                '</div>';
        }
        loadMetrics();
        setInterval(loadMetrics, 5000);
    </script>
</body>
</html>
        """
    
    def get_router(self) -> APIRouter:
        """Retorna el router del dashboard"""
        return self.router

