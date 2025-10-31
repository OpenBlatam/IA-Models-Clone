"""
Dashboard Web Server for AI History Comparison System
Servidor web del dashboard para el sistema de análisis de historial de IA
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardServer:
    """
    Servidor del dashboard web para visualización de datos
    """
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8001",
        websocket_url: str = "ws://localhost:8765",
        port: int = 8002,
        host: str = "0.0.0.0"
    ):
        self.api_base_url = api_base_url
        self.websocket_url = websocket_url
        self.port = port
        self.host = host
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="AI History Comparison Dashboard",
            description="Dashboard web para visualización de análisis de historial de IA",
            version="1.0.0"
        )
        
        # Setup static files and templates
        self.templates = Jinja2Templates(directory="templates")
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        # HTTP client for API calls
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    def _setup_routes(self):
        """Configurar rutas del dashboard"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Página principal del dashboard"""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "api_base_url": self.api_base_url,
                "websocket_url": self.websocket_url
            })
        
        @self.app.get("/api/dashboard/overview")
        async def get_dashboard_overview():
            """Obtener resumen general del dashboard"""
            try:
                # Obtener estadísticas generales
                stats_response = await self.http_client.get(f"{self.api_base_url}/statistics")
                stats = stats_response.json() if stats_response.status_code == 200 else {}
                
                # Obtener insights recientes
                insights_response = await self.http_client.get(f"{self.api_base_url}/insights")
                insights = insights_response.json() if insights_response.status_code == 200 else []
                
                # Obtener recomendaciones
                recommendations_response = await self.http_client.get(f"{self.api_base_url}/recommendations")
                recommendations = recommendations_response.json() if recommendations_response.status_code == 200 else []
                
                # Obtener estado en tiempo real
                realtime_response = await self.http_client.get(f"{self.api_base_url}/realtime/status")
                realtime_status = realtime_response.json() if realtime_response.status_code == 200 else {}
                
                return {
                    "statistics": stats,
                    "insights": insights[:5],  # Últimos 5 insights
                    "recommendations": recommendations[:5],  # Últimas 5 recomendaciones
                    "realtime_status": realtime_status,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting dashboard overview: {e}")
                return {"error": str(e)}
        
        @self.app.get("/api/dashboard/performance")
        async def get_performance_data(days: int = 7):
            """Obtener datos de rendimiento para gráficos"""
            try:
                response = await self.http_client.get(
                    f"{self.api_base_url}/analytics/performance-over-time?days={days}"
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": "Failed to fetch performance data"}
                    
            except Exception as e:
                logger.error(f"Error getting performance data: {e}")
                return {"error": str(e)}
        
        @self.app.get("/api/dashboard/query-analysis")
        async def get_query_analysis():
            """Obtener análisis de queries"""
            try:
                response = await self.http_client.get(f"{self.api_base_url}/analytics/query-analysis")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": "Failed to fetch query analysis"}
                    
            except Exception as e:
                logger.error(f"Error getting query analysis: {e}")
                return {"error": str(e)}
        
        @self.app.get("/api/dashboard/ml-performance")
        async def get_ml_performance():
            """Obtener rendimiento de modelos ML"""
            try:
                response = await self.http_client.get(f"{self.api_base_url}/ml/performance")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": "Failed to fetch ML performance"}
                    
            except Exception as e:
                logger.error(f"Error getting ML performance: {e}")
                return {"error": str(e)}
        
        @self.app.get("/api/dashboard/realtime-metrics")
        async def get_realtime_metrics():
            """Obtener métricas en tiempo real"""
            try:
                response = await self.http_client.get(f"{self.api_base_url}/realtime/metrics")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": "Failed to fetch realtime metrics"}
                    
            except Exception as e:
                logger.error(f"Error getting realtime metrics: {e}")
                return {"error": str(e)}
        
        @self.app.get("/api/dashboard/alerts")
        async def get_alerts():
            """Obtener alertas activas"""
            try:
                response = await self.http_client.get(f"{self.api_base_url}/realtime/alerts")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": "Failed to fetch alerts"}
                    
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                return {"error": str(e)}
        
        @self.app.get("/api/dashboard/trends")
        async def get_trends():
            """Obtener análisis de tendencias"""
            try:
                response = await self.http_client.get(f"{self.api_base_url}/realtime/trends")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": "Failed to fetch trends"}
                    
            except Exception as e:
                logger.error(f"Error getting trends: {e}")
                return {"error": str(e)}
        
        @self.app.post("/api/dashboard/optimize-ml")
        async def optimize_ml():
            """Iniciar optimización de ML"""
            try:
                response = await self.http_client.post(f"{self.api_base_url}/ml/optimize")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": "Failed to start ML optimization"}
                    
            except Exception as e:
                logger.error(f"Error starting ML optimization: {e}")
                return {"error": str(e)}
        
        @self.app.post("/api/dashboard/resolve-alert/{alert_id}")
        async def resolve_alert(alert_id: str):
            """Resolver una alerta"""
            try:
                response = await self.http_client.post(f"{self.api_base_url}/realtime/alerts/{alert_id}/resolve")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": "Failed to resolve alert"}
                    
            except Exception as e:
                logger.error(f"Error resolving alert: {e}")
                return {"error": str(e)}
        
        @self.app.websocket("/ws/dashboard")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket para actualizaciones en tiempo real del dashboard"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Enviar actualizaciones periódicas
                    await asyncio.sleep(5)  # Actualizar cada 5 segundos
                    
                    # Obtener datos actualizados
                    try:
                        overview_data = await self.get_dashboard_overview()
                        await websocket.send_json({
                            "type": "dashboard_update",
                            "data": overview_data,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Error sending dashboard update: {e}")
                        
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
    
    async def broadcast_to_dashboard(self, message: Dict[str, Any]):
        """Transmitir mensaje a todos los clientes del dashboard"""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected_connections = []
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending message to dashboard client: {e}")
                disconnected_connections.append(websocket)
        
        # Limpiar conexiones desconectadas
        for websocket in disconnected_connections:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def get_dashboard_overview(self) -> Dict[str, Any]:
        """Obtener resumen del dashboard (método interno)"""
        try:
            # Obtener estadísticas generales
            stats_response = await self.http_client.get(f"{self.api_base_url}/statistics")
            stats = stats_response.json() if stats_response.status_code == 200 else {}
            
            # Obtener estado en tiempo real
            realtime_response = await self.http_client.get(f"{self.api_base_url}/realtime/status")
            realtime_status = realtime_response.json() if realtime_response.status_code == 200 else {}
            
            return {
                "statistics": stats,
                "realtime_status": realtime_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            return {"error": str(e)}
    
    def run(self):
        """Ejecutar el servidor del dashboard"""
        logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

if __name__ == "__main__":
    dashboard = DashboardServer()
    dashboard.run()



























