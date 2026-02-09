"""
🌐 DASHBOARD WEB v5.0
======================

Dashboard web moderno para el sistema v5.0 que incluye:
- Interfaz de optimización de contenido
- Monitoreo de salud del sistema
- Controles de AI y microservicios
- Analytics en tiempo real
- Gestión de seguridad empresarial
- Control de infraestructura cloud-native
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import v5.0 modules with fallback
try:
    from integrated_system_v5 import IntegratedSystemV5, OptimizationMode
    INTEGRATED_SYSTEM_AVAILABLE = True
    logger.info("✅ Integrated System v5.0 loaded")
except ImportError as e:
    INTEGRATED_SYSTEM_AVAILABLE = False
    logger.warning(f"⚠️ Integrated System not available: {e}")

# Pydantic models
class OptimizationRequest(BaseModel):
    content: str
    mode: str = "auto"

class OptimizationResponse(BaseModel):
    success: bool
    content_id: str
    optimized_content: str
    optimization_score: float
    processing_time: float
    mode_used: str
    message: str

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔌 WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send message: {e}")

# Dashboard configuration
class DashboardConfig:
    def __init__(self):
        self.app = FastAPI(
            title="LinkedIn Optimizer v5.0 Dashboard",
            description="Dashboard web para el sistema integrado v5.0",
            version="5.0.0"
        )
        self.manager = ConnectionManager()
        self.integrated_system = None
        self.setup_routes()
        
        logger.info("🌐 Dashboard Web v5.0 initialized")
    
    def setup_routes(self):
        """Setup API routes and WebSocket endpoints."""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "version": "5.0.0", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            return self._get_dashboard_html()
        
        @self.app.post("/api/optimize", response_model=OptimizationResponse)
        async def optimize_content(request: OptimizationRequest):
            return await self._optimize_content(request)
        
        @self.app.get("/api/status")
        async def get_system_status():
            return await self._get_system_status()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                while True:
                    await asyncio.sleep(30)
                    if self.integrated_system:
                        status = await self.integrated_system.get_system_status()
                        await websocket.send_text(json.dumps({
                            "type": "status_update",
                            "data": status,
                            "timestamp": datetime.now().isoformat()
                        }))
            except WebSocketDisconnect:
                self.manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.manager.disconnect(websocket)
    
    async def _optimize_content(self, request: OptimizationRequest) -> OptimizationResponse:
        """Optimize content using the integrated system."""
        if not self.integrated_system:
            raise HTTPException(status_code=503, detail="Integrated system not available")
        
        try:
            # Determine optimization mode
            if request.mode == "auto":
                target_mode = None
            else:
                mode_mapping = {
                    "basic": OptimizationMode.BASIC,
                    "advanced": OptimizationMode.ADVANCED,
                    "enterprise": OptimizationMode.ENTERPRISE,
                    "quantum": OptimizationMode.QUANTUM
                }
                target_mode = mode_mapping.get(request.mode.lower(), None)
            
            # Perform optimization
            result = await self.integrated_system.optimize_content(
                content=request.content,
                target_mode=target_mode
            )
            
            # Broadcast result
            await self.manager.broadcast(json.dumps({
                "type": "optimization_completed",
                "data": {
                    "content_id": result.content_id,
                    "score": result.optimization_score,
                    "mode": result.mode.name
                }
            }))
            
            return OptimizationResponse(
                success=True,
                content_id=result.content_id,
                optimized_content=result.optimized_content,
                optimization_score=result.optimization_score,
                processing_time=result.processing_time,
                mode_used=result.mode.name,
                message="Content optimized successfully"
            )
            
        except Exception as e:
            logger.error(f"Content optimization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        if not self.integrated_system:
            raise HTTPException(status_code=503, detail="Integrated system not available")
        
        try:
            return await self.integrated_system.get_system_status()
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")
    
    def _get_dashboard_html(self) -> str:
        """Generate the dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LinkedIn Optimizer v5.0 Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <div class="text-3xl">🚀</div>
                    <div>
                        <h1 class="text-2xl font-bold">LinkedIn Optimizer v5.0</h1>
                        <p class="text-blue-100">Dashboard del Sistema Integrado</p>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <div id="system-status" class="px-3 py-1 bg-green-500 rounded-full text-sm font-medium">
                        Conectado
                    </div>
                    <div id="current-mode" class="px-3 py-1 bg-blue-500 rounded-full text-sm font-medium">
                        Cargando...
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto px-6 py-8">
        <!-- Quick Stats -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <div class="flex items-center">
                    <div class="p-2 bg-blue-100 rounded-lg">
                        <div class="text-2xl">🧠</div>
                    </div>
                    <div class="ml-4">
                        <p class="text-sm font-medium text-gray-600">Optimizaciones</p>
                        <p id="total-optimizations" class="text-2xl font-bold text-gray-900">-</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6">
                <div class="flex items-center">
                    <div class="p-2 bg-green-100 rounded-lg">
                        <div class="text-2xl">⚡</div>
                    </div>
                    <div class="ml-4">
                        <p class="text-sm font-medium text-gray-600">Tiempo Promedio</p>
                        <p id="avg-processing-time" class="text-2xl font-bold text-gray-900">-</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6">
                <div class="flex items-center">
                    <div class="p-2 bg-purple-100 rounded-lg">
                        <div class="text-2xl">🛡️</div>
                    </div>
                    <div class="ml-4">
                        <p class="text-sm font-medium text-gray-600">Salud del Sistema</p>
                        <p id="health-score" class="text-2xl font-bold text-gray-900">-</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6">
                <div class="flex items-center">
                    <div class="p-2 bg-orange-100 rounded-lg">
                        <div class="text-2xl">⏱️</div>
                    </div>
                    <div class="ml-4">
                        <p class="text-sm font-medium text-gray-600">Uptime</p>
                        <p id="uptime" class="text-2xl font-bold text-gray-900">-</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Content Optimization -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <!-- Optimization Form -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">🔧 Optimización de Contenido</h2>
                <form id="optimization-form" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Contenido a Optimizar</label>
                        <textarea 
                            id="content-input" 
                            rows="4" 
                            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="Escribe tu contenido de LinkedIn aquí..."
                        ></textarea>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Modo</label>
                        <select id="mode-select" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="auto">Auto (Recomendado)</option>
                            <option value="basic">Básico</option>
                            <option value="advanced">Avanzado</option>
                            <option value="enterprise">Empresarial</option>
                            <option value="quantum">Cuántico</option>
                        </select>
                    </div>
                    
                    <button 
                        type="submit" 
                        class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
                    >
                        🚀 Optimizar Contenido
                    </button>
                </form>
            </div>

            <!-- Optimization Result -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">✨ Resultado de Optimización</h2>
                <div id="optimization-result" class="space-y-4">
                    <div class="text-center text-gray-500 py-8">
                        <div class="text-4xl mb-2">📝</div>
                        <p>Ingresa contenido y haz clic en "Optimizar" para ver los resultados</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- System Status -->
        <div class="bg-white rounded-lg shadow-md p-6">
            <h2 class="text-xl font-bold text-gray-900 mb-4">📊 Estado del Sistema</h2>
            <div id="system-status-details" class="space-y-3">
                <div class="flex justify-between items-center">
                    <span class="text-sm text-gray-600">Estado General:</span>
                    <span id="overall-status" class="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">-</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-sm text-gray-600">Sistemas Activos:</span>
                    <span id="active-systems" class="text-sm font-medium">-</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-sm text-gray-600">Versión:</span>
                    <span id="system-version" class="text-sm font-medium">-</span>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-800 text-white py-6 mt-12">
        <div class="container mx-auto px-6 text-center">
            <p>&copy; 2024 LinkedIn Optimizer v5.0. Sistema Integrado de Próxima Generación.</p>
        </div>
    </footer>

    <script>
        // WebSocket connection
        let ws = null;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeWebSocket();
            setupEventListeners();
            loadSystemStatus();
        });

        function initializeWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                document.getElementById('system-status').textContent = 'Conectado';
                document.getElementById('system-status').className = 'px-3 py-1 bg-green-500 rounded-full text-sm font-medium';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                document.getElementById('system-status').textContent = 'Desconectado';
                document.getElementById('system-status').className = 'px-3 py-1 bg-red-500 rounded-full text-sm font-medium';
                setTimeout(initializeWebSocket, 5000);
            };
        }

        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'status_update':
                    updateDashboardStats(data.data);
                    break;
                case 'optimization_completed':
                    showOptimizationNotification(data.data);
                    break;
            }
        }

        function setupEventListeners() {
            document.getElementById('optimization-form').addEventListener('submit', handleOptimizationSubmit);
        }

        async function handleOptimizationSubmit(event) {
            event.preventDefault();
            
            const content = document.getElementById('content-input').value.trim();
            const mode = document.getElementById('mode-select').value;
            
            if (!content) {
                alert('Por favor ingresa contenido para optimizar');
                return;
            }
            
            const submitButton = event.target.querySelector('button[type="submit"]');
            const originalText = submitButton.textContent;
            submitButton.textContent = '⏳ Optimizando...';
            submitButton.disabled = true;
            
            try {
                const response = await fetch('/api/optimize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        content: content,
                        mode: mode
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayOptimizationResult(result);
                    loadSystemStatus();
                } else {
                    alert('Error en la optimización: ' + result.message);
                }
                
            } catch (error) {
                console.error('Optimization error:', error);
                alert('Error al conectar con el servidor');
            } finally {
                submitButton.textContent = originalText;
                submitButton.disabled = false;
            }
        }

        function displayOptimizationResult(result) {
            const resultDiv = document.getElementById('optimization-result');
            
            resultDiv.innerHTML = `
                <div class="space-y-4">
                    <div class="flex items-center justify-between">
                        <span class="text-sm text-gray-600">Score de Optimización:</span>
                        <span class="px-2 py-1 bg-green-100 text-green-800 rounded text-sm font-medium">
                            ${(result.optimization_score * 100).toFixed(1)}%
                        </span>
                    </div>
                    
                    <div class="flex items-center justify-between">
                        <span class="text-sm text-gray-600">Tiempo de Procesamiento:</span>
                        <span class="text-sm font-medium">${result.processing_time.toFixed(3)}s</span>
                    </div>
                    
                    <div class="flex items-center justify-between">
                        <span class="text-sm text-gray-600">Modo Utilizado:</span>
                        <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm font-medium">
                            ${result.mode_used}
                        </span>
                    </div>
                    
                    <div class="border-t pt-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Contenido Optimizado:</label>
                        <div class="bg-gray-50 p-3 rounded-md text-sm">
                            ${result.optimized_content}
                        </div>
                    </div>
                </div>
            `;
        }

        async function loadSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                updateDashboardStats(status);
            } catch (error) {
                console.error('Failed to load system status:', error);
            }
        }

        function updateDashboardStats(status) {
            // Update quick stats
            document.getElementById('total-optimizations').textContent = status.performance.total_optimizations;
            document.getElementById('avg-processing-time').textContent = status.performance.average_processing_time.toFixed(3) + 's';
            document.getElementById('uptime').textContent = status.performance.uptime;
            
            // Update current mode
            document.getElementById('current-mode').textContent = status.system_info.mode;
            
            // Update system status details
            document.getElementById('overall-status').textContent = status.system_info.status;
            document.getElementById('active-systems').textContent = status.system_info.total_systems;
            document.getElementById('system-version').textContent = 'v' + status.system_info.version;
            
            // Calculate and update health score
            const totalSystems = status.system_info.total_systems;
            const availableSystems = Object.keys(status.systems_status).length;
            const healthScore = totalSystems > 0 ? (availableSystems / totalSystems * 100) : 0;
            
            document.getElementById('health-score').textContent = healthScore.toFixed(1) + '%';
            
            // Update health score color
            const healthElement = document.getElementById('health-score');
            if (healthScore >= 80) {
                healthElement.className = 'text-2xl font-bold text-green-600';
            } else if (healthScore >= 60) {
                healthElement.className = 'text-2xl font-bold text-yellow-600';
            } else {
                healthElement.className = 'text-2xl font-bold text-red-600';
            }
        }

        function showOptimizationNotification(data) {
            const notification = document.createElement('div');
            notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
            notification.innerHTML = `
                <div class="flex items-center space-x-2">
                    <div>✅</div>
                    <div>
                        <div class="font-medium">Optimización Completada</div>
                        <div class="text-sm">Score: ${(data.score * 100).toFixed(1)}%</div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 5000);
        }

        // Auto-refresh system status every 30 seconds
        setInterval(loadSystemStatus, 30000);
    </script>
</body>
</html>
        """

# Main dashboard instance
dashboard = DashboardConfig()

# Initialize integrated system if available
async def initialize_dashboard():
    """Initialize the dashboard with integrated system."""
    if INTEGRATED_SYSTEM_AVAILABLE:
        try:
            dashboard.integrated_system = IntegratedSystemV5()
            await dashboard.integrated_system.start_system()
            logger.info("✅ Integrated System v5.0 started for dashboard")
        except Exception as e:
            logger.error(f"❌ Failed to start integrated system: {e}")
    else:
        logger.warning("⚠️ Dashboard running without integrated system")

# Run dashboard
if __name__ == "__main__":
    import uvicorn
    
    # Initialize dashboard
    asyncio.run(initialize_dashboard())
    
    # Start web server
    uvicorn.run(
        dashboard.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
