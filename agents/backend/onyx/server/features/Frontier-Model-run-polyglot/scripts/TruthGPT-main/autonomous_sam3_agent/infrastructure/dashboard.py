"""
Web Dashboard
=============

Real-time web dashboard for monitoring the autonomous agent.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if TYPE_CHECKING:
    from ..core.agent_core import AutonomousSAM3Agent

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


def create_dashboard_html() -> str:
    """Create the dashboard HTML page."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Agent Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        .status-healthy { background: #00ff88; }
        .status-degraded { background: #ffaa00; }
        .status-unhealthy { background: #ff4444; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .card h2 {
            font-size: 1.2rem;
            margin-bottom: 20px;
            opacity: 0.8;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric:last-child { border-bottom: none; }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #00d9ff;
        }
        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            transition: width 0.3s ease;
        }
        .workers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(40px, 1fr));
            gap: 8px;
        }
        .worker {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .worker-active { background: #00ff88; color: #000; }
        .worker-idle { background: rgba(255, 255, 255, 0.1); }
        .log-container {
            height: 200px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.85rem;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 12px;
        }
        .log-entry { padding: 4px 0; opacity: 0.8; }
        .log-time { color: #00d9ff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Autonomous Agent Dashboard</h1>
        <p style="margin-top: 10px; opacity: 0.6;">
            <span class="status-indicator status-healthy" id="statusIndicator"></span>
            <span id="statusText">Connecting...</span>
        </p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h2>📊 Task Statistics</h2>
            <div class="metric">
                <span>Total Tasks</span>
                <span class="metric-value" id="totalTasks">0</span>
            </div>
            <div class="metric">
                <span>Completed</span>
                <span class="metric-value" id="completedTasks">0</span>
            </div>
            <div class="metric">
                <span>Failed</span>
                <span class="metric-value" id="failedTasks">0</span>
            </div>
            <div class="metric">
                <span>Success Rate</span>
                <span class="metric-value" id="successRate">0%</span>
            </div>
        </div>
        
        <div class="card">
            <h2>⚡ Workers</h2>
            <div class="metric">
                <span>Active Workers</span>
                <span class="metric-value" id="activeWorkers">0</span>
            </div>
            <div class="metric">
                <span>Queue Size</span>
                <span class="metric-value" id="queueSize">0</span>
            </div>
            <div style="margin-top: 16px;">
                <div class="workers-grid" id="workersGrid"></div>
            </div>
        </div>
        
        <div class="card">
            <h2>💾 System Resources</h2>
            <div class="metric">
                <span>Memory</span>
                <span id="memoryPercent">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="memoryBar" style="width: 0%"></div>
            </div>
            <div class="metric" style="margin-top: 16px;">
                <span>CPU</span>
                <span id="cpuPercent">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="cpuBar" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="card" style="grid-column: span 2;">
            <h2>📝 Recent Activity</h2>
            <div class="log-container" id="logContainer"></div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            document.getElementById('statusText').textContent = 'Connected';
            document.getElementById('statusIndicator').className = 'status-indicator status-healthy';
        };
        
        ws.onclose = () => {
            document.getElementById('statusText').textContent = 'Disconnected';
            document.getElementById('statusIndicator').className = 'status-indicator status-unhealthy';
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        function updateDashboard(data) {
            // Update stats
            if (data.stats) {
                document.getElementById('totalTasks').textContent = data.stats.total_tasks || 0;
                document.getElementById('completedTasks').textContent = data.stats.completed_tasks || 0;
                document.getElementById('failedTasks').textContent = data.stats.failed_tasks || 0;
                
                const rate = data.stats.total_tasks > 0 
                    ? ((data.stats.completed_tasks / data.stats.total_tasks) * 100).toFixed(1)
                    : 0;
                document.getElementById('successRate').textContent = rate + '%';
                
                document.getElementById('activeWorkers').textContent = data.stats.active_workers || 0;
                document.getElementById('queueSize').textContent = data.stats.queue_size || 0;
                
                // Update workers grid
                const grid = document.getElementById('workersGrid');
                grid.innerHTML = '';
                const active = data.stats.active_workers || 0;
                const max = data.stats.max_workers || 10;
                for (let i = 0; i < max; i++) {
                    const worker = document.createElement('div');
                    worker.className = 'worker ' + (i < active ? 'worker-active' : 'worker-idle');
                    worker.textContent = i + 1;
                    grid.appendChild(worker);
                }
            }
            
            // Update health
            if (data.health && data.health.checks) {
                const memory = data.health.checks.memory;
                const cpu = data.health.checks.cpu;
                
                if (memory) {
                    const memPct = memory.details?.percent || 0;
                    document.getElementById('memoryPercent').textContent = memPct.toFixed(1) + '%';
                    document.getElementById('memoryBar').style.width = memPct + '%';
                }
                
                if (cpu) {
                    const cpuPct = cpu.details?.percent || 0;
                    document.getElementById('cpuPercent').textContent = cpuPct.toFixed(1) + '%';
                    document.getElementById('cpuBar').style.width = cpuPct + '%';
                }
            }
            
            // Add log entry
            if (data.log) {
                const container = document.getElementById('logContainer');
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                const time = new Date().toLocaleTimeString();
                entry.innerHTML = `<span class="log-time">[${time}]</span> ${data.log}`;
                container.insertBefore(entry, container.firstChild);
                
                // Keep only last 50 entries
                while (container.children.length > 50) {
                    container.removeChild(container.lastChild);
                }
            }
        }
        
        // Request updates every 2 seconds
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'get_stats'}));
            }
        }, 2000);
    </script>
</body>
</html>
"""


class Dashboard:
    """
    Web dashboard for monitoring the autonomous agent.
    
    Features:
    - Real-time statistics via WebSocket
    - Task and worker monitoring
    - System resource visualization
    - Activity logging
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        agent: Optional["AutonomousSAM3Agent"] = None
    ):
        """
        Initialize dashboard.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            agent: Agent instance to monitor
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for dashboard. "
                "Install with: pip install fastapi uvicorn"
            )
        
        self.host = host
        self.port = port
        self.agent = agent
        self.app = FastAPI(title="Autonomous Agent Dashboard")
        self.manager = ConnectionManager()
        self._running = False
        self._server_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        
        self._setup_routes()
        self._setup_cors()
        
        logger.info(f"Initialized Dashboard (http://{host}:{port})")
    
    def _setup_cors(self):
        """Setup CORS middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            return create_dashboard_html()
        
        @self.app.get("/stats")
        async def get_stats():
            if self.agent:
                return self.agent.get_stats()
            return {"error": "Agent not connected"}
        
        @self.app.get("/health")
        async def get_health():
            if self.agent and hasattr(self.agent, "health_check"):
                return await self.agent.health_check()
            return {"status": "unknown"}
        
        @self.app.get("/tasks")
        async def get_tasks():
            if self.agent and hasattr(self.agent, "task_manager"):
                return await self.agent.task_manager.get_pending_tasks(limit=50)
            return []
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    msg = json.loads(data)
                    
                    if msg.get("type") == "get_stats":
                        response = await self._get_dashboard_data()
                        await websocket.send_json(response)
            except WebSocketDisconnect:
                self.manager.disconnect(websocket)
    
    async def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data."""
        data = {"timestamp": datetime.now().isoformat()}
        
        if self.agent:
            data["stats"] = self.agent.get_stats()
            
            if hasattr(self.agent, "health_monitor"):
                data["health"] = self.agent.health_monitor.get_current_health()
        
        return data
    
    async def start(self):
        """Start the dashboard server."""
        if self._running:
            logger.warning("Dashboard is already running")
            return
        
        self._running = True
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        
        self._server_task = asyncio.create_task(server.serve())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        
        logger.info(f"Dashboard started at http://{self.host}:{self.port}")
    
    async def stop(self):
        """Stop the dashboard server."""
        if not self._running:
            return
        
        self._running = False
        
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dashboard stopped")
    
    async def _broadcast_loop(self):
        """Broadcast updates to all connected clients."""
        while self._running:
            try:
                data = await self._get_dashboard_data()
                await self.manager.broadcast(data)
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(5)
    
    async def log_activity(self, message: str):
        """
        Log an activity message to the dashboard.
        
        Args:
            message: Message to log
        """
        await self.manager.broadcast({"log": message})
