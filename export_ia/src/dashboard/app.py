"""
Management dashboard application for Export IA.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import json
from datetime import datetime, timedelta

from .api import DashboardAPI
from .components import (
    MetricsWidget,
    TaskMonitor,
    ServiceStatus,
    PerformanceChart,
    SystemHealth
)

logger = logging.getLogger(__name__)


class DashboardApp:
    """Main dashboard application."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Export IA Dashboard",
            description="Management dashboard for Export IA system",
            version="2.0.0"
        )
        
        # Initialize components
        self.api = DashboardAPI()
        self.metrics_widget = MetricsWidget()
        self.task_monitor = TaskMonitor()
        self.service_status = ServiceStatus()
        self.performance_chart = PerformanceChart()
        self.system_health = SystemHealth()
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        self._setup_websocket()
    
    def _setup_routes(self) -> None:
        """Setup dashboard routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard page."""
            return self._render_dashboard(request)
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get system metrics."""
            return await self.metrics_widget.get_metrics()
        
        @self.app.get("/api/tasks")
        async def get_tasks():
            """Get task information."""
            return await self.task_monitor.get_tasks()
        
        @self.app.get("/api/services")
        async def get_services():
            """Get service status."""
            return await self.service_status.get_services()
        
        @self.app.get("/api/performance")
        async def get_performance():
            """Get performance data."""
            return await self.performance_chart.get_performance_data()
        
        @self.app.get("/api/health")
        async def get_health():
            """Get system health."""
            return await self.system_health.get_health_status()
        
        @self.app.get("/api/statistics")
        async def get_statistics():
            """Get system statistics."""
            return await self.api.get_system_statistics()
    
    def _setup_websocket(self) -> None:
        """Setup WebSocket for real-time updates."""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                # Send initial data
                await self._send_initial_data(websocket)
                
                # Keep connection alive and send updates
                while True:
                    await asyncio.sleep(5)  # Update every 5 seconds
                    await self._send_update(websocket)
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
    
    async def _send_initial_data(self, websocket: WebSocket) -> None:
        """Send initial data to WebSocket client."""
        try:
            data = {
                "type": "initial_data",
                "timestamp": datetime.now().isoformat(),
                "metrics": await self.metrics_widget.get_metrics(),
                "tasks": await self.task_monitor.get_tasks(),
                "services": await self.service_status.get_services(),
                "performance": await self.performance_chart.get_performance_data(),
                "health": await self.system_health.get_health_status()
            }
            
            await websocket.send_text(json.dumps(data))
            
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    async def _send_update(self, websocket: WebSocket) -> None:
        """Send update to WebSocket client."""
        try:
            data = {
                "type": "update",
                "timestamp": datetime.now().isoformat(),
                "metrics": await self.metrics_widget.get_metrics(),
                "tasks": await self.task_monitor.get_tasks(),
                "services": await self.service_status.get_services(),
                "performance": await self.performance_chart.get_performance_data(),
                "health": await self.system_health.get_health_status()
            }
            
            await websocket.send_text(json.dumps(data))
            
        except Exception as e:
            logger.error(f"Error sending update: {e}")
    
    def _render_dashboard(self, request: Request) -> str:
        """Render dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Export IA Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .metric-card { @apply bg-white rounded-lg shadow-md p-6; }
                .status-indicator { @apply w-3 h-3 rounded-full; }
                .status-healthy { @apply bg-green-500; }
                .status-warning { @apply bg-yellow-500; }
                .status-error { @apply bg-red-500; }
            </style>
        </head>
        <body class="bg-gray-100">
            <div class="container mx-auto px-4 py-8">
                <h1 class="text-3xl font-bold text-gray-800 mb-8">Export IA Dashboard</h1>
                
                <!-- System Health Overview -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <div class="metric-card">
                        <h3 class="text-lg font-semibold text-gray-700 mb-2">System Health</h3>
                        <div class="flex items-center">
                            <div id="health-status" class="status-indicator status-healthy mr-2"></div>
                            <span id="health-text" class="text-gray-600">Healthy</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h3 class="text-lg font-semibold text-gray-700 mb-2">Active Tasks</h3>
                        <div class="text-2xl font-bold text-blue-600" id="active-tasks">0</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3 class="text-lg font-semibold text-gray-700 mb-2">Completed Today</h3>
                        <div class="text-2xl font-bold text-green-600" id="completed-today">0</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3 class="text-lg font-semibold text-gray-700 mb-2">Services Running</h3>
                        <div class="text-2xl font-bold text-purple-600" id="services-running">0</div>
                    </div>
                </div>
                
                <!-- Charts Row -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    <div class="metric-card">
                        <h3 class="text-lg font-semibold text-gray-700 mb-4">Performance Metrics</h3>
                        <canvas id="performance-chart" width="400" height="200"></canvas>
                    </div>
                    
                    <div class="metric-card">
                        <h3 class="text-lg font-semibold text-gray-700 mb-4">Task Distribution</h3>
                        <canvas id="task-chart" width="400" height="200"></canvas>
                    </div>
                </div>
                
                <!-- Services Status -->
                <div class="metric-card mb-8">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Services Status</h3>
                    <div id="services-list" class="space-y-2">
                        <!-- Services will be populated here -->
                    </div>
                </div>
                
                <!-- Recent Tasks -->
                <div class="metric-card">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Recent Tasks</h3>
                    <div id="recent-tasks" class="space-y-2">
                        <!-- Tasks will be populated here -->
                    </div>
                </div>
            </div>
            
            <script>
                // WebSocket connection
                const ws = new WebSocket('ws://localhost:8080/ws');
                
                // Chart instances
                let performanceChart = null;
                let taskChart = null;
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                function updateDashboard(data) {
                    // Update health status
                    const healthStatus = document.getElementById('health-status');
                    const healthText = document.getElementById('health-text');
                    
                    if (data.health && data.health.status === 'healthy') {
                        healthStatus.className = 'status-indicator status-healthy mr-2';
                        healthText.textContent = 'Healthy';
                    } else {
                        healthStatus.className = 'status-indicator status-error mr-2';
                        healthText.textContent = 'Issues Detected';
                    }
                    
                    // Update metrics
                    if (data.metrics) {
                        document.getElementById('active-tasks').textContent = data.metrics.active_tasks || 0;
                        document.getElementById('completed-today').textContent = data.metrics.completed_today || 0;
                        document.getElementById('services-running').textContent = data.metrics.services_running || 0;
                    }
                    
                    // Update services
                    if (data.services) {
                        updateServicesList(data.services);
                    }
                    
                    // Update tasks
                    if (data.tasks) {
                        updateRecentTasks(data.tasks);
                    }
                    
                    // Update charts
                    if (data.performance) {
                        updatePerformanceChart(data.performance);
                    }
                    
                    if (data.tasks) {
                        updateTaskChart(data.tasks);
                    }
                }
                
                function updateServicesList(services) {
                    const servicesList = document.getElementById('services-list');
                    servicesList.innerHTML = '';
                    
                    Object.entries(services).forEach(([name, service]) => {
                        const serviceDiv = document.createElement('div');
                        serviceDiv.className = 'flex items-center justify-between p-3 bg-gray-50 rounded';
                        
                        const statusClass = service.status === 'healthy' ? 'status-healthy' : 'status-error';
                        
                        serviceDiv.innerHTML = `
                            <div class="flex items-center">
                                <div class="status-indicator ${statusClass} mr-3"></div>
                                <span class="font-medium">${name}</span>
                            </div>
                            <span class="text-sm text-gray-500">${service.status}</span>
                        `;
                        
                        servicesList.appendChild(serviceDiv);
                    });
                }
                
                function updateRecentTasks(tasks) {
                    const recentTasks = document.getElementById('recent-tasks');
                    recentTasks.innerHTML = '';
                    
                    tasks.slice(0, 5).forEach(task => {
                        const taskDiv = document.createElement('div');
                        taskDiv.className = 'flex items-center justify-between p-3 bg-gray-50 rounded';
                        
                        const statusClass = task.status === 'completed' ? 'status-healthy' : 
                                          task.status === 'failed' ? 'status-error' : 'status-warning';
                        
                        taskDiv.innerHTML = `
                            <div class="flex items-center">
                                <div class="status-indicator ${statusClass} mr-3"></div>
                                <span class="font-medium">${task.id}</span>
                            </div>
                            <span class="text-sm text-gray-500">${task.status}</span>
                        `;
                        
                        recentTasks.appendChild(taskDiv);
                    });
                }
                
                function updatePerformanceChart(performance) {
                    const ctx = document.getElementById('performance-chart').getContext('2d');
                    
                    if (performanceChart) {
                        performanceChart.destroy();
                    }
                    
                    performanceChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: performance.labels || [],
                            datasets: [{
                                label: 'Response Time (ms)',
                                data: performance.response_times || [],
                                borderColor: 'rgb(75, 192, 192)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
                
                function updateTaskChart(tasks) {
                    const ctx = document.getElementById('task-chart').getContext('2d');
                    
                    if (taskChart) {
                        taskChart.destroy();
                    }
                    
                    // Count tasks by status
                    const statusCounts = tasks.reduce((acc, task) => {
                        acc[task.status] = (acc[task.status] || 0) + 1;
                        return acc;
                    }, {});
                    
                    taskChart = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: Object.keys(statusCounts),
                            datasets: [{
                                data: Object.values(statusCounts),
                                backgroundColor: [
                                    'rgb(34, 197, 94)',  // green
                                    'rgb(239, 68, 68)',  // red
                                    'rgb(245, 158, 11)', // yellow
                                    'rgb(59, 130, 246)'  // blue
                                ]
                            }]
                        },
                        options: {
                            responsive: true
                        }
                    });
                }
            </script>
        </body>
        </html>
        """
    
    async def start(self) -> None:
        """Start the dashboard application."""
        import uvicorn
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info(f"Dashboard starting on {self.host}:{self.port}")
        await server.serve()
    
    async def stop(self) -> None:
        """Stop the dashboard application."""
        # Close all WebSocket connections
        for connection in self.active_connections:
            await connection.close()
        
        logger.info("Dashboard stopped")


# Global dashboard instance
_dashboard_app: Optional[DashboardApp] = None


def get_dashboard_app() -> DashboardApp:
    """Get the global dashboard application instance."""
    global _dashboard_app
    if _dashboard_app is None:
        _dashboard_app = DashboardApp()
    return _dashboard_app




