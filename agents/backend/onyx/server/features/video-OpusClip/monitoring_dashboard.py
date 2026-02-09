"""
Real-time Monitoring Dashboard for Ultimate Opus Clip

Advanced monitoring dashboard that provides real-time insights into
system performance, job status, and resource utilization.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
import asyncio
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
import psutil
import threading
from datetime import datetime, timedelta
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = structlog.get_logger("monitoring_dashboard")

class MetricType(Enum):
    """Types of metrics."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    JOB_COUNT = "job_count"
    PROCESSING_TIME = "processing_time"
    ERROR_RATE = "error_rate"

@dataclass
class MetricData:
    """A single metric data point."""
    timestamp: float
    value: float
    metric_type: MetricType
    tags: Dict[str, str] = None

@dataclass
class SystemStatus:
    """Current system status."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_processing_time: float
    error_rate: float
    timestamp: float

class MonitoringDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self):
        self.metrics: List[MetricData] = []
        self.max_metrics = 1000
        self.websocket_connections: List[WebSocket] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Ultimate Opus Clip Monitoring Dashboard")
        self._setup_routes()
        
        logger.info("Monitoring dashboard initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def dashboard():
            """Serve the main dashboard."""
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current system status."""
            return self.get_system_status()
        
        @self.app.get("/api/metrics")
        async def get_metrics(metric_type: Optional[str] = None, limit: int = 100):
            """Get metrics data."""
            return self.get_metrics_data(metric_type, limit)
        
        @self.app.get("/api/jobs")
        async def get_jobs():
            """Get job information."""
            return self.get_jobs_data()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    status = self.get_system_status()
                    await websocket.send_json(status)
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Get the dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ultimate Opus Clip - Monitoring Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
                .metric-value { font-size: 24px; font-weight: bold; color: #27ae60; }
                .metric-chart { height: 200px; margin-top: 10px; }
                .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
                .status-online { background: #27ae60; }
                .status-offline { background: #e74c3c; }
                .jobs-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                .jobs-table th, .jobs-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                .jobs-table th { background: #f8f9fa; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎬 Ultimate Opus Clip - Monitoring Dashboard</h1>
                    <p>Real-time system monitoring and performance metrics</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">System Status</div>
                        <div id="system-status">
                            <span class="status-indicator status-online"></span>
                            <span>Online</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">CPU Usage</div>
                        <div class="metric-value" id="cpu-usage">0%</div>
                        <canvas id="cpu-chart" class="metric-chart"></canvas>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Memory Usage</div>
                        <div class="metric-value" id="memory-usage">0%</div>
                        <canvas id="memory-chart" class="metric-chart"></canvas>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">GPU Usage</div>
                        <div class="metric-value" id="gpu-usage">0%</div>
                        <canvas id="gpu-chart" class="metric-chart"></canvas>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Active Jobs</div>
                        <div class="metric-value" id="active-jobs">0</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Completed Jobs</div>
                        <div class="metric-value" id="completed-jobs">0</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Failed Jobs</div>
                        <div class="metric-value" id="failed-jobs">0</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Average Processing Time</div>
                        <div class="metric-value" id="avg-processing-time">0s</div>
                    </div>
                </div>
                
                <div class="metric-card" style="margin-top: 20px;">
                    <div class="metric-title">Recent Jobs</div>
                    <table class="jobs-table" id="jobs-table">
                        <thead>
                            <tr>
                                <th>Job ID</th>
                                <th>Status</th>
                                <th>Progress</th>
                                <th>Created</th>
                                <th>Duration</th>
                            </tr>
                        </thead>
                        <tbody id="jobs-tbody">
                        </tbody>
                    </table>
                </div>
            </div>
            
            <script>
                // Initialize charts
                const charts = {};
                const chartConfigs = {
                    'cpu-chart': { color: '#3498db', label: 'CPU Usage %' },
                    'memory-chart': { color: '#e74c3c', label: 'Memory Usage %' },
                    'gpu-chart': { color: '#9b59b6', label: 'GPU Usage %' }
                };
                
                Object.keys(chartConfigs).forEach(chartId => {
                    const ctx = document.getElementById(chartId).getContext('2d');
                    charts[chartId] = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: chartConfigs[chartId].label,
                                data: [],
                                borderColor: chartConfigs[chartId].color,
                                backgroundColor: chartConfigs[chartId].color + '20',
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: { beginAtZero: true, max: 100 }
                            }
                        }
                    });
                });
                
                // WebSocket connection
                const ws = new WebSocket('ws://localhost:8001/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                function updateDashboard(data) {
                    // Update metric values
                    document.getElementById('cpu-usage').textContent = data.cpu_usage.toFixed(1) + '%';
                    document.getElementById('memory-usage').textContent = data.memory_usage.toFixed(1) + '%';
                    document.getElementById('gpu-usage').textContent = data.gpu_usage.toFixed(1) + '%';
                    document.getElementById('active-jobs').textContent = data.active_jobs;
                    document.getElementById('completed-jobs').textContent = data.completed_jobs;
                    document.getElementById('failed-jobs').textContent = data.failed_jobs;
                    document.getElementById('avg-processing-time').textContent = data.average_processing_time.toFixed(1) + 's';
                    
                    // Update charts
                    const now = new Date().toLocaleTimeString();
                    Object.keys(charts).forEach(chartId => {
                        const chart = charts[chartId];
                        chart.data.labels.push(now);
                        chart.data.datasets[0].data.push(data[chartId.replace('-chart', '_usage')]);
                        
                        // Keep only last 20 data points
                        if (chart.data.labels.length > 20) {
                            chart.data.labels.shift();
                            chart.data.datasets[0].data.shift();
                        }
                        
                        chart.update();
                    });
                }
                
                // Fetch initial data
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => updateDashboard(data));
            </script>
        </body>
        </html>
        """
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Send updates to WebSocket connections
                self._send_websocket_updates()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _collect_metrics(self):
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU usage (if available)
            gpu_usage = 0.0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Add metrics
            self._add_metric(MetricType.CPU_USAGE, cpu_usage)
            self._add_metric(MetricType.MEMORY_USAGE, memory_usage)
            self._add_metric(MetricType.GPU_USAGE, gpu_usage)
            self._add_metric(MetricType.DISK_USAGE, disk_usage)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _add_metric(self, metric_type: MetricType, value: float, tags: Dict[str, str] = None):
        """Add a metric data point."""
        with self.lock:
            metric = MetricData(
                timestamp=time.time(),
                value=value,
                metric_type=metric_type,
                tags=tags or {}
            )
            
            self.metrics.append(metric)
            
            # Keep only recent metrics
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
    
    def _send_websocket_updates(self):
        """Send updates to WebSocket connections."""
        if not self.websocket_connections:
            return
        
        try:
            status = self.get_system_status()
            status_json = json.dumps(status)
            
            # Send to all connected clients
            for websocket in self.websocket_connections.copy():
                try:
                    asyncio.create_task(websocket.send_text(status_json))
                except Exception as e:
                    logger.error(f"Error sending WebSocket update: {e}")
                    self.websocket_connections.remove(websocket)
                    
        except Exception as e:
            logger.error(f"Error sending WebSocket updates: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            # Get latest metrics
            cpu_usage = self._get_latest_metric(MetricType.CPU_USAGE) or 0.0
            memory_usage = self._get_latest_metric(MetricType.MEMORY_USAGE) or 0.0
            gpu_usage = self._get_latest_metric(MetricType.GPU_USAGE) or 0.0
            disk_usage = self._get_latest_metric(MetricType.DISK_USAGE) or 0.0
            
            # Get job statistics (placeholder - would integrate with actual job system)
            active_jobs = 0
            completed_jobs = 0
            failed_jobs = 0
            average_processing_time = 0.0
            error_rate = 0.0
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "gpu_usage": gpu_usage,
                "disk_usage": disk_usage,
                "active_jobs": active_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "average_processing_time": average_processing_time,
                "error_rate": error_rate,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    def _get_latest_metric(self, metric_type: MetricType) -> Optional[float]:
        """Get the latest value for a metric type."""
        with self.lock:
            for metric in reversed(self.metrics):
                if metric.metric_type == metric_type:
                    return metric.value
        return None
    
    def get_metrics_data(self, metric_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics data."""
        with self.lock:
            metrics = self.metrics
            
            if metric_type:
                metrics = [m for m in metrics if m.metric_type.value == metric_type]
            
            # Return most recent metrics
            metrics = metrics[-limit:]
            
            return [asdict(metric) for metric in metrics]
    
    def get_jobs_data(self) -> List[Dict[str, Any]]:
        """Get job data (placeholder)."""
        # This would integrate with the actual job system
        return []
    
    def run_dashboard(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the monitoring dashboard."""
        self.start_monitoring()
        
        logger.info(f"Starting monitoring dashboard on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )

# Global dashboard instance
_global_dashboard: Optional[MonitoringDashboard] = None

def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = MonitoringDashboard()
    return _global_dashboard

def start_monitoring_dashboard():
    """Start the monitoring dashboard."""
    dashboard = get_monitoring_dashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    start_monitoring_dashboard()


