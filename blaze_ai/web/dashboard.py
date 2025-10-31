"""
Advanced Web Dashboard for Blaze AI

This module provides a comprehensive web dashboard with real-time monitoring,
interactive charts, and system management capabilities.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import webbrowser
import threading

try:
    from flask import Flask, render_template, jsonify, request, Response
    from flask_socketio import SocketIO, emit
    import plotly.graph_objs as go
    import plotly.utils
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    # Mock classes for when Flask is not available
    class Flask:
        def __init__(self, *args, **kwargs): pass
        def route(self, *args, **kwargs): pass
        def run(self, *args, **kwargs): pass
    class SocketIO:
        def __init__(self, *args, **kwargs): pass
        def on(self, *args, **kwargs): pass
        def emit(self, *args, **kwargs): pass
    class go:
        pass
    class plotly:
        pass

from ...core.interfaces import CoreConfig
from ...engines import get_engine_manager
from ...utils.logging import get_logger
from ...utils.metrics import get_metrics_collector
from ...utils.alerting import get_alerting_engine

# =============================================================================
# Dashboard Configuration
# =============================================================================

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    auto_open: bool = True
    refresh_interval: int = 5000  # milliseconds
    max_data_points: int = 1000
    theme: str = "dark"  # dark, light
    enable_socketio: bool = True
    enable_websockets: bool = True

# =============================================================================
# Dashboard Data Models
# =============================================================================

@dataclass
class DashboardData:
    """Dashboard data structure."""
    system_health: Dict[str, Any] = field(default_factory=dict)
    engine_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: Dict[str, Any] = field(default_factory=dict)
    charts_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class ChartData:
    """Chart data management."""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.data: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_data_point(self, chart_name: str, timestamp: float, value: float, label: str = ""):
        """Add a data point to a chart."""
        if chart_name not in self.data:
            self.data[chart_name] = []
        
        self.data[chart_name].append({
            "timestamp": timestamp,
            "value": value,
            "label": label
        })
        
        # Keep only the last max_points
        if len(self.data[chart_name]) > self.max_points:
            self.data[chart_name] = self.data[chart_name][-self.max_points:]
    
    def get_chart_data(self, chart_name: str) -> List[Dict[str, Any]]:
        """Get data for a specific chart."""
        return self.data.get(chart_name, [])
    
    def get_all_charts_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get data for all charts."""
        return self.data

# =============================================================================
# Advanced Web Dashboard
# =============================================================================

class AdvancedWebDashboard:
    """Advanced web dashboard for Blaze AI system."""
    
    def __init__(self, config: Optional[CoreConfig] = None, dashboard_config: Optional[DashboardConfig] = None):
        self.config = config
        self.dashboard_config = dashboard_config or DashboardConfig()
        self.logger = get_logger("web_dashboard")
        
        # Initialize components
        self.engine_manager = None
        self.metrics_collector = None
        self.alerting_engine = None
        
        # Chart data management
        self.chart_data = ChartData(self.dashboard_config.max_data_points)
        
        # Dashboard data
        self.dashboard_data = DashboardData()
        
        # Background tasks
        self._data_collection_task: Optional[asyncio.Task] = None
        self._socketio_task: Optional[asyncio.Task] = None
        
        # Initialize Flask app if available
        if FLASK_AVAILABLE:
            self._setup_flask_app()
        else:
            self.logger.warning("Flask not available, dashboard will not function")
    
    def _setup_flask_app(self):
        """Setup Flask application and routes."""
        try:
            self.app = Flask(__name__)
            self.app.config['SECRET_KEY'] = 'blaze_ai_dashboard_secret_key'
            
            # Setup SocketIO
            if self.dashboard_config.enable_socketio:
                self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            else:
                self.socketio = None
            
            # Setup routes
            self._setup_routes()
            
            # Setup SocketIO events
            if self.socketio:
                self._setup_socketio_events()
            
        except Exception as e:
            self.logger.error(f"Failed to setup Flask app: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        try:
            @self.app.route('/')
            def index():
                """Main dashboard page."""
                return self._render_dashboard()
            
            @self.app.route('/api/health')
            def api_health():
                """Health check API endpoint."""
                return jsonify(self._get_health_data())
            
            @self.app.route('/api/metrics')
            def api_metrics():
                """Metrics API endpoint."""
                return jsonify(self._get_metrics_data())
            
            @self.app.route('/api/alerts')
            def api_alerts():
                """Alerts API endpoint."""
                return jsonify(self._get_alerts_data())
            
            @self.app.route('/api/charts/<chart_name>')
            def api_chart_data(chart_name):
                """Chart data API endpoint."""
                return jsonify(self._get_chart_data(chart_name))
            
            @self.app.route('/api/system/status')
            def api_system_status():
                """System status API endpoint."""
                return jsonify(self._get_system_status())
            
            @self.app.route('/api/system/actions', methods=['POST'])
            def api_system_actions():
                """System actions API endpoint."""
                action = request.json.get('action')
                return jsonify(self._execute_system_action(action))
            
        except Exception as e:
            self.logger.error(f"Failed to setup routes: {e}")
    
    def _setup_socketio_events(self):
        """Setup SocketIO events."""
        try:
            @self.socketio.on('connect')
            def handle_connect():
                """Handle client connection."""
                self.logger.info("Client connected to dashboard")
                emit('connected', {'status': 'connected'})
            
            @self.socketio.on('disconnect')
            def handle_disconnect():
                """Handle client disconnection."""
                self.logger.info("Client disconnected from dashboard")
            
            @self.socketio.on('request_update')
            def handle_update_request():
                """Handle update request from client."""
                self._emit_dashboard_update()
            
        except Exception as e:
            self.logger.error(f"Failed to setup SocketIO events: {e}")
    
    def _render_dashboard(self) -> str:
        """Render the main dashboard HTML."""
        try:
            # This would return the actual HTML template
            # For now, return a simple HTML structure
            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Blaze AI Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
                    .card { background: #2a2a2a; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                    .metric { display: flex; justify-content: space-between; margin: 10px 0; }
                    .chart-container { height: 300px; }
                    .status-healthy { color: #4CAF50; }
                    .status-warning { color: #FF9800; }
                    .status-error { color: #F44336; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 Blaze AI System Dashboard</h1>
                    <p>Real-time monitoring and management</p>
                </div>
                
                <div class="dashboard-grid">
                    <div class="card">
                        <h3>System Health</h3>
                        <div id="system-health"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Engine Status</h3>
                        <div id="engine-status"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Performance Metrics</h3>
                        <div id="performance-metrics"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Active Alerts</h3>
                        <div id="active-alerts"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Response Time Trends</h3>
                        <div id="response-time-chart" class="chart-container"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Throughput Trends</h3>
                        <div id="throughput-chart" class="chart-container"></div>
                    </div>
                </div>
                
                <script>
                    // Dashboard JavaScript code
                    const socket = io();
                    
                    socket.on('connect', function() {
                        console.log('Connected to dashboard');
                        requestUpdate();
                    });
                    
                    socket.on('dashboard_update', function(data) {
                        updateDashboard(data);
                    });
                    
                    function requestUpdate() {
                        socket.emit('request_update');
                    }
                    
                    function updateDashboard(data) {
                        // Update system health
                        if (data.system_health) {
                            updateSystemHealth(data.system_health);
                        }
                        
                        // Update engine status
                        if (data.engine_metrics) {
                            updateEngineStatus(data.engine_metrics);
                        }
                        
                        // Update performance metrics
                        if (data.performance_metrics) {
                            updatePerformanceMetrics(data.performance_metrics);
                        }
                        
                        // Update alerts
                        if (data.alerts) {
                            updateAlerts(data.alerts);
                        }
                        
                        // Update charts
                        if (data.charts_data) {
                            updateCharts(data.charts_data);
                        }
                    }
                    
                    function updateSystemHealth(health) {
                        const container = document.getElementById('system-health');
                        container.innerHTML = `
                            <div class="metric">
                                <span>Overall Status:</span>
                                <span class="status-${health.status}">${health.status.toUpperCase()}</span>
                            </div>
                            <div class="metric">
                                <span>Last Update:</span>
                                <span>${new Date(health.timestamp * 1000).toLocaleTimeString()}</span>
                            </div>
                        `;
                    }
                    
                    function updateEngineStatus(engines) {
                        const container = document.getElementById('engine-status');
                        let html = '';
                        for (const [name, status] of Object.entries(engines)) {
                            html += `
                                <div class="metric">
                                    <span>${name}:</span>
                                    <span class="status-${status.status}">${status.status}</span>
                                </div>
                            `;
                        }
                        container.innerHTML = html;
                    }
                    
                    function updatePerformanceMetrics(metrics) {
                        const container = document.getElementById('performance-metrics');
                        container.innerHTML = `
                            <div class="metric">
                                <span>Total Requests:</span>
                                <span>${metrics.total_requests || 0}</span>
                            </div>
                            <div class="metric">
                                <span>Success Rate:</span>
                                <span>${metrics.success_rate || '0%'}</span>
                            </div>
                            <div class="metric">
                                <span>Avg Response Time:</span>
                                <span>${metrics.avg_response_time || '0ms'}</span>
                            </div>
                        `;
                    }
                    
                    function updateAlerts(alerts) {
                        const container = document.getElementById('active-alerts');
                        if (alerts.active_alerts && alerts.active_alerts > 0) {
                            container.innerHTML = `
                                <div class="metric">
                                    <span>Active Alerts:</span>
                                    <span class="status-error">${alerts.active_alerts}</span>
                                </div>
                            `;
                        } else {
                            container.innerHTML = `
                                <div class="metric">
                                    <span>Active Alerts:</span>
                                    <span class="status-healthy">None</span>
                                </div>
                            `;
                        }
                    }
                    
                    function updateCharts(chartsData) {
                        // Update response time chart
                        if (chartsData.response_time) {
                            updateResponseTimeChart(chartsData.response_time);
                        }
                        
                        // Update throughput chart
                        if (chartsData.throughput) {
                            updateThroughputChart(chartsData.throughput);
                        }
                    }
                    
                    function updateResponseTimeChart(data) {
                        const trace = {
                            x: data.map(d => new Date(d.timestamp * 1000)),
                            y: data.map(d => d.value),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Response Time (ms)',
                            line: { color: '#4CAF50' }
                        };
                        
                        const layout = {
                            title: 'Response Time Trends',
                            xaxis: { title: 'Time' },
                            yaxis: { title: 'Response Time (ms)' },
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: 'white' }
                        };
                        
                        Plotly.newPlot('response-time-chart', [trace], layout);
                    }
                    
                    function updateThroughputChart(data) {
                        const trace = {
                            x: data.map(d => new Date(d.timestamp * 1000)),
                            y: data.map(d => d.value),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Throughput (req/s)',
                            line: { color: '#2196F3' }
                        };
                        
                        const layout = {
                            title: 'Throughput Trends',
                            xaxis: { title: 'Time' },
                            yaxis: { title: 'Throughput (req/s)' },
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: 'white' }
                        };
                        
                        Plotly.newPlot('throughput-chart', [trace], layout);
                    }
                    
                    // Request updates every 5 seconds
                    setInterval(requestUpdate, 5000);
                </script>
            </body>
            </html>
            """
            return html
            
        except Exception as e:
            self.logger.error(f"Failed to render dashboard: {e}")
            return f"<h1>Dashboard Error</h1><p>{str(e)}</p>"
    
    def _get_health_data(self) -> Dict[str, Any]:
        """Get system health data."""
        try:
            if not self.engine_manager:
                return {"status": "unknown", "timestamp": time.time()}
            
            # Get engine status
            engine_status = self.engine_manager.get_engine_status()
            
            # Determine overall health
            overall_status = "healthy"
            if any(engine.get("status") == "unhealthy" for engine in engine_status.values()):
                overall_status = "unhealthy"
            elif any(engine.get("status") == "degraded" for engine in engine_status.values()):
                overall_status = "degraded"
            
            return {
                "status": overall_status,
                "engines": engine_status,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get health data: {e}")
            return {"status": "error", "error": str(e), "timestamp": time.time()}
    
    def _get_metrics_data(self) -> Dict[str, Any]:
        """Get system metrics data."""
        try:
            if not self.metrics_collector:
                return {"metrics": {}, "timestamp": time.time()}
            
            metrics = self.metrics_collector.get_metrics_summary()
            return {
                "metrics": metrics,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics data: {e}")
            return {"metrics": {}, "error": str(e), "timestamp": time.time()}
    
    def _get_alerts_data(self) -> Dict[str, Any]:
        """Get alerts data."""
        try:
            if not self.alerting_engine:
                return {"alerts": {}, "timestamp": time.time()}
            
            alerts = self.alerting_engine.get_alerts_summary()
            return {
                "alerts": alerts,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get alerts data: {e}")
            return {"alerts": {}, "error": str(e), "timestamp": time.time()}
    
    def _get_chart_data(self, chart_name: str) -> List[Dict[str, Any]]:
        """Get chart data for a specific chart."""
        try:
            return self.chart_data.get_chart_data(chart_name)
        except Exception as e:
            self.logger.error(f"Failed to get chart data for {chart_name}: {e}")
            return []
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                "timestamp": time.time(),
                "health": self._get_health_data(),
                "metrics": self._get_metrics_data(),
                "alerts": self._get_alerts_data(),
                "charts": self.chart_data.get_all_charts_data()
            }
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def _execute_system_action(self, action: str) -> Dict[str, Any]:
        """Execute a system action."""
        try:
            if action == "refresh":
                # Trigger data refresh
                self._collect_dashboard_data()
                return {"status": "success", "message": "Data refreshed"}
            
            elif action == "clear_charts":
                # Clear chart data
                self.chart_data.data.clear()
                return {"status": "success", "message": "Chart data cleared"}
            
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            self.logger.error(f"Failed to execute action {action}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _emit_dashboard_update(self):
        """Emit dashboard update via SocketIO."""
        try:
            if self.socketio:
                data = self._get_system_status()
                self.socketio.emit('dashboard_update', data)
                
        except Exception as e:
            self.logger.error(f"Failed to emit dashboard update: {e}")
    
    async def _collect_dashboard_data(self):
        """Collect dashboard data from all sources."""
        try:
            # Collect system health
            self.dashboard_data.system_health = self._get_health_data()
            
            # Collect engine metrics
            if self.engine_manager:
                self.dashboard_data.engine_metrics = self.engine_manager.get_system_metrics()
            
            # Collect performance metrics
            if self.metrics_collector:
                self.dashboard_data.performance_metrics = self.metrics_collector.get_metrics_summary()
            
            # Collect alerts
            self.dashboard_data.alerts = self._get_alerts_data()
            
            # Collect chart data
            self.dashboard_data.charts_data = self.chart_data.get_all_charts_data()
            
            # Update timestamp
            self.dashboard_data.timestamp = time.time()
            
            # Add sample chart data (in real implementation, this would come from actual metrics)
            current_time = time.time()
            self.chart_data.add_data_point("response_time", current_time, 150.0 + (current_time % 100))
            self.chart_data.add_data_point("throughput", current_time, 50.0 + (current_time % 30))
            
        except Exception as e:
            self.logger.error(f"Failed to collect dashboard data: {e}")
    
    async def _data_collection_loop(self):
        """Background data collection loop."""
        while True:
            try:
                await asyncio.sleep(5)  # Collect every 5 seconds
                await self._collect_dashboard_data()
                
                # Emit updates if SocketIO is available
                if self.socketio:
                    self._emit_dashboard_update()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Data collection error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _start_background_tasks(self):
        """Start background tasks."""
        try:
            # Initialize components
            if not self.engine_manager:
                self.engine_manager = get_engine_manager(self.config)
            
            if not self.metrics_collector:
                self.metrics_collector = get_metrics_collector(self.config)
            
            if not self.alerting_engine:
                self.alerting_engine = get_alerting_engine(self.config)
            
            # Start data collection
            self._data_collection_task = asyncio.create_task(self._data_collection_loop())
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
    
    def run(self):
        """Run the dashboard server."""
        try:
            if not FLASK_AVAILABLE:
                self.logger.error("Flask not available, cannot run dashboard")
                return
            
            # Start background tasks
            asyncio.run(self._start_background_tasks())
            
            # Open browser if configured
            if self.dashboard_config.auto_open:
                def open_browser():
                    time.sleep(1)  # Wait for server to start
                    webbrowser.open(f"http://{self.dashboard_config.host}:{self.dashboard_config.port}")
                
                threading.Thread(target=open_browser, daemon=True).start()
            
            # Run Flask app
            if self.socketio:
                self.socketio.run(
                    self.app,
                    host=self.dashboard_config.host,
                    port=self.dashboard_config.port,
                    debug=self.dashboard_config.debug
                )
            else:
                self.app.run(
                    host=self.dashboard_config.host,
                    port=self.dashboard_config.port,
                    debug=self.dashboard_config.debug
                )
                
        except Exception as e:
            self.logger.error(f"Failed to run dashboard: {e}")
    
    async def shutdown(self):
        """Shutdown the dashboard."""
        self.logger.info("Shutting down web dashboard...")
        
        # Cancel background tasks
        if self._data_collection_task:
            self._data_collection_task.cancel()
            try:
                await self._data_collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Web dashboard shutdown complete")

# =============================================================================
# Global Dashboard Instance
# =============================================================================

_global_dashboard: Optional[AdvancedWebDashboard] = None

def get_web_dashboard(config: Optional[CoreConfig] = None, 
                     dashboard_config: Optional[DashboardConfig] = None) -> AdvancedWebDashboard:
    """Get the global web dashboard instance."""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = AdvancedWebDashboard(config, dashboard_config)
    return _global_dashboard

async def shutdown_web_dashboard():
    """Shutdown the global web dashboard."""
    global _global_dashboard
    if _global_dashboard:
        await _global_dashboard.shutdown()
        _global_dashboard = None


