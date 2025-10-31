from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
import json
import sqlite3
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import functools
import weakref
import structlog
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import redis.asyncio as redis
from .api_performance_metrics import (
from .api_performance_optimizer import (
        from .api_performance_metrics import get_api_monitor
        from .api_performance_optimizer import get_api_optimizer
from typing import Any, List, Dict, Optional
"""
📊 API Performance Dashboard
============================

Comprehensive API performance dashboard system with:
- Real-time performance visualization
- Interactive metrics charts
- Performance alerts dashboard
- SLA compliance monitoring
- Optimization recommendations display
- Historical trend analysis
- Performance comparison tools
- Custom dashboard widgets
- Export and reporting capabilities
- Multi-endpoint monitoring
"""



    APIPerformanceMonitor, APIPerformanceMetrics, MetricPriority, 
    PerformanceThreshold, LatencyType
)
    APIPerformanceOptimizer, OptimizationType, OptimizationRecommendation
)

logger = structlog.get_logger(__name__)

class DashboardWidget(Enum):
    """Dashboard widget types"""
    RESPONSE_TIME_CHART = "response_time_chart"
    THROUGHPUT_CHART = "throughput_chart"
    ERROR_RATE_CHART = "error_rate_chart"
    LATENCY_BREAKDOWN = "latency_breakdown"
    SLA_COMPLIANCE = "sla_compliance"
    OPTIMIZATION_RECOMMENDATIONS = "optimization_recommendations"
    PERFORMANCE_ALERTS = "performance_alerts"
    TREND_ANALYSIS = "trend_analysis"
    ENDPOINT_COMPARISON = "endpoint_comparison"
    SYSTEM_RESOURCES = "system_resources"

class DashboardConfig:
    """Dashboard configuration"""
    
    def __init__(self) -> Any:
        self.refresh_interval = 5  # seconds
        self.history_window = 3600  # 1 hour
        self.max_data_points = 1000
        self.enable_real_time = True
        self.enable_alerts = True
        self.enable_optimizations = True
        self.default_widgets = [
            DashboardWidget.RESPONSE_TIME_CHART,
            DashboardWidget.THROUGHPUT_CHART,
            DashboardWidget.SLA_COMPLIANCE,
            DashboardWidget.OPTIMIZATION_RECOMMENDATIONS
        ]

@dataclass
class DashboardData:
    """Dashboard data structure"""
    timestamp: float
    metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    sla_compliance: Dict[str, Any]
    trends: Dict[str, Any]

class DashboardManager:
    """Dashboard data manager"""
    
    def __init__(self, monitor: APIPerformanceMonitor, optimizer: APIPerformanceOptimizer):
        
    """__init__ function."""
self.monitor = monitor
        self.optimizer = optimizer
        self.config = DashboardConfig()
        
        # Dashboard data storage
        self.dashboard_data: deque = deque(maxlen=self.config.max_data_points)
        self.websocket_connections: List[WebSocket] = []
        self.websocket_lock = asyncio.Lock()
        
        # Dashboard state
        self.is_running = False
        self.last_update = time.time()
        
        logger.info("Dashboard Manager initialized")
    
    async def start(self) -> Any:
        """Start the dashboard manager"""
        self.is_running = True
        logger.info("Dashboard Manager started")
        
        # Start data collection loop
        asyncio.create_task(self._data_collection_loop())
        
        # Start websocket broadcast loop
        asyncio.create_task(self._websocket_broadcast_loop())
    
    async def stop(self) -> Any:
        """Stop the dashboard manager"""
        self.is_running = False
        logger.info("Dashboard Manager stopped")
    
    async def _data_collection_loop(self) -> Any:
        """Collect dashboard data periodically"""
        while self.is_running:
            try:
                # Collect current data
                dashboard_data = await self._collect_dashboard_data()
                
                # Store data
                self.dashboard_data.append(dashboard_data)
                self.last_update = time.time()
                
                # Wait for next collection
                await asyncio.sleep(self.config.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(10)
    
    async def _collect_dashboard_data(self) -> DashboardData:
        """Collect current dashboard data"""
        # Get metrics
        metrics_summary = self.monitor.get_performance_summary()
        
        # Get alerts
        alerts = self.monitor.get_alerts()
        alert_data = [
            {
                "id": alert.id,
                "endpoint": alert.endpoint,
                "metric_type": alert.metric_type,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp
            }
            for alert in alerts if not alert.resolved
        ]
        
        # Get recommendations
        recommendations = self.optimizer.get_recommendations()
        recommendation_data = [
            {
                "id": rec.id,
                "endpoint": rec.endpoint,
                "type": rec.optimization_type.value,
                "description": rec.description,
                "impact": rec.impact.value,
                "priority": rec.priority.value,
                "estimated_improvement": rec.estimated_improvement
            }
            for rec in recommendations if not rec.implemented
        ]
        
        # Get SLA compliance
        sla_summary = self.optimizer.sla_monitor.get_compliance_summary()
        
        # Get trends
        optimization_summary = self.optimizer.get_optimization_summary()
        trends = optimization_summary.get("trends", {})
        
        return DashboardData(
            timestamp=time.time(),
            metrics=metrics_summary,
            alerts=alert_data,
            recommendations=recommendation_data,
            sla_compliance=sla_summary,
            trends=trends
        )
    
    async def _websocket_broadcast_loop(self) -> Any:
        """Broadcast dashboard updates to websocket clients"""
        while self.is_running:
            try:
                if self.dashboard_data:
                    latest_data = self.dashboard_data[-1]
                    
                    # Prepare broadcast message
                    message = {
                        "type": "dashboard_update",
                        "timestamp": latest_data.timestamp,
                        "data": {
                            "metrics": latest_data.metrics,
                            "alerts": latest_data.alerts,
                            "recommendations": latest_data.recommendations,
                            "sla_compliance": latest_data.sla_compliance
                        }
                    }
                    
                    # Broadcast to all connected clients
                    await self._broadcast_to_websockets(message)
                
                await asyncio.sleep(self.config.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in websocket broadcast loop: {e}")
                await asyncio.sleep(10)
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add a new websocket connection"""
        async with self.websocket_lock:
            self.websocket_connections.append(websocket)
            logger.info(f"WebSocket connection added. Total connections: {len(self.websocket_connections)}")
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove a websocket connection"""
        async with self.websocket_lock:
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)
                logger.info(f"WebSocket connection removed. Total connections: {len(self.websocket_connections)}")
    
    async def _broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast message to all websocket connections"""
        async with self.websocket_lock:
            disconnected = []
            
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.warning(f"Failed to send to websocket: {e}")
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for websocket in disconnected:
                self.websocket_connections.remove(websocket)
    
    def get_dashboard_data(self, 
                          widget_type: Optional[DashboardWidget] = None,
                          time_window: Optional[int] = None) -> Dict[str, Any]:
        """Get dashboard data for specific widget or time window"""
        if not self.dashboard_data:
            return {"error": "No dashboard data available"}
        
        if time_window:
            cutoff_time = time.time() - time_window
            filtered_data = [
                data for data in self.dashboard_data 
                if data.timestamp >= cutoff_time
            ]
        else:
            filtered_data = list(self.dashboard_data)
        
        if not filtered_data:
            return {"error": "No data available for specified time window"}
        
        if widget_type:
            return self._get_widget_data(widget_type, filtered_data)
        else:
            return {
                "latest": {
                    "timestamp": filtered_data[-1].timestamp,
                    "metrics": filtered_data[-1].metrics,
                    "alerts": filtered_data[-1].alerts,
                    "recommendations": filtered_data[-1].recommendations,
                    "sla_compliance": filtered_data[-1].sla_compliance
                },
                "history": [
                    {
                        "timestamp": data.timestamp,
                        "metrics_summary": data.metrics["summary"]
                    }
                    for data in filtered_data[-100:]  # Last 100 data points
                ]
            }
    
    def _get_widget_data(self, widget_type: DashboardWidget, data: List[DashboardData]) -> Dict[str, Any]:
        """Get data for specific widget type"""
        if widget_type == DashboardWidget.RESPONSE_TIME_CHART:
            return self._get_response_time_chart_data(data)
        elif widget_type == DashboardWidget.THROUGHPUT_CHART:
            return self._get_throughput_chart_data(data)
        elif widget_type == DashboardWidget.ERROR_RATE_CHART:
            return self._get_error_rate_chart_data(data)
        elif widget_type == DashboardWidget.LATENCY_BREAKDOWN:
            return self._get_latency_breakdown_data(data)
        elif widget_type == DashboardWidget.SLA_COMPLIANCE:
            return self._get_sla_compliance_data(data)
        elif widget_type == DashboardWidget.OPTIMIZATION_RECOMMENDATIONS:
            return self._get_optimization_recommendations_data(data)
        elif widget_type == DashboardWidget.PERFORMANCE_ALERTS:
            return self._get_performance_alerts_data(data)
        elif widget_type == DashboardWidget.TREND_ANALYSIS:
            return self._get_trend_analysis_data(data)
        else:
            return {"error": f"Unknown widget type: {widget_type.value}"}
    
    def _get_response_time_chart_data(self, data: List[DashboardData]) -> Dict[str, Any]:
        """Get response time chart data"""
        chart_data = {
            "labels": [],
            "datasets": {
                "average": [],
                "p95": [],
                "p99": []
            }
        }
        
        for data_point in data:
            chart_data["labels"].append(datetime.fromtimestamp(data_point.timestamp).strftime("%H:%M:%S"))
            
            metrics = data_point.metrics
            if "summary" in metrics:
                summary = metrics["summary"]
                chart_data["datasets"]["average"].append(summary.get("overall_avg_response_time", 0))
                chart_data["datasets"]["p95"].append(summary.get("overall_p95_response_time", 0))
                chart_data["datasets"]["p99"].append(summary.get("overall_p99_response_time", 0))
        
        return chart_data
    
    def _get_throughput_chart_data(self, data: List[DashboardData]) -> Dict[str, Any]:
        """Get throughput chart data"""
        chart_data = {
            "labels": [],
            "datasets": {
                "requests_per_second": [],
                "total_requests": []
            }
        }
        
        for data_point in data:
            chart_data["labels"].append(datetime.fromtimestamp(data_point.timestamp).strftime("%H:%M:%S"))
            
            metrics = data_point.metrics
            if "summary" in metrics:
                summary = metrics["summary"]
                chart_data["datasets"]["requests_per_second"].append(summary.get("total_throughput", 0))
                chart_data["datasets"]["total_requests"].append(summary.get("total_requests", 0))
        
        return chart_data
    
    def _get_error_rate_chart_data(self, data: List[DashboardData]) -> Dict[str, Any]:
        """Get error rate chart data"""
        chart_data = {
            "labels": [],
            "datasets": {
                "error_rate": []
            }
        }
        
        for data_point in data:
            chart_data["labels"].append(datetime.fromtimestamp(data_point.timestamp).strftime("%H:%M:%S"))
            
            metrics = data_point.metrics
            if "summary" in metrics:
                summary = metrics["summary"]
                success_rate = summary.get("success_rate", 1.0)
                error_rate = 1 - success_rate
                chart_data["datasets"]["error_rate"].append(error_rate)
        
        return chart_data
    
    def _get_latency_breakdown_data(self, data: List[DashboardData]) -> Dict[str, Any]:
        """Get latency breakdown data"""
        if not data:
            return {"error": "No data available"}
        
        latest_data = data[-1]
        endpoints = latest_data.metrics.get("endpoints", {})
        
        breakdown_data = {}
        for endpoint, metrics in endpoints.items():
            latency = metrics.get("latency_breakdown", {})
            breakdown_data[endpoint] = {
                "total": latency.get("total", 0),
                "network": latency.get("network", 0),
                "processing": latency.get("processing", 0),
                "database": latency.get("database", 0),
                "cache": latency.get("cache", 0),
                "external_api": latency.get("external_api", 0)
            }
        
        return breakdown_data
    
    def _get_sla_compliance_data(self, data: List[DashboardData]) -> Dict[str, Any]:
        """Get SLA compliance data"""
        if not data:
            return {"error": "No data available"}
        
        latest_data = data[-1]
        return latest_data.sla_compliance
    
    def _get_optimization_recommendations_data(self, data: List[DashboardData]) -> Dict[str, Any]:
        """Get optimization recommendations data"""
        if not data:
            return {"error": "No data available"}
        
        latest_data = data[-1]
        recommendations = latest_data.recommendations
        
        # Group by type and priority
        grouped = defaultdict(lambda: defaultdict(list))
        for rec in recommendations:
            grouped[rec["type"]][rec["priority"]].append(rec)
        
        return {
            "total": len(recommendations),
            "grouped": dict(grouped),
            "by_priority": {
                priority: len([r for r in recommendations if r["priority"] == priority])
                for priority in ["critical", "high", "medium", "low"]
            }
        }
    
    def _get_performance_alerts_data(self, data: List[DashboardData]) -> Dict[str, Any]:
        """Get performance alerts data"""
        if not data:
            return {"error": "No data available"}
        
        latest_data = data[-1]
        alerts = latest_data.alerts
        
        # Group by severity
        grouped = defaultdict(list)
        for alert in alerts:
            grouped[alert["severity"]].append(alert)
        
        return {
            "total": len(alerts),
            "grouped": dict(grouped),
            "by_severity": {
                severity: len(alerts)
                for severity, alerts in grouped.items()
            }
        }
    
    def _get_trend_analysis_data(self, data: List[DashboardData]) -> Dict[str, Any]:
        """Get trend analysis data"""
        if not data:
            return {"error": "No data available"}
        
        latest_data = data[-1]
        return latest_data.trends

class DashboardAPI:
    """FastAPI application for the dashboard"""
    
    def __init__(self, dashboard_manager: DashboardManager):
        
    """__init__ function."""
self.dashboard_manager = dashboard_manager
        self.app = FastAPI(title="API Performance Dashboard", version="1.0.0")
        self._setup_routes()
    
    def _setup_routes(self) -> Any:
        """Setup API routes"""
        
        @self.app.get("/")
        async def dashboard_home():
            """Dashboard home page"""
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.get("/api/dashboard/data")
        async def get_dashboard_data(
            widget: Optional[str] = None,
            time_window: Optional[int] = None
        ):
            """Get dashboard data"""
            widget_type = None
            if widget:
                try:
                    widget_type = DashboardWidget(widget)
                except ValueError:
                    return JSONResponse({"error": f"Invalid widget type: {widget}"}, status_code=400)
            
            data = self.dashboard_manager.get_dashboard_data(widget_type, time_window)
            return JSONResponse(data)
        
        @self.app.get("/api/dashboard/widgets")
        async def get_available_widgets():
            """Get available dashboard widgets"""
            return JSONResponse({
                "widgets": [widget.value for widget in DashboardWidget],
                "default_widgets": [widget.value for widget in self.dashboard_manager.config.default_widgets]
            })
        
        @self.app.get("/api/dashboard/config")
        async def get_dashboard_config():
            """Get dashboard configuration"""
            return JSONResponse({
                "refresh_interval": self.dashboard_manager.config.refresh_interval,
                "history_window": self.dashboard_manager.config.history_window,
                "enable_real_time": self.dashboard_manager.config.enable_real_time,
                "enable_alerts": self.dashboard_manager.config.enable_alerts,
                "enable_optimizations": self.dashboard_manager.config.enable_optimizations
            })
        
        @self.app.websocket("/ws/dashboard")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            await self.dashboard_manager.add_websocket_connection(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                await self.dashboard_manager.remove_websocket_connection(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Performance Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
                .widget { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .widget h3 { margin-top: 0; }
                .alert { background-color: #ffebee; border: 1px solid #f44336; padding: 10px; margin: 10px 0; border-radius: 3px; }
                .recommendation { background-color: #e8f5e8; border: 1px solid #4caf50; padding: 10px; margin: 10px 0; border-radius: 3px; }
                .status { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; }
                .status.good { background-color: #4caf50; }
                .status.warning { background-color: #ff9800; }
                .status.critical { background-color: #f44336; }
            </style>
        </head>
        <body>
            <h1>🚀 API Performance Dashboard</h1>
            
            <div class="dashboard">
                <div class="widget">
                    <h3>Response Time</h3>
                    <canvas id="responseTimeChart"></canvas>
                </div>
                
                <div class="widget">
                    <h3>Throughput</h3>
                    <canvas id="throughputChart"></canvas>
                </div>
                
                <div class="widget">
                    <h3>SLA Compliance</h3>
                    <div id="slaCompliance"></div>
                </div>
                
                <div class="widget">
                    <h3>Performance Alerts</h3>
                    <div id="alerts"></div>
                </div>
                
                <div class="widget">
                    <h3>Optimization Recommendations</h3>
                    <div id="recommendations"></div>
                </div>
            </div>
            
            <script>
                // Initialize charts
                const responseTimeCtx = document.getElementById('responseTimeChart').getContext('2d');
                const throughputCtx = document.getElementById('throughputChart').getContext('2d');
                
                const responseTimeChart = new Chart(responseTimeCtx, {
                    type: 'line',
                    data: { labels: [], datasets: [] },
                    options: { responsive: true, scales: { y: { beginAtZero: true } } }
                });
                
                const throughputChart = new Chart(throughputCtx, {
                    type: 'line',
                    data: { labels: [], datasets: [] },
                    options: { responsive: true, scales: { y: { beginAtZero: true } } }
                });
                
                // WebSocket connection
                const ws = new WebSocket('ws://localhost:8000/ws/dashboard');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'dashboard_update') {
                        updateDashboard(data.data);
                    }
                };
                
                function updateDashboard(data) {
                    // Update charts
                    updateResponseTimeChart(data.metrics);
                    updateThroughputChart(data.metrics);
                    
                    // Update widgets
                    updateSLACompliance(data.sla_compliance);
                    updateAlerts(data.alerts);
                    updateRecommendations(data.recommendations);
                }
                
                function updateResponseTimeChart(metrics) {
                    // Update response time chart data
                    // Implementation would depend on the actual data structure
                }
                
                function updateThroughputChart(metrics) {
                    // Update throughput chart data
                    // Implementation would depend on the actual data structure
                }
                
                function updateSLACompliance(slaData) {
                    const container = document.getElementById('slaCompliance');
                    container.innerHTML = '';
                    
                    if (slaData.overall_compliance_rate >= 0.99) {
                        container.innerHTML = '<div class="status good">Excellent</div>';
                    } else if (slaData.overall_compliance_rate >= 0.95) {
                        container.innerHTML = '<div class="status warning">Good</div>';
                    } else {
                        container.innerHTML = '<div class="status critical">Poor</div>';
                    }
                }
                
                function updateAlerts(alerts) {
                    const container = document.getElementById('alerts');
                    container.innerHTML = '';
                    
                    alerts.forEach(alert => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = 'alert';
                        alertDiv.innerHTML = `<strong>${alert.severity.toUpperCase()}</strong>: ${alert.message}`;
                        container.appendChild(alertDiv);
                    });
                }
                
                function updateRecommendations(recommendations) {
                    const container = document.getElementById('recommendations');
                    container.innerHTML = '';
                    
                    recommendations.forEach(rec => {
                        const recDiv = document.createElement('div');
                        recDiv.className = 'recommendation';
                        recDiv.innerHTML = `<strong>${rec.priority.toUpperCase()}</strong>: ${rec.description}`;
                        container.appendChild(recDiv);
                    });
                }
                
                // Initial load
                fetch('/api/dashboard/data')
                    .then(response => response.json())
                    .then(data => {
                        if (data.latest) {
                            updateDashboard(data.latest);
                        }
                    });
            </script>
        </body>
        </html>
        """

# Global dashboard manager instance
_dashboard_manager: Optional[DashboardManager] = None

async def get_dashboard_manager() -> DashboardManager:
    """Get the global dashboard manager instance"""
    global _dashboard_manager
    if _dashboard_manager is None:
        
        monitor = await get_api_monitor()
        optimizer = await get_api_optimizer()
        _dashboard_manager = DashboardManager(monitor, optimizer)
    return _dashboard_manager

async def example_usage():
    """Example usage of the API performance dashboard"""
    
    # Get dashboard manager
    dashboard_manager = await get_dashboard_manager()
    
    # Start dashboard
    await dashboard_manager.start()
    
    # Create dashboard API
    dashboard_api = DashboardAPI(dashboard_manager)
    
    # Simulate some performance data
    monitor = await get_api_monitor()
    
    # Register endpoints and record some data
    monitor.register_endpoint("/api/users", "GET", MetricPriority.HIGH)
    monitor.register_endpoint("/api/admin", "POST", MetricPriority.CRITICAL)
    
    for i in range(20):
        monitor.record_request(
            endpoint="/api/users",
            method="GET",
            response_time=0.5 + (i % 5) * 0.1,
            status_code=200
        )
        
        monitor.record_request(
            endpoint="/api/admin",
            method="POST",
            response_time=1.0 + (i % 3) * 0.2,
            status_code=200 if i % 10 != 0 else 500
        )
        
        await asyncio.sleep(1)
    
    # Get dashboard data
    dashboard_data = dashboard_manager.get_dashboard_data()
    print("Dashboard Data Summary:")
    print(f"Total endpoints: {dashboard_data['latest']['metrics']['summary']['total_endpoints']}")
    print(f"Total requests: {dashboard_data['latest']['metrics']['summary']['total_requests']}")
    print(f"Active alerts: {len(dashboard_data['latest']['alerts'])}")
    print(f"Recommendations: {len(dashboard_data['latest']['recommendations'])}")
    
    # Get specific widget data
    response_time_data = dashboard_manager.get_dashboard_data(DashboardWidget.RESPONSE_TIME_CHART)
    print(f"\nResponse Time Chart Data: {len(response_time_data.get('labels', []))} data points")
    
    # Stop dashboard
    await dashboard_manager.stop()

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 