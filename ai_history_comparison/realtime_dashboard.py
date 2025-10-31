"""
Real-time Dashboard for AI History Analyzer
==========================================

This module provides a real-time dashboard with WebSocket support for live
monitoring of AI model performance, trends, and comparisons.

Features:
- Real-time WebSocket connections
- Live performance metrics
- Interactive charts and visualizations
- Model comparison views
- Trend analysis displays
- Alert notifications
- Historical data visualization
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config
from .integration_system import get_integration_system

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics"""
    timestamp: datetime
    active_models: int
    total_measurements: int
    average_quality: float
    average_response_time: float
    cost_efficiency: float
    top_performing_model: str
    alerts_count: int
    trends_analyzed: int


@dataclass
class ModelPerformanceSnapshot:
    """Snapshot of model performance for dashboard"""
    model_name: str
    quality_score: float
    response_time: float
    cost_efficiency: float
    token_efficiency: float
    trend_direction: str
    trend_strength: float
    last_updated: datetime


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        self.subscriptions: Dict[WebSocket, Set[str]] = defaultdict(set)
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.now(),
            "last_ping": datetime.now(),
            "subscriptions": set()
        }
        logger.info(f"Dashboard WebSocket connected: {self.connection_metadata[websocket]['client_id']}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_id = self.connection_metadata.get(websocket, {}).get("client_id", "unknown")
            if websocket in self.subscriptions:
                del self.subscriptions[websocket]
            del self.connection_metadata[websocket]
            logger.info(f"Dashboard WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str, subscription_type: str = None):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        for websocket in self.active_connections:
            try:
                # Check if client is subscribed to this type
                if subscription_type:
                    client_subscriptions = self.subscriptions.get(websocket, set())
                    if subscription_type not in client_subscriptions:
                        continue
                
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)
    
    def subscribe(self, websocket: WebSocket, subscription_type: str):
        """Subscribe client to specific updates"""
        self.subscriptions[websocket].add(subscription_type)
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscriptions"].add(subscription_type)
    
    def unsubscribe(self, websocket: WebSocket, subscription_type: str):
        """Unsubscribe client from specific updates"""
        self.subscriptions[websocket].discard(subscription_type)
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscriptions"].discard(subscription_type)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about all connections"""
        info = []
        for websocket, metadata in self.connection_metadata.items():
            info.append({
                "client_id": metadata["client_id"],
                "connected_at": metadata["connected_at"].isoformat(),
                "last_ping": metadata["last_ping"].isoformat(),
                "subscriptions": list(metadata["subscriptions"])
            })
        return info


class RealtimeDashboard:
    """Real-time dashboard for AI history analyzer"""
    
    def __init__(self):
        self.app = FastAPI(title="AI History Analyzer Dashboard")
        self.connection_manager = ConnectionManager()
        
        # Global instances
        self.analyzer: Optional[AIHistoryAnalyzer] = None
        self.config = None
        self.integration_system = None
        
        # Performance tracking
        self.performance_snapshots: Dict[str, ModelPerformanceSnapshot] = {}
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Background tasks
        self.metrics_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def dashboard():
            """Serve the main dashboard"""
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.connection_manager.connect(websocket)
            try:
                while True:
                    # Wait for client messages
                    data = await websocket.receive_text()
                    await self._handle_websocket_message(websocket, data)
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
        
        @self.app.get("/api/metrics")
        async def get_current_metrics():
            """Get current system metrics"""
            if not self.analyzer:
                raise HTTPException(status_code=503, detail="Analyzer not initialized")
            
            metrics = self._collect_dashboard_metrics()
            return {
                "timestamp": metrics.timestamp.isoformat(),
                "active_models": metrics.active_models,
                "total_measurements": metrics.total_measurements,
                "average_quality": metrics.average_quality,
                "average_response_time": metrics.average_response_time,
                "cost_efficiency": metrics.cost_efficiency,
                "top_performing_model": metrics.top_performing_model,
                "alerts_count": metrics.alerts_count,
                "trends_analyzed": metrics.trends_analyzed
            }
        
        @self.app.get("/api/models/performance")
        async def get_models_performance():
            """Get current performance of all models"""
            if not self.analyzer:
                raise HTTPException(status_code=503, detail="Analyzer not initialized")
            
            snapshots = []
            for model_name, snapshot in self.performance_snapshots.items():
                snapshots.append({
                    "model_name": snapshot.model_name,
                    "quality_score": snapshot.quality_score,
                    "response_time": snapshot.response_time,
                    "cost_efficiency": snapshot.cost_efficiency,
                    "token_efficiency": snapshot.token_efficiency,
                    "trend_direction": snapshot.trend_direction,
                    "trend_strength": snapshot.trend_strength,
                    "last_updated": snapshot.last_updated.isoformat()
                })
            
            return {"models": snapshots}
        
        @self.app.get("/api/trends/{model_name}")
        async def get_model_trends(model_name: str, days: int = 30):
            """Get trend data for a specific model"""
            if not self.analyzer:
                raise HTTPException(status_code=503, detail="Analyzer not initialized")
            
            trends = {}
            for metric in PerformanceMetric:
                trend_analysis = self.analyzer.analyze_trends(model_name, metric, days)
                if trend_analysis:
                    trends[metric.value] = {
                        "direction": trend_analysis.trend_direction,
                        "strength": trend_analysis.trend_strength,
                        "confidence": trend_analysis.confidence,
                        "forecast": [
                            {"date": date.isoformat(), "value": value}
                            for date, value in trend_analysis.forecast
                        ],
                        "anomalies": [
                            {"date": date.isoformat(), "value": value}
                            for date, value in trend_analysis.anomalies
                        ]
                    }
            
            return {"model_name": model_name, "trends": trends}
        
        @self.app.get("/api/comparisons")
        async def get_recent_comparisons(limit: int = 10):
            """Get recent model comparisons"""
            if not self.analyzer:
                raise HTTPException(status_code=503, detail="Analyzer not initialized")
            
            recent_comparisons = self.analyzer.model_comparisons[-limit:]
            comparisons_data = []
            
            for comparison in recent_comparisons:
                comparisons_data.append({
                    "model_a": comparison.model_a,
                    "model_b": comparison.model_b,
                    "metric": comparison.metric.value,
                    "comparison_score": comparison.comparison_score,
                    "confidence": comparison.confidence,
                    "sample_size": comparison.sample_size,
                    "timestamp": comparison.timestamp.isoformat(),
                    "winner": comparison.model_a if comparison.comparison_score > 0 else comparison.model_b
                })
            
            return {"comparisons": comparisons_data}
        
        @self.app.get("/api/alerts")
        async def get_active_alerts():
            """Get active performance alerts"""
            if not self.integration_system:
                raise HTTPException(status_code=503, detail="Integration system not initialized")
            
            insights = self.integration_system.get_performance_insights()
            alerts = []
            
            for insight in insights:
                if insight.type in ["warning", "alert"]:
                    alerts.append({
                        "type": insight.type,
                        "severity": insight.severity,
                        "message": insight.message,
                        "model_name": insight.model_name,
                        "metric": insight.metric,
                        "current_value": insight.current_value,
                        "recommended_value": insight.recommended_value,
                        "confidence": insight.confidence,
                        "timestamp": insight.timestamp.isoformat()
                    })
            
            return {"alerts": alerts}
        
        @self.app.get("/api/rankings/{metric}")
        async def get_model_rankings(metric: str, days: int = 30):
            """Get model rankings for a specific metric"""
            if not self.analyzer:
                raise HTTPException(status_code=503, detail="Analyzer not initialized")
            
            try:
                metric_enum = PerformanceMetric(metric)
                rankings = self.analyzer.get_model_rankings(metric_enum, days)
                
                return {
                    "metric": metric,
                    "days": days,
                    "rankings": rankings
                }
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}")
        
        @self.app.get("/api/connections")
        async def get_connections():
            """Get WebSocket connection information"""
            return {
                "active_connections": self.connection_manager.get_connection_count(),
                "connections": self.connection_manager.get_connection_info()
            }
    
    async def _handle_websocket_message(self, websocket: WebSocket, message: str):
        """Handle WebSocket messages from clients"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                # Subscribe to specific update types
                subscription_types = data.get("subscriptions", [])
                for sub_type in subscription_types:
                    self.connection_manager.subscribe(websocket, sub_type)
                
                await self.connection_manager.send_personal_message(
                    json.dumps({"type": "subscription_confirmed", "subscriptions": subscription_types}),
                    websocket
                )
            
            elif message_type == "unsubscribe":
                # Unsubscribe from specific update types
                subscription_types = data.get("subscriptions", [])
                for sub_type in subscription_types:
                    self.connection_manager.unsubscribe(websocket, sub_type)
                
                await self.connection_manager.send_personal_message(
                    json.dumps({"type": "unsubscription_confirmed", "subscriptions": subscription_types}),
                    websocket
                )
            
            elif message_type == "ping":
                # Update last ping time
                if websocket in self.connection_manager.connection_metadata:
                    self.connection_manager.connection_metadata[websocket]["last_ping"] = datetime.now()
                
                await self.connection_manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            
            elif message_type == "request_data":
                # Send current data
                data_type = data.get("data_type")
                if data_type == "metrics":
                    metrics = self._collect_dashboard_metrics()
                    await self.connection_manager.send_personal_message(
                        json.dumps({"type": "metrics_update", "data": asdict(metrics)}),
                        websocket
                    )
                elif data_type == "models":
                    await self.connection_manager.send_personal_message(
                        json.dumps({"type": "models_update", "data": list(self.performance_snapshots.values())}),
                        websocket
                    )
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON in WebSocket message")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
    
    def _collect_dashboard_metrics(self) -> DashboardMetrics:
        """Collect current dashboard metrics"""
        try:
            if not self.analyzer:
                return DashboardMetrics(
                    timestamp=datetime.now(),
                    active_models=0,
                    total_measurements=0,
                    average_quality=0.0,
                    average_response_time=0.0,
                    cost_efficiency=0.0,
                    top_performing_model="N/A",
                    alerts_count=0,
                    trends_analyzed=0
                )
            
            stats = self.analyzer.performance_stats
            
            # Calculate average metrics
            quality_scores = []
            response_times = []
            cost_efficiencies = []
            
            for model_name in stats["models_tracked"]:
                summary = self.analyzer.get_performance_summary(model_name, days=7)
                if summary and "metrics" in summary:
                    if "quality_score" in summary["metrics"]:
                        quality_scores.append(summary["metrics"]["quality_score"]["mean"])
                    if "response_time" in summary["metrics"]:
                        response_times.append(summary["metrics"]["response_time"]["mean"])
                    if "cost_efficiency" in summary["metrics"]:
                        cost_efficiencies.append(summary["metrics"]["cost_efficiency"]["mean"])
            
            # Find top performing model
            top_model = "N/A"
            if quality_scores:
                best_quality = max(quality_scores)
                for model_name in stats["models_tracked"]:
                    summary = self.analyzer.get_performance_summary(model_name, days=7)
                    if (summary and "metrics" in summary and 
                        "quality_score" in summary["metrics"] and
                        summary["metrics"]["quality_score"]["mean"] == best_quality):
                        top_model = model_name
                        break
            
            # Get alerts count
            alerts_count = 0
            if self.integration_system:
                insights = self.integration_system.get_performance_insights()
                alerts_count = len([i for i in insights if i.type in ["warning", "alert"]])
            
            return DashboardMetrics(
                timestamp=datetime.now(),
                active_models=len(stats["models_tracked"]),
                total_measurements=stats["total_measurements"],
                average_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                average_response_time=sum(response_times) / len(response_times) if response_times else 0.0,
                cost_efficiency=sum(cost_efficiencies) / len(cost_efficiencies) if cost_efficiencies else 0.0,
                top_performing_model=top_model,
                alerts_count=alerts_count,
                trends_analyzed=len(self.analyzer.trend_analyses)
            )
            
        except Exception as e:
            logger.error(f"Error collecting dashboard metrics: {str(e)}")
            return DashboardMetrics(
                timestamp=datetime.now(),
                active_models=0,
                total_measurements=0,
                average_quality=0.0,
                average_response_time=0.0,
                cost_efficiency=0.0,
                top_performing_model="N/A",
                alerts_count=0,
                trends_analyzed=0
            )
    
    async def _update_performance_snapshots(self):
        """Update performance snapshots for all models"""
        try:
            if not self.analyzer:
                return
            
            stats = self.analyzer.performance_stats
            
            for model_name in stats["models_tracked"]:
                summary = self.analyzer.get_performance_summary(model_name, days=7)
                
                if not summary or "metrics" not in summary:
                    continue
                
                # Get trend analysis
                trend_direction = "stable"
                trend_strength = 0.0
                
                if "quality_score" in summary["metrics"]:
                    trend_analysis = self.analyzer.analyze_trends(
                        model_name, PerformanceMetric.QUALITY_SCORE, days=30
                    )
                    if trend_analysis:
                        trend_direction = trend_analysis.trend_direction
                        trend_strength = trend_analysis.trend_strength
                
                snapshot = ModelPerformanceSnapshot(
                    model_name=model_name,
                    quality_score=summary["metrics"].get("quality_score", {}).get("mean", 0.0),
                    response_time=summary["metrics"].get("response_time", {}).get("mean", 0.0),
                    cost_efficiency=summary["metrics"].get("cost_efficiency", {}).get("mean", 0.0),
                    token_efficiency=summary["metrics"].get("token_efficiency", {}).get("mean", 0.0),
                    trend_direction=trend_direction,
                    trend_strength=trend_strength,
                    last_updated=datetime.now()
                )
                
                self.performance_snapshots[model_name] = snapshot
            
        except Exception as e:
            logger.error(f"Error updating performance snapshots: {str(e)}")
    
    async def _metrics_broadcast_loop(self):
        """Background task to collect and broadcast metrics"""
        while True:
            try:
                if self.analyzer:
                    # Update performance snapshots
                    await self._update_performance_snapshots()
                    
                    # Collect metrics
                    metrics = self._collect_dashboard_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Broadcast to all connected clients
                    message = json.dumps({
                        "type": "metrics_update",
                        "data": {
                            "timestamp": metrics.timestamp.isoformat(),
                            "active_models": metrics.active_models,
                            "total_measurements": metrics.total_measurements,
                            "average_quality": metrics.average_quality,
                            "average_response_time": metrics.average_response_time,
                            "cost_efficiency": metrics.cost_efficiency,
                            "top_performing_model": metrics.top_performing_model,
                            "alerts_count": metrics.alerts_count,
                            "trends_analyzed": metrics.trends_analyzed
                        }
                    })
                    
                    await self.connection_manager.broadcast(message, "metrics")
                    
                    # Broadcast model performance updates
                    models_message = json.dumps({
                        "type": "models_update",
                        "data": [
                            {
                                "model_name": snapshot.model_name,
                                "quality_score": snapshot.quality_score,
                                "response_time": snapshot.response_time,
                                "cost_efficiency": snapshot.cost_efficiency,
                                "token_efficiency": snapshot.token_efficiency,
                                "trend_direction": snapshot.trend_direction,
                                "trend_strength": snapshot.trend_strength,
                                "last_updated": snapshot.last_updated.isoformat()
                            }
                            for snapshot in self.performance_snapshots.values()
                        ]
                    })
                    
                    await self.connection_manager.broadcast(models_message, "models")
                
                # Wait before next update
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics broadcast loop: {str(e)}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _cleanup_loop(self):
        """Background task for cleanup operations"""
        while True:
            try:
                # Clean up old metrics history
                if len(self.metrics_history) > 1000:
                    # Keep only the most recent 1000 metrics
                    recent_metrics = list(self.metrics_history)[-1000:]
                    self.metrics_history.clear()
                    self.metrics_history.extend(recent_metrics)
                
                # Wait 1 hour before next cleanup
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(3600)
    
    async def initialize(self):
        """Initialize the dashboard with analyzer instances"""
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        self.integration_system = get_integration_system()
        
        # Start background tasks
        self.metrics_task = asyncio.create_task(self._metrics_broadcast_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Real-time dashboard initialized")
    
    async def shutdown(self):
        """Shutdown the dashboard"""
        if self.metrics_task:
            self.metrics_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        logger.info("Real-time dashboard shutdown")
    
    def _get_dashboard_html(self) -> str:
        """Get the HTML for the real-time dashboard"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI History Analyzer - Real-time Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1400px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
                .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
                .metric-label { color: #7f8c8d; margin-top: 5px; font-size: 0.9em; }
                .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
                .models-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .model-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .model-name { font-size: 1.2em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
                .model-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
                .model-metric { text-align: center; }
                .model-metric-value { font-size: 1.1em; font-weight: bold; }
                .model-metric-label { font-size: 0.8em; color: #7f8c8d; }
                .trend-indicator { display: inline-block; margin-left: 10px; }
                .trend-up { color: #27ae60; }
                .trend-down { color: #e74c3c; }
                .trend-stable { color: #f39c12; }
                .alerts-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
                .alert.warning { background: #fff3cd; border-left: 4px solid #ffc107; }
                .alert.error { background: #f8d7da; border-left: 4px solid #dc3545; }
                .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }
                .status-connected { background: #4caf50; }
                .status-disconnected { background: #f44336; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AI History Analyzer - Real-time Dashboard</h1>
                    <p>Live monitoring of AI model performance and trends</p>
                    <div>
                        <span class="status-indicator" id="connectionStatus"></span>
                        <span id="connectionText">Connecting...</span>
                    </div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="activeModels">-</div>
                        <div class="metric-label">Active Models</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="totalMeasurements">-</div>
                        <div class="metric-label">Total Measurements</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="averageQuality">-</div>
                        <div class="metric-label">Average Quality</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="averageResponseTime">-</div>
                        <div class="metric-label">Avg Response Time (s)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="costEfficiency">-</div>
                        <div class="metric-label">Cost Efficiency</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="topModel">-</div>
                        <div class="metric-label">Top Performing Model</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="alertsCount">-</div>
                        <div class="metric-label">Active Alerts</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="trendsAnalyzed">-</div>
                        <div class="metric-label">Trends Analyzed</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>Performance Trends</h3>
                    <canvas id="performanceChart" width="400" height="200"></canvas>
                </div>
                
                <div class="models-grid" id="modelsGrid">
                    <p>Loading model performance data...</p>
                </div>
                
                <div class="alerts-container">
                    <h3>Active Alerts</h3>
                    <div id="alertsList">
                        <p>Loading alerts...</p>
                    </div>
                </div>
            </div>
            
            <script>
                let ws;
                let performanceChart;
                let metricsHistory = [];
                
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function() {
                        console.log('WebSocket connected');
                        updateConnectionStatus(true);
                        
                        // Subscribe to updates
                        ws.send(JSON.stringify({
                            type: 'subscribe',
                            subscriptions: ['metrics', 'models']
                        }));
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'metrics_update') {
                            updateMetrics(data.data);
                            updateChart(data.data);
                        } else if (data.type === 'models_update') {
                            updateModels(data.data);
                        }
                    };
                    
                    ws.onclose = function() {
                        console.log('WebSocket disconnected');
                        updateConnectionStatus(false);
                        setTimeout(connectWebSocket, 5000);
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                        updateConnectionStatus(false);
                    };
                }
                
                function updateConnectionStatus(connected) {
                    const statusIndicator = document.getElementById('connectionStatus');
                    const statusText = document.getElementById('connectionText');
                    
                    if (connected) {
                        statusIndicator.className = 'status-indicator status-connected';
                        statusText.textContent = 'Connected';
                    } else {
                        statusIndicator.className = 'status-indicator status-disconnected';
                        statusText.textContent = 'Disconnected';
                    }
                }
                
                function updateMetrics(data) {
                    document.getElementById('activeModels').textContent = data.active_models;
                    document.getElementById('totalMeasurements').textContent = data.total_measurements.toLocaleString();
                    document.getElementById('averageQuality').textContent = data.average_quality.toFixed(3);
                    document.getElementById('averageResponseTime').textContent = data.average_response_time.toFixed(2);
                    document.getElementById('costEfficiency').textContent = data.cost_efficiency.toFixed(3);
                    document.getElementById('topModel').textContent = data.top_performing_model;
                    document.getElementById('alertsCount').textContent = data.alerts_count;
                    document.getElementById('trendsAnalyzed').textContent = data.trends_analyzed;
                }
                
                function updateModels(models) {
                    const modelsGrid = document.getElementById('modelsGrid');
                    
                    if (models.length === 0) {
                        modelsGrid.innerHTML = '<p>No model data available</p>';
                        return;
                    }
                    
                    modelsGrid.innerHTML = models.map(model => `
                        <div class="model-card">
                            <div class="model-name">
                                ${model.model_name}
                                <span class="trend-indicator trend-${model.trend_direction}">
                                    ${model.trend_direction === 'improving' ? '↗' : 
                                      model.trend_direction === 'declining' ? '↘' : '→'}
                                </span>
                            </div>
                            <div class="model-metrics">
                                <div class="model-metric">
                                    <div class="model-metric-value">${model.quality_score.toFixed(3)}</div>
                                    <div class="model-metric-label">Quality</div>
                                </div>
                                <div class="model-metric">
                                    <div class="model-metric-value">${model.response_time.toFixed(2)}s</div>
                                    <div class="model-metric-label">Response Time</div>
                                </div>
                                <div class="model-metric">
                                    <div class="model-metric-value">${model.cost_efficiency.toFixed(3)}</div>
                                    <div class="model-metric-label">Cost Efficiency</div>
                                </div>
                                <div class="model-metric">
                                    <div class="model-metric-value">${model.token_efficiency.toFixed(3)}</div>
                                    <div class="model-metric-label">Token Efficiency</div>
                                </div>
                            </div>
                        </div>
                    `).join('');
                }
                
                function updateChart(data) {
                    metricsHistory.push({
                        timestamp: new Date(data.timestamp),
                        averageQuality: data.average_quality,
                        averageResponseTime: data.average_response_time,
                        costEfficiency: data.cost_efficiency
                    });
                    
                    if (metricsHistory.length > 20) {
                        metricsHistory.shift();
                    }
                    
                    if (performanceChart) {
                        performanceChart.update();
                    }
                }
                
                function initializeChart() {
                    const ctx = document.getElementById('performanceChart').getContext('2d');
                    performanceChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [
                                {
                                    label: 'Average Quality',
                                    data: [],
                                    borderColor: 'rgb(75, 192, 192)',
                                    tension: 0.1
                                },
                                {
                                    label: 'Cost Efficiency',
                                    data: [],
                                    borderColor: 'rgb(255, 99, 132)',
                                    tension: 0.1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 1
                                }
                            }
                        }
                    });
                }
                
                function loadAlerts() {
                    fetch('/api/alerts')
                        .then(response => response.json())
                        .then(data => {
                            const alertsList = document.getElementById('alertsList');
                            if (data.alerts.length === 0) {
                                alertsList.innerHTML = '<p>No active alerts</p>';
                                return;
                            }
                            
                            alertsList.innerHTML = data.alerts.map(alert => `
                                <div class="alert ${alert.severity}">
                                    <strong>${alert.severity.toUpperCase()}</strong> - ${alert.message}
                                    <br><small>Model: ${alert.model_name} | ${new Date(alert.timestamp).toLocaleString()}</small>
                                </div>
                            `).join('');
                        })
                        .catch(error => {
                            console.error('Error loading alerts:', error);
                            document.getElementById('alertsList').innerHTML = '<p>Error loading alerts</p>';
                        });
                }
                
                // Initialize dashboard
                document.addEventListener('DOMContentLoaded', function() {
                    initializeChart();
                    connectWebSocket();
                    loadAlerts();
                    
                    // Refresh alerts every 30 seconds
                    setInterval(loadAlerts, 30000);
                });
            </script>
        </body>
        </html>
        """
    
    def run(self, host: str = "0.0.0.0", port: int = 8003):
        """Run the real-time dashboard"""
        uvicorn.run(self.app, host=host, port=port)


# Global dashboard instance
_dashboard: Optional[RealtimeDashboard] = None


def get_realtime_dashboard() -> RealtimeDashboard:
    """Get or create global real-time dashboard"""
    global _dashboard
    if _dashboard is None:
        _dashboard = RealtimeDashboard()
    return _dashboard


async def initialize_dashboard():
    """Initialize the real-time dashboard"""
    dashboard = get_realtime_dashboard()
    await dashboard.initialize()
    return dashboard


if __name__ == "__main__":
    # Example usage
    dashboard = get_realtime_dashboard()
    dashboard.run()

























