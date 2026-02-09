"""
Real-time Analytics Dashboard for Refactored Opus Clip

Advanced analytics dashboard with:
- Real-time metrics visualization
- Performance monitoring
- User behavior analytics
- System health monitoring
- Custom dashboards
- Data export capabilities
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
import asyncio
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import redis
import psutil
import numpy as np
from collections import defaultdict, deque
import pandas as pd
from pathlib import Path

logger = structlog.get_logger("analytics_dashboard")

# Initialize FastAPI app
app = FastAPI(
    title="Opus Clip Analytics Dashboard",
    description="Real-time analytics and monitoring dashboard",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Redis client for real-time data
redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)

# WebSocket connection manager
class DashboardConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.dashboard_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, dashboard_id: str = "default"):
        await websocket.accept()
        self.active_connections.append(websocket)
        if dashboard_id not in self.dashboard_connections:
            self.dashboard_connections[dashboard_id] = []
        self.dashboard_connections[dashboard_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, dashboard_id: str = "default"):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if dashboard_id in self.dashboard_connections:
            if websocket in self.dashboard_connections[dashboard_id]:
                self.dashboard_connections[dashboard_id].remove(websocket)
    
    async def send_to_dashboard(self, message: str, dashboard_id: str = "default"):
        if dashboard_id in self.dashboard_connections:
            for connection in self.dashboard_connections[dashboard_id]:
                try:
                    await connection.send_text(message)
                except:
                    self.disconnect(connection, dashboard_id)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.active_connections.remove(connection)

manager = DashboardConnectionManager()

# Analytics data structures
@dataclass
class MetricData:
    """Metric data structure."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    name: str
    widgets: List[Dict[str, Any]]
    refresh_interval: int = 5  # seconds
    auto_refresh: bool = True

class RealTimeAnalytics:
    """Real-time analytics engine."""
    
    def __init__(self):
        self.logger = structlog.get_logger("real_time_analytics")
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.real_time_data = {}
        self.dashboard_configs = {}
        self.analytics_task = None
        
    async def start_analytics(self):
        """Start real-time analytics collection."""
        if self.analytics_task and not self.analytics_task.done():
            return
        
        self.analytics_task = asyncio.create_task(self._analytics_loop())
        self.logger.info("Real-time analytics started")
    
    async def stop_analytics(self):
        """Stop real-time analytics collection."""
        if self.analytics_task:
            self.analytics_task.cancel()
            try:
                await self.analytics_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Real-time analytics stopped")
    
    async def _analytics_loop(self):
        """Main analytics collection loop."""
        while True:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
                # Process and store metrics
                await self._process_metrics()
                
                # Broadcast to connected clients
                await self._broadcast_metrics()
                
                # Wait for next collection cycle
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Analytics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free = disk.free / (1024**3)  # GB
            
            # Network metrics
            net_io = psutil.net_io_counters()
            network_bytes_sent = net_io.bytes_sent
            network_bytes_recv = net_io.bytes_recv
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            process_cpu = process.cpu_percent()
            
            # Store metrics
            current_time = datetime.now()
            
            self.metrics_history['cpu_percent'].append(MetricData(current_time, cpu_percent))
            self.metrics_history['memory_percent'].append(MetricData(current_time, memory_percent))
            self.metrics_history['memory_available'].append(MetricData(current_time, memory_available))
            self.metrics_history['disk_percent'].append(MetricData(current_time, disk_percent))
            self.metrics_history['disk_free'].append(MetricData(current_time, disk_free))
            self.metrics_history['network_bytes_sent'].append(MetricData(current_time, network_bytes_sent))
            self.metrics_history['network_bytes_recv'].append(MetricData(current_time, network_bytes_recv))
            self.metrics_history['process_memory'].append(MetricData(current_time, process_memory))
            self.metrics_history['process_cpu'].append(MetricData(current_time, process_cpu))
            
            # Update real-time data
            self.real_time_data.update({
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory_percent,
                'memory_available': memory_available,
                'disk_percent': disk_percent,
                'disk_free': disk_free,
                'network_bytes_sent': network_bytes_sent,
                'network_bytes_recv': network_bytes_recv,
                'process_memory': process_memory,
                'process_cpu': process_cpu,
                'timestamp': current_time.isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Get metrics from Redis
            job_stats = redis_client.hgetall("job_stats")
            user_stats = redis_client.hgetall("user_stats")
            performance_stats = redis_client.hgetall("performance_stats")
            
            # Process job statistics
            total_jobs = int(job_stats.get('total_jobs', 0))
            completed_jobs = int(job_stats.get('completed_jobs', 0))
            failed_jobs = int(job_stats.get('failed_jobs', 0))
            running_jobs = int(job_stats.get('running_jobs', 0))
            
            # Process user statistics
            active_users = int(user_stats.get('active_users', 0))
            total_users = int(user_stats.get('total_users', 0))
            new_users_today = int(user_stats.get('new_users_today', 0))
            
            # Process performance statistics
            avg_response_time = float(performance_stats.get('avg_response_time', 0))
            requests_per_minute = float(performance_stats.get('requests_per_minute', 0))
            error_rate = float(performance_stats.get('error_rate', 0))
            
            # Calculate derived metrics
            success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
            job_throughput = completed_jobs / 60 if completed_jobs > 0 else 0  # jobs per minute
            
            current_time = datetime.now()
            
            # Store application metrics
            self.metrics_history['total_jobs'].append(MetricData(current_time, total_jobs))
            self.metrics_history['completed_jobs'].append(MetricData(current_time, completed_jobs))
            self.metrics_history['failed_jobs'].append(MetricData(current_time, failed_jobs))
            self.metrics_history['running_jobs'].append(MetricData(current_time, running_jobs))
            self.metrics_history['active_users'].append(MetricData(current_time, active_users))
            self.metrics_history['success_rate'].append(MetricData(current_time, success_rate))
            self.metrics_history['avg_response_time'].append(MetricData(current_time, avg_response_time))
            self.metrics_history['requests_per_minute'].append(MetricData(current_time, requests_per_minute))
            self.metrics_history['error_rate'].append(MetricData(current_time, error_rate))
            self.metrics_history['job_throughput'].append(MetricData(current_time, job_throughput))
            
            # Update real-time data
            self.real_time_data.update({
                'total_jobs': total_jobs,
                'completed_jobs': completed_jobs,
                'failed_jobs': failed_jobs,
                'running_jobs': running_jobs,
                'active_users': active_users,
                'total_users': total_users,
                'new_users_today': new_users_today,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'requests_per_minute': requests_per_minute,
                'error_rate': error_rate,
                'job_throughput': job_throughput
            })
            
        except Exception as e:
            self.logger.error(f"Application metrics collection failed: {e}")
    
    async def _process_metrics(self):
        """Process and analyze metrics."""
        try:
            # Calculate moving averages
            for metric_name, history in self.metrics_history.items():
                if len(history) > 0:
                    # Calculate 1-minute moving average
                    recent_data = [m.value for m in list(history)[-60:]]
                    if recent_data:
                        moving_avg = np.mean(recent_data)
                        self.real_time_data[f'{metric_name}_1m_avg'] = moving_avg
                    
                    # Calculate 5-minute moving average
                    recent_data = [m.value for m in list(history)[-300:]]
                    if recent_data:
                        moving_avg = np.mean(recent_data)
                        self.real_time_data[f'{metric_name}_5m_avg'] = moving_avg
            
            # Calculate trends
            await self._calculate_trends()
            
            # Detect anomalies
            await self._detect_anomalies()
            
        except Exception as e:
            self.logger.error(f"Metrics processing failed: {e}")
    
    async def _calculate_trends(self):
        """Calculate metric trends."""
        try:
            for metric_name, history in self.metrics_history.items():
                if len(history) >= 10:
                    recent_values = [m.value for m in list(history)[-10:]]
                    older_values = [m.value for m in list(history)[-20:-10]]
                    
                    if len(recent_values) >= 10 and len(older_values) >= 10:
                        recent_avg = np.mean(recent_values)
                        older_avg = np.mean(older_values)
                        
                        if older_avg != 0:
                            trend = ((recent_avg - older_avg) / older_avg) * 100
                            self.real_time_data[f'{metric_name}_trend'] = trend
                        
        except Exception as e:
            self.logger.error(f"Trend calculation failed: {e}")
    
    async def _detect_anomalies(self):
        """Detect metric anomalies."""
        try:
            for metric_name, history in self.metrics_history.items():
                if len(history) >= 20:
                    values = [m.value for m in list(history)[-20:]]
                    
                    if len(values) >= 20:
                        mean = np.mean(values)
                        std = np.std(values)
                        
                        if std > 0:
                            # Z-score based anomaly detection
                            current_value = values[-1]
                            z_score = abs((current_value - mean) / std)
                            
                            if z_score > 2:  # Threshold for anomaly
                                self.real_time_data[f'{metric_name}_anomaly'] = True
                                self.real_time_data[f'{metric_name}_z_score'] = z_score
                            else:
                                self.real_time_data[f'{metric_name}_anomaly'] = False
                        
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to connected clients."""
        try:
            if self.real_time_data:
                message = json.dumps({
                    "type": "metrics_update",
                    "data": self.real_time_data,
                    "timestamp": datetime.now().isoformat()
                })
                
                await manager.broadcast(message)
                
        except Exception as e:
            self.logger.error(f"Metrics broadcast failed: {e}")
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for specified time period."""
        try:
            if metric_name not in self.metrics_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            history = self.metrics_history[metric_name]
            
            filtered_data = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.value,
                    "metadata": m.metadata
                }
                for m in history
                if m.timestamp >= cutoff_time
            ]
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Failed to get metric history: {e}")
            return []
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get metric summary statistics."""
        try:
            history = self.get_metric_history(metric_name, hours)
            
            if not history:
                return {}
            
            values = [d["value"] for d in history]
            
            return {
                "metric_name": metric_name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "latest": values[-1] if values else None,
                "trend": self.real_time_data.get(f'{metric_name}_trend', 0),
                "anomaly": self.real_time_data.get(f'{metric_name}_anomaly', False)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metric summary: {e}")
            return {}

# Initialize analytics
analytics = RealTimeAnalytics()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Serve the main analytics dashboard."""
    return templates.TemplateResponse("analytics_dashboard.html", {
        "request": request,
        "title": "Opus Clip Analytics Dashboard"
    })

@app.get("/api/metrics/real-time")
async def get_real_time_metrics():
    """Get real-time metrics."""
    return {
        "success": True,
        "data": analytics.real_time_data,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/metrics/history/{metric_name}")
async def get_metric_history(metric_name: str, hours: int = 24):
    """Get metric history."""
    history = analytics.get_metric_history(metric_name, hours)
    return {
        "success": True,
        "metric_name": metric_name,
        "hours": hours,
        "data": history,
        "count": len(history)
    }

@app.get("/api/metrics/summary/{metric_name}")
async def get_metric_summary(metric_name: str, hours: int = 24):
    """Get metric summary."""
    summary = analytics.get_metric_summary(metric_name, hours)
    return {
        "success": True,
        "summary": summary
    }

@app.get("/api/metrics/available")
async def get_available_metrics():
    """Get list of available metrics."""
    available_metrics = list(analytics.metrics_history.keys())
    return {
        "success": True,
        "metrics": available_metrics,
        "count": len(available_metrics)
    }

@app.get("/api/dashboards")
async def get_dashboards():
    """Get available dashboards."""
    dashboards = [
        {
            "id": "system",
            "name": "System Performance",
            "description": "System resource utilization and performance metrics",
            "widgets": ["cpu", "memory", "disk", "network"]
        },
        {
            "id": "application",
            "name": "Application Metrics",
            "description": "Application-specific metrics and KPIs",
            "widgets": ["jobs", "users", "performance", "errors"]
        },
        {
            "id": "custom",
            "name": "Custom Dashboard",
            "description": "User-defined custom dashboard",
            "widgets": []
        }
    ]
    
    return {
        "success": True,
        "dashboards": dashboards
    }

@app.post("/api/dashboards/{dashboard_id}/config")
async def update_dashboard_config(dashboard_id: str, config: DashboardConfig):
    """Update dashboard configuration."""
    analytics.dashboard_configs[dashboard_id] = config
    return {
        "success": True,
        "message": f"Dashboard {dashboard_id} configuration updated"
    }

@app.get("/api/export/metrics")
async def export_metrics(metric_names: str = None, hours: int = 24, format: str = "json"):
    """Export metrics data."""
    try:
        if metric_names:
            metric_list = metric_names.split(",")
        else:
            metric_list = list(analytics.metrics_history.keys())
        
        export_data = {}
        for metric_name in metric_list:
            history = analytics.get_metric_history(metric_name, hours)
            export_data[metric_name] = history
        
        if format == "csv":
            # Convert to CSV format
            df_data = []
            for metric_name, history in export_data.items():
                for point in history:
                    df_data.append({
                        "metric": metric_name,
                        "timestamp": point["timestamp"],
                        "value": point["value"]
                    })
            
            df = pd.DataFrame(df_data)
            csv_content = df.to_csv(index=False)
            
            return JSONResponse(
                content={"success": True, "data": csv_content, "format": "csv"},
                headers={"Content-Type": "text/csv"}
            )
        else:
            return {
                "success": True,
                "data": export_data,
                "format": "json",
                "exported_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.websocket("/ws/analytics")
async def websocket_analytics(websocket: WebSocket, dashboard_id: str = "default"):
    """WebSocket endpoint for real-time analytics."""
    await manager.connect(websocket, dashboard_id)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, dashboard_id)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize analytics on startup."""
    await analytics.start_analytics()
    logger.info("Analytics dashboard started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await analytics.stop_analytics()
    logger.info("Analytics dashboard stopped")

if __name__ == "__main__":
    uvicorn.run(
        "real_time_dashboard:app",
        host="0.0.0.0",
        port=3001,
        reload=True,
        log_level="info"
    )


