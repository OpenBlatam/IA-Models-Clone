#!/usr/bin/env python3
"""
📊 HeyGen AI - Advanced Monitoring & Analytics System V2
=======================================================

Comprehensive monitoring, analytics, and observability system for the HeyGen AI platform.

Author: AI Assistant
Date: December 2024
Version: 2.0.0
"""

import asyncio
import logging
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('heygen_ai_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('heygen_ai_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('heygen_ai_active_connections', 'Number of active connections')
SYSTEM_CPU_USAGE = Gauge('heygen_ai_system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('heygen_ai_system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('heygen_ai_system_disk_usage_percent', 'System disk usage percentage')

class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertLevel(Enum):
    """Alert level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricData:
    """Metric data class"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    """Alert data class"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    active_connections: int
    response_time: float
    throughput: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

# Database setup
Base = declarative_base()

class MetricRecord(Base):
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    labels = Column(Text)  # JSON string
    metric_type = Column(String(50), nullable=False)

class AlertRecord(Base):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    alert_id = Column(String(255), unique=True, nullable=False)
    level = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    metric_name = Column(String(255), nullable=False)
    threshold = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)

class AdvancedMonitoringAnalyticsSystemV2:
    """Advanced Monitoring & Analytics System V2"""
    
    def __init__(self):
        self.name = "Advanced Monitoring & Analytics System V2"
        self.version = "2.0.0"
        
        # Initialize database
        self.db_engine = create_engine('sqlite:///heygen_ai_monitoring.db')
        Base.metadata.create_all(self.db_engine)
        Session = sessionmaker(bind=self.db_engine)
        self.db_session = Session()
        
        # Initialize Redis for real-time data
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            logger.warning("Redis not available, using in-memory storage")
        
        # In-memory storage for real-time data
        self.real_time_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts = {}
        self.alert_history = []
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,
            'error_rate': 5.0
        }
        
        # Monitoring configuration
        self.monitoring_config = {
            'collection_interval': 5,  # seconds
            'retention_days': 30,
            'alert_cooldown': 300,  # seconds
            'max_alerts_per_metric': 10
        }
        
        # Initialize FastAPI app for monitoring dashboard
        self.app = FastAPI(
            title="HeyGen AI Monitoring Dashboard",
            description="Real-time monitoring and analytics dashboard",
            version="2.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Start monitoring
        self._start_monitoring()
        
        # Start Prometheus metrics server
        self._start_prometheus_server()
    
    def _setup_routes(self):
        """Setup monitoring dashboard routes"""
        
        @self.app.get("/")
        async def dashboard():
            """Main dashboard"""
            return HTMLResponse(self._generate_dashboard_html())
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics"""
            return {
                "system_metrics": self._get_system_metrics(),
                "application_metrics": self._get_application_metrics(),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/api/metrics/{metric_name}")
        async def get_metric_history(metric_name: str, hours: int = 24):
            """Get metric history"""
            return self._get_metric_history(metric_name, hours)
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get active alerts"""
            return {
                "active_alerts": list(self.active_alerts.values()),
                "alert_history": self.alert_history[-100:],  # Last 100 alerts
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/api/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve an alert"""
            return self._resolve_alert(alert_id)
        
        @self.app.get("/api/performance")
        async def get_performance_summary():
            """Get performance summary"""
            return self._get_performance_summary()
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "database_connected": True,
                "redis_available": self.redis_available
            }
    
    def _start_monitoring(self):
        """Start monitoring processes"""
        # Start system metrics collection
        system_thread = threading.Thread(target=self._system_monitoring_loop, daemon=True)
        system_thread.start()
        
        # Start alert processing
        alert_thread = threading.Thread(target=self._alert_processing_loop, daemon=True)
        alert_thread.start()
        
        # Start data cleanup
        cleanup_thread = threading.Thread(target=self._data_cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        logger.info("Monitoring processes started")
    
    def _start_prometheus_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(8001)
            logger.info("Prometheus metrics server started on port 8001")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
    
    def _system_monitoring_loop(self):
        """System monitoring loop"""
        while True:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(self.monitoring_config['collection_interval'])
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(10)
    
    def _alert_processing_loop(self):
        """Alert processing loop"""
        while True:
            try:
                # Process alerts
                self._process_alerts()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                time.sleep(60)
    
    def _data_cleanup_loop(self):
        """Data cleanup loop"""
        while True:
            try:
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")
                time.sleep(3600)
    
    def _collect_system_metrics(self) -> List[MetricData]:
        """Collect system metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            metrics.append(MetricData(
                name="system_cpu_usage",
                value=cpu_usage,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE
            ))
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            metrics.append(MetricData(
                name="system_memory_usage",
                value=memory_info.percent,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE
            ))
            
            # Disk usage
            disk_usage = psutil.disk_usage('/').percent
            metrics.append(MetricData(
                name="system_disk_usage",
                value=disk_usage,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE
            ))
            
            # Network usage
            network_io = psutil.net_io_counters()
            metrics.append(MetricData(
                name="system_network_bytes_sent",
                value=network_io.bytes_sent,
                timestamp=timestamp,
                metric_type=MetricType.COUNTER
            ))
            metrics.append(MetricData(
                name="system_network_bytes_recv",
                value=network_io.bytes_recv,
                timestamp=timestamp,
                metric_type=MetricType.COUNTER
            ))
            
            # Process metrics
            process = psutil.Process()
            metrics.append(MetricData(
                name="process_cpu_usage",
                value=process.cpu_percent(),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE
            ))
            metrics.append(MetricData(
                name="process_memory_usage",
                value=process.memory_info().rss / 1024 / 1024,  # MB
                timestamp=timestamp,
                metric_type=MetricType.GAUGE
            ))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _store_metrics(self, metrics: List[MetricData]):
        """Store metrics in database and real-time storage"""
        try:
            # Store in database
            for metric in metrics:
                record = MetricRecord(
                    name=metric.name,
                    value=metric.value,
                    timestamp=metric.timestamp,
                    labels=json.dumps(metric.labels),
                    metric_type=metric.metric_type.value
                )
                self.db_session.add(record)
            
            self.db_session.commit()
            
            # Store in real-time storage
            for metric in metrics:
                self.real_time_metrics[metric.name].append(metric)
                
                # Store in Redis if available
                if self.redis_available:
                    key = f"metric:{metric.name}:{int(metric.timestamp.timestamp())}"
                    self.redis_client.setex(key, 3600, json.dumps({
                        'value': metric.value,
                        'timestamp': metric.timestamp.isoformat(),
                        'labels': metric.labels
                    }))
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
            self.db_session.rollback()
    
    def _update_prometheus_metrics(self, metrics: List[MetricData]):
        """Update Prometheus metrics"""
        try:
            for metric in metrics:
                if metric.name == "system_cpu_usage":
                    SYSTEM_CPU_USAGE.set(metric.value)
                elif metric.name == "system_memory_usage":
                    SYSTEM_MEMORY_USAGE.set(metric.value)
                elif metric.name == "system_disk_usage":
                    SYSTEM_DISK_USAGE.set(metric.value)
                elif metric.name == "system_network_bytes_sent":
                    # Network metrics would need more complex handling
                    pass
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    def _check_alerts(self, metrics: List[MetricData]):
        """Check metrics against thresholds and generate alerts"""
        try:
            for metric in metrics:
                threshold = self.thresholds.get(metric.name)
                if threshold is None:
                    continue
                
                if metric.value > threshold:
                    # Check if alert already exists and is not in cooldown
                    alert_key = f"{metric.name}_{threshold}"
                    if alert_key in self.active_alerts:
                        continue
                    
                    # Create alert
                    alert = Alert(
                        id=alert_key,
                        level=AlertLevel.WARNING if metric.value < threshold * 1.5 else AlertLevel.CRITICAL,
                        message=f"{metric.name} is {metric.value:.2f}, above threshold {threshold}",
                        metric_name=metric.name,
                        threshold=threshold,
                        current_value=metric.value,
                        timestamp=datetime.now()
                    )
                    
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert)
                    
                    logger.warning(f"Alert triggered: {alert.message}")
                    
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _process_alerts(self):
        """Process and manage alerts"""
        try:
            # Check for resolved alerts
            resolved_alerts = []
            for alert_id, alert in self.active_alerts.items():
                # Check if metric is back to normal
                recent_metrics = list(self.real_time_metrics.get(alert.metric_name, []))[-5:]
                if recent_metrics:
                    avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
                    if avg_value <= alert.threshold:
                        alert.resolved = True
                        alert.resolved_at = datetime.now()
                        resolved_alerts.append(alert_id)
                        logger.info(f"Alert resolved: {alert.message}")
            
            # Remove resolved alerts
            for alert_id in resolved_alerts:
                del self.active_alerts[alert_id]
            
        except Exception as e:
            logger.error(f"Error processing alerts: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data"""
        try:
            # Clean up database
            cutoff_date = datetime.now() - timedelta(days=self.monitoring_config['retention_days'])
            
            # Delete old metrics
            self.db_session.query(MetricRecord).filter(
                MetricRecord.timestamp < cutoff_date
            ).delete()
            
            # Delete old alerts
            self.db_session.query(AlertRecord).filter(
                AlertRecord.timestamp < cutoff_date
            ).delete()
            
            self.db_session.commit()
            
            # Clean up in-memory data
            for metric_name, data in self.real_time_metrics.items():
                # Keep only recent data
                cutoff_time = datetime.now() - timedelta(hours=24)
                while data and data[0].timestamp < cutoff_time:
                    data.popleft()
            
            logger.info("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/').percent
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_info.percent,
                "disk_usage": disk_usage,
                "memory_available": memory_info.available,
                "memory_total": memory_info.total,
                "disk_free": psutil.disk_usage('/').free,
                "disk_total": psutil.disk_usage('/').total
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def _get_application_metrics(self) -> Dict[str, Any]:
        """Get application metrics"""
        return {
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "metrics_collected": sum(len(data) for data in self.real_time_metrics.values()),
            "database_connected": True,
            "redis_available": self.redis_available
        }
    
    def _get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Get from database
            records = self.db_session.query(MetricRecord).filter(
                MetricRecord.name == metric_name,
                MetricRecord.timestamp >= cutoff_time
            ).order_by(MetricRecord.timestamp).all()
            
            return [
                {
                    "value": record.value,
                    "timestamp": record.timestamp.isoformat(),
                    "labels": json.loads(record.labels) if record.labels else {}
                }
                for record in records
            ]
        except Exception as e:
            logger.error(f"Error getting metric history: {e}")
            return []
    
    def _resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                
                return {
                    "success": True,
                    "message": f"Alert {alert_id} resolved",
                    "alert": alert.__dict__
                }
            else:
                return {
                    "success": False,
                    "message": f"Alert {alert_id} not found"
                }
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            # Get recent metrics
            recent_metrics = {}
            for metric_name, data in self.real_time_metrics.items():
                if data:
                    recent_metrics[metric_name] = {
                        "current": data[-1].value,
                        "average": sum(m.value for m in data) / len(data),
                        "min": min(m.value for m in data),
                        "max": max(m.value for m in data)
                    }
            
            return {
                "recent_metrics": recent_metrics,
                "active_alerts": len(self.active_alerts),
                "system_health": self._calculate_system_health(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health"""
        try:
            # Get current metrics
            system_metrics = self._get_system_metrics()
            
            # Check thresholds
            health_score = 100
            if system_metrics.get('cpu_usage', 0) > 80:
                health_score -= 20
            if system_metrics.get('memory_usage', 0) > 85:
                health_score -= 20
            if system_metrics.get('disk_usage', 0) > 90:
                health_score -= 20
            if len(self.active_alerts) > 5:
                health_score -= 20
            
            if health_score >= 80:
                return "healthy"
            elif health_score >= 60:
                return "warning"
            else:
                return "critical"
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return "unknown"
    
    def _generate_dashboard_html(self) -> str:
        """Generate monitoring dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>HeyGen AI Monitoring Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }
                .metric-value { font-size: 24px; font-weight: bold; }
                .alert { background-color: #ffebee; border-left: 4px solid #f44336; }
                .warning { background-color: #fff3e0; border-left: 4px solid #ff9800; }
                .healthy { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
            </style>
        </head>
        <body>
            <h1>HeyGen AI Monitoring Dashboard</h1>
            <div id="metrics"></div>
            <div id="alerts"></div>
            <script>
                async function loadMetrics() {
                    const response = await fetch('/api/metrics');
                    const data = await response.json();
                    
                    const metricsDiv = document.getElementById('metrics');
                    metricsDiv.innerHTML = `
                        <div class="metric-card">
                            <h3>CPU Usage</h3>
                            <div class="metric-value">${data.system_metrics.cpu_usage.toFixed(1)}%</div>
                        </div>
                        <div class="metric-card">
                            <h3>Memory Usage</h3>
                            <div class="metric-value">${data.system_metrics.memory_usage.toFixed(1)}%</div>
                        </div>
                        <div class="metric-card">
                            <h3>Disk Usage</h3>
                            <div class="metric-value">${data.system_metrics.disk_usage.toFixed(1)}%</div>
                        </div>
                    `;
                }
                
                async function loadAlerts() {
                    const response = await fetch('/api/alerts');
                    const data = await response.json();
                    
                    const alertsDiv = document.getElementById('alerts');
                    if (data.active_alerts.length > 0) {
                        alertsDiv.innerHTML = '<h2>Active Alerts</h2>' + 
                            data.active_alerts.map(alert => 
                                `<div class="alert metric-card">
                                    <strong>${alert.level.toUpperCase()}</strong>: ${alert.message}
                                    <button onclick="resolveAlert('${alert.id}')">Resolve</button>
                                </div>`
                            ).join('');
                    } else {
                        alertsDiv.innerHTML = '<h2>No Active Alerts</h2><div class="healthy metric-card">System is healthy</div>';
                    }
                }
                
                async function resolveAlert(alertId) {
                    await fetch(`/api/alerts/${alertId}/resolve`, {method: 'POST'});
                    loadAlerts();
                }
                
                // Load data every 5 seconds
                setInterval(() => {
                    loadMetrics();
                    loadAlerts();
                }, 5000);
                
                // Initial load
                loadMetrics();
                loadAlerts();
            </script>
        </body>
        </html>
        """
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8002):
        """Run the monitoring server"""
        logger.info(f"Starting {self.name} server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

# Global monitoring instance
monitoring = AdvancedMonitoringAnalyticsSystemV2()

# Convenience functions
def run_monitoring_server(host: str = "0.0.0.0", port: int = 8002):
    """Run the monitoring server"""
    monitoring.run_server(host, port)

def get_app():
    """Get the FastAPI app instance"""
    return monitoring.app

# Example usage and testing
async def main():
    """Main function for testing the monitoring system"""
    try:
        print("📊 HeyGen AI - Advanced Monitoring & Analytics System V2")
        print("=" * 70)
        
        # Wait a bit for metrics to be collected
        print("Collecting initial metrics...")
        await asyncio.sleep(10)
        
        # Get system metrics
        print("\n📊 System Metrics:")
        system_metrics = monitoring._get_system_metrics()
        for metric, value in system_metrics.items():
            print(f"  {metric}: {value}")
        
        # Get application metrics
        print("\n📈 Application Metrics:")
        app_metrics = monitoring._get_application_metrics()
        for metric, value in app_metrics.items():
            print(f"  {metric}: {value}")
        
        # Get performance summary
        print("\n🎯 Performance Summary:")
        performance = monitoring._get_performance_summary()
        print(f"  System Health: {performance.get('system_health', 'unknown')}")
        print(f"  Active Alerts: {performance.get('active_alerts', 0)}")
        
        # Show recent metrics
        print("\n📊 Recent Metrics:")
        for metric_name, data in monitoring.real_time_metrics.items():
            if data:
                recent_value = data[-1].value
                print(f"  {metric_name}: {recent_value:.2f}")
        
        print("\n✅ Monitoring system is running!")
        print("Dashboard available at: http://localhost:8002")
        print("Prometheus metrics at: http://localhost:8001")
        
    except Exception as e:
        logger.error(f"Monitoring system test failed: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
    
    # Uncomment to run the server
    # run_monitoring_server()



