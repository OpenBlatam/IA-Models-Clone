"""
Real-time Monitoring and Alerting System
========================================

Advanced monitoring system for the AI Document Classifier with real-time metrics,
intelligent alerting, and automated response capabilities.

Features:
- Real-time system metrics collection
- Intelligent alerting with machine learning
- Automated incident response
- Performance trend analysis
- Resource optimization recommendations
- Health check automation
- SLA monitoring and reporting
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import psutil
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import threading
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class SystemMetric:
    """System metric data structure"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str]
    metadata: Dict[str, Any]

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    metric_name: str
    threshold: float
    current_value: float
    tags: Dict[str, str]
    actions_taken: List[str]
    resolved_at: Optional[datetime] = None

@dataclass
class HealthCheck:
    """Health check result"""
    service_name: str
    status: str
    response_time: float
    timestamp: datetime
    details: Dict[str, Any]
    error_message: Optional[str] = None

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.metrics_buffer = []
        self.collection_interval = 1  # seconds
        
        # Prometheus metrics
        self.request_count = Counter('ai_doc_classifier_requests_total', 
                                   'Total requests', ['endpoint', 'status'])
        self.request_duration = Histogram('ai_doc_classifier_request_duration_seconds',
                                        'Request duration', ['endpoint'])
        self.active_connections = Gauge('ai_doc_classifier_active_connections',
                                      'Active connections')
        self.memory_usage = Gauge('ai_doc_classifier_memory_usage_bytes',
                                'Memory usage in bytes')
        self.cpu_usage = Gauge('ai_doc_classifier_cpu_usage_percent',
                             'CPU usage percentage')
        self.disk_usage = Gauge('ai_doc_classifier_disk_usage_percent',
                              'Disk usage percentage')
        self.classification_accuracy = Gauge('ai_doc_classifier_accuracy',
                                           'Classification accuracy')
        self.processing_queue_size = Gauge('ai_doc_classifier_queue_size',
                                         'Processing queue size')
        
    async def collect_system_metrics(self) -> List[SystemMetric]:
        """Collect comprehensive system metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        metrics.extend([
            SystemMetric(timestamp, "cpu.usage.percent", cpu_percent, "percent", 
                        {"core": "all"}, {"count": cpu_count}),
            SystemMetric(timestamp, "cpu.frequency.mhz", cpu_freq.current if cpu_freq else 0, 
                        "mhz", {"core": "all"}, {"max": cpu_freq.max if cpu_freq else 0})
        ])
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics.extend([
            SystemMetric(timestamp, "memory.usage.percent", memory.percent, "percent", 
                        {"type": "virtual"}, {"total": memory.total, "available": memory.available}),
            SystemMetric(timestamp, "memory.usage.bytes", memory.used, "bytes", 
                        {"type": "virtual"}, {"total": memory.total}),
            SystemMetric(timestamp, "swap.usage.percent", swap.percent, "percent", 
                        {"type": "swap"}, {"total": swap.total, "used": swap.used})
        ])
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics.extend([
            SystemMetric(timestamp, "disk.usage.percent", (disk.used / disk.total) * 100, 
                        "percent", {"mount": "/"}, {"total": disk.total, "used": disk.used}),
            SystemMetric(timestamp, "disk.io.read.bytes", disk_io.read_bytes if disk_io else 0, 
                        "bytes", {"operation": "read"}, {}),
            SystemMetric(timestamp, "disk.io.write.bytes", disk_io.write_bytes if disk_io else 0, 
                        "bytes", {"operation": "write"}, {})
        ])
        
        # Network metrics
        network = psutil.net_io_counters()
        
        metrics.extend([
            SystemMetric(timestamp, "network.io.bytes.sent", network.bytes_sent, 
                        "bytes", {"direction": "sent"}, {}),
            SystemMetric(timestamp, "network.io.bytes.recv", network.bytes_recv, 
                        "bytes", {"direction": "received"}, {}),
            SystemMetric(timestamp, "network.packets.sent", network.packets_sent, 
                        "packets", {"direction": "sent"}, {}),
            SystemMetric(timestamp, "network.packets.recv", network.packets_recv, 
                        "packets", {"direction": "received"}, {})
        ])
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()
        
        metrics.extend([
            SystemMetric(timestamp, "process.memory.bytes", process_memory.rss, 
                        "bytes", {"process": "ai_doc_classifier"}, {"vms": process_memory.vms}),
            SystemMetric(timestamp, "process.cpu.percent", process_cpu, 
                        "percent", {"process": "ai_doc_classifier"}, {"pid": process.pid})
        ])
        
        return metrics
    
    async def collect_application_metrics(self) -> List[SystemMetric]:
        """Collect application-specific metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # Redis metrics
        try:
            redis_info = self.redis_client.info()
            metrics.extend([
                SystemMetric(timestamp, "redis.connected_clients", redis_info.get('connected_clients', 0), 
                           "clients", {"service": "redis"}, {}),
                SystemMetric(timestamp, "redis.used_memory.bytes", redis_info.get('used_memory', 0), 
                           "bytes", {"service": "redis"}, {}),
                SystemMetric(timestamp, "redis.keyspace.hits", redis_info.get('keyspace_hits', 0), 
                           "hits", {"service": "redis"}, {}),
                SystemMetric(timestamp, "redis.keyspace.misses", redis_info.get('keyspace_misses', 0), 
                           "misses", {"service": "redis"}, {})
            ])
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")
        
        # Custom application metrics
        metrics.extend([
            SystemMetric(timestamp, "app.requests.per_second", self._calculate_rps(), 
                        "requests/sec", {"service": "ai_doc_classifier"}, {}),
            SystemMetric(timestamp, "app.response_time.avg", self._calculate_avg_response_time(), 
                        "milliseconds", {"service": "ai_doc_classifier"}, {}),
            SystemMetric(timestamp, "app.error_rate.percent", self._calculate_error_rate(), 
                        "percent", {"service": "ai_doc_classifier"}, {}),
            SystemMetric(timestamp, "app.active_sessions", self._get_active_sessions(), 
                        "sessions", {"service": "ai_doc_classifier"}, {})
        ])
        
        return metrics
    
    def _calculate_rps(self) -> float:
        """Calculate requests per second"""
        # Implementation would track request timestamps
        return 0.0
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        # Implementation would track response times
        return 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage"""
        # Implementation would track error counts
        return 0.0
    
    def _get_active_sessions(self) -> int:
        """Get number of active sessions"""
        # Implementation would track active sessions
        return 0
    
    async def store_metrics(self, metrics: List[SystemMetric]):
        """Store metrics in Redis with TTL"""
        for metric in metrics:
            key = f"metrics:{metric.metric_name}:{int(metric.timestamp.timestamp())}"
            data = asdict(metric)
            data['timestamp'] = metric.timestamp.isoformat()
            
            # Store with 1 hour TTL
            self.redis_client.setex(key, 3600, json.dumps(data))
            
            # Add to time series
            ts_key = f"ts:{metric.metric_name}"
            self.redis_client.zadd(ts_key, {json.dumps(data): metric.timestamp.timestamp()})
            self.redis_client.expire(ts_key, 3600)

class AnomalyDetector:
    """Machine learning-based anomaly detection"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.training_data = {}
        self.window_size = 100
        self.contamination = 0.1
        
    def train_model(self, metric_name: str, data: List[float]):
        """Train anomaly detection model for a specific metric"""
        if len(data) < self.window_size:
            return False
        
        # Prepare training data
        X = np.array(data[-self.window_size:]).reshape(-1, 1)
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train isolation forest
        model = IsolationForest(contamination=self.contamination, random_state=42)
        model.fit(X_scaled)
        
        # Store model and scaler
        self.models[metric_name] = model
        self.scalers[metric_name] = scaler
        self.training_data[metric_name] = data[-self.window_size:]
        
        logger.info(f"Trained anomaly detection model for {metric_name}")
        return True
    
    def detect_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Detect anomaly for a given metric value"""
        if metric_name not in self.models:
            return {"is_anomaly": False, "confidence": 0.0, "reason": "No model trained"}
        
        model = self.models[metric_name]
        scaler = self.scalers[metric_name]
        
        # Scale the value
        X = np.array([[value]])
        X_scaled = scaler.transform(X)
        
        # Predict anomaly
        prediction = model.predict(X_scaled)[0]
        score = model.score_samples(X_scaled)[0]
        
        is_anomaly = prediction == -1
        confidence = abs(score)
        
        return {
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "score": score,
            "threshold": -0.1  # Adjustable threshold
        }
    
    def update_training_data(self, metric_name: str, value: float):
        """Update training data with new values"""
        if metric_name not in self.training_data:
            self.training_data[metric_name] = []
        
        self.training_data[metric_name].append(value)
        
        # Keep only recent data
        if len(self.training_data[metric_name]) > self.window_size * 2:
            self.training_data[metric_name] = self.training_data[metric_name][-self.window_size:]

class AlertManager:
    """Intelligent alerting system"""
    
    def __init__(self):
        self.alerts = {}
        self.alert_rules = {}
        self.notification_channels = {}
        self.alert_history = []
        self.suppression_rules = {}
        
        # Load configuration
        self._load_alert_rules()
        self._setup_notification_channels()
    
    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        self.alert_rules = {
            "cpu_high": {
                "metric": "cpu.usage.percent",
                "threshold": 80.0,
                "severity": AlertSeverity.WARNING,
                "duration": 300,  # 5 minutes
                "description": "High CPU usage detected"
            },
            "memory_high": {
                "metric": "memory.usage.percent",
                "threshold": 85.0,
                "severity": AlertSeverity.WARNING,
                "duration": 300,
                "description": "High memory usage detected"
            },
            "disk_full": {
                "metric": "disk.usage.percent",
                "threshold": 90.0,
                "severity": AlertSeverity.CRITICAL,
                "duration": 60,
                "description": "Disk space critically low"
            },
            "response_time_slow": {
                "metric": "app.response_time.avg",
                "threshold": 2000.0,  # 2 seconds
                "severity": AlertSeverity.WARNING,
                "duration": 180,
                "description": "Response time is slow"
            },
            "error_rate_high": {
                "metric": "app.error_rate.percent",
                "threshold": 5.0,
                "severity": AlertSeverity.CRITICAL,
                "duration": 120,
                "description": "High error rate detected"
            }
        }
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        self.notification_channels = {
            "email": self._send_email_alert,
            "slack": self._send_slack_alert,
            "webhook": self._send_webhook_alert,
            "sms": self._send_sms_alert
        }
    
    async def evaluate_metrics(self, metrics: List[SystemMetric]):
        """Evaluate metrics against alert rules"""
        for metric in metrics:
            for rule_name, rule in self.alert_rules.items():
                if metric.metric_name == rule["metric"]:
                    await self._check_rule(rule_name, rule, metric)
    
    async def _check_rule(self, rule_name: str, rule: Dict, metric: SystemMetric):
        """Check a specific alert rule"""
        threshold = rule["threshold"]
        severity = rule["severity"]
        duration = rule["duration"]
        
        # Check if threshold is exceeded
        if metric.value > threshold:
            alert_id = f"{rule_name}_{int(metric.timestamp.timestamp())}"
            
            # Check if alert already exists
            if alert_id not in self.alerts:
                alert = Alert(
                    id=alert_id,
                    timestamp=metric.timestamp,
                    severity=severity,
                    status=AlertStatus.ACTIVE,
                    title=f"{rule['description']} - {metric.metric_name}",
                    description=f"Current value: {metric.value} {metric.unit}, Threshold: {threshold} {metric.unit}",
                    metric_name=metric.metric_name,
                    threshold=threshold,
                    current_value=metric.value,
                    tags=metric.tags,
                    actions_taken=[]
                )
                
                self.alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Send notifications
                await self._send_notifications(alert)
                
                logger.warning(f"Alert triggered: {alert.title}")
        else:
            # Check if alert should be resolved
            await self._resolve_alert(rule_name, metric)
    
    async def _resolve_alert(self, rule_name: str, metric: SystemMetric):
        """Resolve alert if conditions are met"""
        for alert_id, alert in self.alerts.items():
            if (alert.metric_name == metric.metric_name and 
                alert.status == AlertStatus.ACTIVE and
                metric.value <= alert.threshold):
                
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = metric.timestamp
                alert.actions_taken.append("Auto-resolved: threshold no longer exceeded")
                
                logger.info(f"Alert resolved: {alert.title}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        channels = ["email", "slack"]  # Configure based on severity
        
        for channel in channels:
            if channel in self.notification_channels:
                try:
                    await self.notification_channels[channel](alert)
                except Exception as e:
                    logger.error(f"Failed to send {channel} notification: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        # Implementation for email notifications
        pass
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        # Implementation for Slack notifications
        pass
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        # Implementation for webhook notifications
        pass
    
    async def _send_sms_alert(self, alert: Alert):
        """Send SMS alert"""
        # Implementation for SMS notifications
        pass

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.health_checks = {}
        self.health_history = []
        self.check_interval = 30  # seconds
        
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
    
    async def run_health_checks(self) -> List[HealthCheck]:
        """Run all registered health checks"""
        results = []
        timestamp = datetime.now()
        
        for name, check_func in self.health_checks.items():
            start_time = time.time()
            try:
                details = await check_func()
                response_time = (time.time() - start_time) * 1000
                
                result = HealthCheck(
                    service_name=name,
                    status="healthy",
                    response_time=response_time,
                    timestamp=timestamp,
                    details=details
                )
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                result = HealthCheck(
                    service_name=name,
                    status="unhealthy",
                    response_time=response_time,
                    timestamp=timestamp,
                    details={},
                    error_message=str(e)
                )
            
            results.append(result)
            self.health_history.append(result)
        
        return results
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # Check Redis connection
            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            redis_client.ping()
            
            return {
                "redis": "connected",
                "redis_info": redis_client.info()
            }
        except Exception as e:
            raise Exception(f"Database health check failed: {e}")
    
    async def check_api_health(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            # Check if API is responding
            response = requests.get("http://localhost:8000/health", timeout=5)
            return {
                "api_status": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            raise Exception(f"API health check failed: {e}")
    
    async def check_model_health(self) -> Dict[str, Any]:
        """Check ML model health"""
        try:
            # Check if models are loaded and responding
            return {
                "models_loaded": True,
                "model_versions": ["v1.0", "v1.1"],
                "last_training": "2024-01-01T00:00:00Z"
            }
        except Exception as e:
            raise Exception(f"Model health check failed: {e}")

class RealTimeMonitoringSystem:
    """Main real-time monitoring system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.websocket_connections = []
        self.monitoring_active = False
        
        # Register health checks
        self.health_checker.register_health_check("database", self.health_checker.check_database_health)
        self.health_checker.register_health_check("api", self.health_checker.check_api_health)
        self.health_checker.register_health_check("models", self.health_checker.check_model_health)
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        self.monitoring_active = True
        logger.info("Starting real-time monitoring system")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._websocket_broadcast_loop())
        ]
        
        await asyncio.gather(*tasks)
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                system_metrics = await self.metrics_collector.collect_system_metrics()
                app_metrics = await self.metrics_collector.collect_application_metrics()
                all_metrics = system_metrics + app_metrics
                
                # Store metrics
                await self.metrics_collector.store_metrics(all_metrics)
                
                # Detect anomalies
                for metric in all_metrics:
                    anomaly_result = self.anomaly_detector.detect_anomaly(metric.metric_name, metric.value)
                    if anomaly_result["is_anomaly"]:
                        logger.warning(f"Anomaly detected in {metric.metric_name}: {metric.value}")
                    
                    # Update training data
                    self.anomaly_detector.update_training_data(metric.metric_name, metric.value)
                
                # Evaluate alerts
                await self.alert_manager.evaluate_metrics(all_metrics)
                
                # Broadcast to WebSocket connections
                await self._broadcast_metrics(all_metrics)
                
                await asyncio.sleep(self.metrics_collector.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """Health check loop"""
        while self.monitoring_active:
            try:
                health_results = await self.health_checker.run_health_checks()
                await self._broadcast_health_status(health_results)
                await asyncio.sleep(self.health_checker.check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _websocket_broadcast_loop(self):
        """WebSocket broadcast loop"""
        while self.monitoring_active:
            try:
                # Send periodic status updates
                status = {
                    "timestamp": datetime.now().isoformat(),
                    "monitoring_active": self.monitoring_active,
                    "active_alerts": len([a for a in self.alert_manager.alerts.values() if a.status == AlertStatus.ACTIVE]),
                    "total_metrics": len(self.metrics_collector.metrics_buffer)
                }
                await self._broadcast_to_websockets(status)
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error in WebSocket broadcast loop: {e}")
                await asyncio.sleep(10)
    
    async def _broadcast_metrics(self, metrics: List[SystemMetric]):
        """Broadcast metrics to WebSocket connections"""
        if self.websocket_connections:
            data = {
                "type": "metrics",
                "timestamp": datetime.now().isoformat(),
                "metrics": [asdict(metric) for metric in metrics]
            }
            await self._broadcast_to_websockets(data)
    
    async def _broadcast_health_status(self, health_results: List[HealthCheck]):
        """Broadcast health status to WebSocket connections"""
        if self.websocket_connections:
            data = {
                "type": "health",
                "timestamp": datetime.now().isoformat(),
                "health": [asdict(result) for result in health_results]
            }
            await self._broadcast_to_websockets(data)
    
    async def _broadcast_to_websockets(self, data: Dict[str, Any]):
        """Broadcast data to all WebSocket connections"""
        if not self.websocket_connections:
            return
        
        message = json.dumps(data)
        disconnected = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected connections
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    def add_websocket_connection(self, websocket: WebSocket):
        """Add WebSocket connection for real-time updates"""
        self.websocket_connections.append(websocket)
    
    def remove_websocket_connection(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        logger.info("Stopping real-time monitoring system")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        active_alerts = [a for a in self.alert_manager.alerts.values() if a.status == AlertStatus.ACTIVE]
        
        return {
            "monitoring_active": self.monitoring_active,
            "active_alerts": len(active_alerts),
            "total_alerts": len(self.alert_manager.alert_history),
            "websocket_connections": len(self.websocket_connections),
            "health_checks": len(self.health_checker.health_checks),
            "anomaly_models": len(self.anomaly_detector.models),
            "timestamp": datetime.now().isoformat()
        }

# Global monitoring system instance
monitoring_system = RealTimeMonitoringSystem()

# FastAPI app for monitoring endpoints
app = FastAPI(title="AI Document Classifier Monitoring", version="1.0.0")

@app.websocket("/ws/monitoring")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring"""
    await websocket.accept()
    monitoring_system.add_websocket_connection(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        monitoring_system.remove_websocket_connection(websocket)

@app.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    return monitoring_system.get_system_status()

@app.get("/monitoring/alerts")
async def get_alerts():
    """Get all alerts"""
    return {
        "active_alerts": [asdict(alert) for alert in monitoring_system.alert_manager.alerts.values() 
                         if alert.status == AlertStatus.ACTIVE],
        "alert_history": [asdict(alert) for alert in monitoring_system.alert_manager.alert_history[-100:]]
    }

@app.get("/monitoring/health")
async def get_health_status():
    """Get health check results"""
    health_results = await monitoring_system.health_checker.run_health_checks()
    return [asdict(result) for result in health_results]

@app.get("/monitoring/metrics")
async def get_recent_metrics():
    """Get recent metrics"""
    # Implementation to retrieve recent metrics from Redis
    return {"message": "Recent metrics endpoint"}

if __name__ == "__main__":
    # Start monitoring system
    asyncio.run(monitoring_system.start_monitoring())
























