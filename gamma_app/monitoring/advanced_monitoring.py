"""
Gamma App - Advanced Monitoring System
Ultra-advanced monitoring with real-time analytics and intelligent alerting
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import numpy as np
from collections import defaultdict, deque
import threading
import websockets
from websockets.server import WebSocketServerProtocol
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import structlog
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests

logger = structlog.get_logger(__name__)

class AlertLevel(Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    level: AlertLevel
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    service: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    name: str
    description: str
    refresh_interval: int = 30
    metrics: List[str] = None
    widgets: List[Dict[str, Any]] = None
    auto_refresh: bool = True

class AdvancedMonitoringSystem:
    """
    Ultra-advanced monitoring system with real-time analytics
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize advanced monitoring system"""
        self.config = config or {}
        
        # Core components
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[Alert] = []
        self.dashboards: Dict[str, DashboardConfig] = {}
        self.websocket_clients: List[WebSocketServerProtocol] = []
        
        # Prometheus metrics
        self.prometheus_metrics = {
            'system_cpu_percent': Gauge('system_cpu_percent', 'System CPU usage percentage'),
            'system_memory_percent': Gauge('system_memory_percent', 'System memory usage percentage'),
            'system_disk_percent': Gauge('system_disk_percent', 'System disk usage percentage'),
            'system_network_io': Gauge('system_network_io', 'System network I/O bytes'),
            'application_requests_total': Counter('application_requests_total', 'Total application requests', ['method', 'endpoint', 'status']),
            'application_response_time': Histogram('application_response_time', 'Application response time', ['endpoint']),
            'application_active_connections': Gauge('application_active_connections', 'Active application connections'),
            'ai_model_inference_time': Histogram('ai_model_inference_time', 'AI model inference time', ['model_name']),
            'cache_hit_rate': Gauge('cache_hit_rate', 'Cache hit rate'),
            'security_events_total': Counter('security_events_total', 'Total security events', ['event_type', 'severity']),
            'database_connections': Gauge('database_connections', 'Database connections'),
            'queue_size': Gauge('queue_size', 'Queue size', ['queue_name']),
            'error_rate': Gauge('error_rate', 'Application error rate'),
            'uptime_seconds': Gauge('uptime_seconds', 'Application uptime in seconds')
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 80, 'critical': 95},
            'disk_percent': {'warning': 85, 'critical': 95},
            'response_time': {'warning': 2.0, 'critical': 5.0},
            'error_rate': {'warning': 5.0, 'critical': 10.0},
            'cache_hit_rate': {'warning': 70, 'critical': 50}
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None
        self.websocket_task: Optional[asyncio.Task] = None
        
        # Redis for distributed monitoring
        self.redis_client = None
        self._init_redis()
        
        # Notification channels
        self.notification_channels = {
            'email': self._send_email_notification,
            'webhook': self._send_webhook_notification,
            'slack': self._send_slack_notification,
            'discord': self._send_discord_notification
        }
        
        # Custom metrics collectors
        self.custom_collectors: Dict[str, Callable] = {}
        
        # Start time for uptime calculation
        self.start_time = time.time()
        
        logger.info("Advanced Monitoring System initialized")
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established for monitoring")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    
    async def start_monitoring(self, interval: int = 10):
        """Start comprehensive monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        self.alert_task = asyncio.create_task(self._alert_processing_loop(30))
        self.websocket_task = asyncio.create_task(self._websocket_server())
        
        logger.info(f"Advanced monitoring started (interval: {interval}s)")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        
        # Cancel tasks
        for task in [self.monitor_task, self.alert_task, self.websocket_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close WebSocket connections
        for client in self.websocket_clients:
            await client.close()
        
        logger.info("Advanced monitoring stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
                # Collect custom metrics
                await self._collect_custom_metrics()
                
                # Update Prometheus metrics
                await self._update_prometheus_metrics()
                
                # Broadcast to WebSocket clients
                await self._broadcast_metrics()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric('system_cpu_percent', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._add_metric('system_memory_percent', memory.percent)
            self._add_metric('system_memory_used_gb', memory.used / 1024**3)
            self._add_metric('system_memory_available_gb', memory.available / 1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self._add_metric('system_disk_percent', disk.percent)
            self._add_metric('system_disk_used_gb', disk.used / 1024**3)
            self._add_metric('system_disk_free_gb', disk.free / 1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            self._add_metric('system_network_bytes_sent', network.bytes_sent)
            self._add_metric('system_network_bytes_recv', network.bytes_recv)
            
            # Process metrics
            process = psutil.Process()
            self._add_metric('process_cpu_percent', process.cpu_percent())
            self._add_metric('process_memory_mb', process.memory_info().rss / 1024**2)
            self._add_metric('process_threads', process.num_threads())
            
            # Uptime
            uptime = time.time() - self.start_time
            self._add_metric('uptime_seconds', uptime)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-level metrics"""
        try:
            # This would integrate with the application services
            # For now, we'll simulate some metrics
            
            # Request metrics
            self._add_metric('application_requests_per_second', np.random.poisson(10))
            self._add_metric('application_response_time_avg', np.random.normal(0.5, 0.1))
            self._add_metric('application_active_connections', np.random.randint(10, 100))
            
            # AI model metrics
            self._add_metric('ai_model_inference_time', np.random.normal(1.0, 0.2))
            self._add_metric('ai_model_memory_usage_gb', np.random.normal(2.0, 0.5))
            
            # Cache metrics
            self._add_metric('cache_hit_rate', np.random.normal(85, 5))
            self._add_metric('cache_size_mb', np.random.normal(100, 20))
            
            # Security metrics
            self._add_metric('security_events_per_minute', np.random.poisson(2))
            self._add_metric('blocked_ips_count', np.random.randint(0, 10))
            
            # Database metrics
            self._add_metric('database_connections', np.random.randint(5, 20))
            self._add_metric('database_query_time_avg', np.random.normal(0.1, 0.02))
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    async def _collect_custom_metrics(self):
        """Collect custom metrics from registered collectors"""
        try:
            for name, collector in self.custom_collectors.items():
                try:
                    if asyncio.iscoroutinefunction(collector):
                        value = await collector()
                    else:
                        value = collector()
                    
                    self._add_metric(name, value)
                except Exception as e:
                    logger.error(f"Error in custom collector {name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting custom metrics: {e}")
    
    def _add_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add metric to collection"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type=MetricType.GAUGE
        )
        
        self.metrics[name].append(metric)
        
        # Store in Redis for distributed access
        if self.redis_client:
            try:
                key = f"metrics:{name}"
                self.redis_client.lpush(key, json.dumps(asdict(metric), default=str))
                self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 values
                self.redis_client.expire(key, 3600)  # Expire after 1 hour
            except Exception as e:
                logger.error(f"Error storing metric in Redis: {e}")
    
    async def _update_prometheus_metrics(self):
        """Update Prometheus metrics"""
        try:
            # Update system metrics
            if 'system_cpu_percent' in self.metrics and self.metrics['system_cpu_percent']:
                self.prometheus_metrics['system_cpu_percent'].set(
                    self.metrics['system_cpu_percent'][-1].value
                )
            
            if 'system_memory_percent' in self.metrics and self.metrics['system_memory_percent']:
                self.prometheus_metrics['system_memory_percent'].set(
                    self.metrics['system_memory_percent'][-1].value
                )
            
            if 'system_disk_percent' in self.metrics and self.metrics['system_disk_percent']:
                self.prometheus_metrics['system_disk_percent'].set(
                    self.metrics['system_disk_percent'][-1].value
                )
            
            # Update application metrics
            if 'application_active_connections' in self.metrics and self.metrics['application_active_connections']:
                self.prometheus_metrics['application_active_connections'].set(
                    self.metrics['application_active_connections'][-1].value
                )
            
            if 'cache_hit_rate' in self.metrics and self.metrics['cache_hit_rate']:
                self.prometheus_metrics['cache_hit_rate'].set(
                    self.metrics['cache_hit_rate'][-1].value
                )
            
            # Update uptime
            uptime = time.time() - self.start_time
            self.prometheus_metrics['uptime_seconds'].set(uptime)
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    async def _alert_processing_loop(self, interval: int):
        """Alert processing loop"""
        while self.monitoring_active:
            try:
                # Check alert conditions
                await self._check_alert_conditions()
                
                # Process pending alerts
                await self._process_pending_alerts()
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(interval)
    
    async def _check_alert_conditions(self):
        """Check alert conditions"""
        try:
            for metric_name, thresholds in self.alert_thresholds.items():
                if metric_name not in self.metrics or not self.metrics[metric_name]:
                    continue
                
                current_value = self.metrics[metric_name][-1].value
                
                # Check warning threshold
                if 'warning' in thresholds and current_value >= thresholds['warning']:
                    await self._create_alert(
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold_value=thresholds['warning'],
                        level=AlertLevel.WARNING,
                        title=f"High {metric_name.replace('_', ' ').title()}",
                        description=f"{metric_name} is at {current_value:.2f}, above warning threshold of {thresholds['warning']}"
                    )
                
                # Check critical threshold
                if 'critical' in thresholds and current_value >= thresholds['critical']:
                    await self._create_alert(
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold_value=thresholds['critical'],
                        level=AlertLevel.CRITICAL,
                        title=f"Critical {metric_name.replace('_', ' ').title()}",
                        description=f"{metric_name} is at {current_value:.2f}, above critical threshold of {thresholds['critical']}"
                    )
                    
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
    
    async def _create_alert(self, metric_name: str, current_value: float, 
                          threshold_value: float, level: AlertLevel, 
                          title: str, description: str):
        """Create new alert"""
        try:
            # Check if similar alert already exists and is not resolved
            existing_alert = None
            for alert in self.alerts:
                if (alert.metric_name == metric_name and 
                    alert.level == level and 
                    not alert.resolved):
                    existing_alert = alert
                    break
            
            if existing_alert:
                # Update existing alert
                existing_alert.current_value = current_value
                existing_alert.timestamp = datetime.now()
                return
            
            # Create new alert
            alert = Alert(
                id=f"{metric_name}_{level.value}_{int(time.time())}",
                timestamp=datetime.now(),
                level=level,
                title=title,
                description=description,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                service="monitoring",
                metadata={
                    "threshold_type": "greater_than",
                    "alert_rule": f"{metric_name} > {threshold_value}"
                }
            )
            
            self.alerts.append(alert)
            
            # Store in Redis
            if self.redis_client:
                try:
                    key = f"alerts:{alert.id}"
                    self.redis_client.setex(
                        key, 
                        86400,  # 24 hours
                        json.dumps(asdict(alert), default=str)
                    )
                except Exception as e:
                    logger.error(f"Error storing alert in Redis: {e}")
            
            logger.warning(f"Alert created: {title}", 
                         alert_id=alert.id,
                         level=level.value,
                         metric=metric_name,
                         value=current_value,
                         threshold=threshold_value)
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    async def _process_pending_alerts(self):
        """Process pending alerts"""
        try:
            for alert in self.alerts:
                if alert.resolved:
                    continue
                
                # Check if alert should be resolved
                if alert.metric_name in self.metrics and self.metrics[alert.metric_name]:
                    current_value = self.metrics[alert.metric_name][-1].value
                    
                    # Resolve alert if condition is no longer met
                    if current_value < alert.threshold_value * 0.9:  # 10% hysteresis
                        alert.resolved = True
                        alert.resolved_at = datetime.now()
                        
                        logger.info(f"Alert resolved: {alert.title}", 
                                  alert_id=alert.id,
                                  metric=alert.metric_name,
                                  value=current_value)
                
                # Send notifications for new alerts
                if not hasattr(alert, 'notified'):
                    await self._send_alert_notifications(alert)
                    alert.notified = True
                    
        except Exception as e:
            logger.error(f"Error processing pending alerts: {e}")
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications"""
        try:
            notification_config = self.config.get('notifications', {})
            
            for channel, enabled in notification_config.items():
                if not enabled:
                    continue
                
                if channel in self.notification_channels:
                    try:
                        await self.notification_channels[channel](alert)
                    except Exception as e:
                        logger.error(f"Error sending {channel} notification: {e}")
                        
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            smtp_config = self.config.get('smtp', {})
            if not smtp_config:
                return
            
            msg = MimeMultipart()
            msg['From'] = smtp_config.get('from_email')
            msg['To'] = smtp_config.get('to_email')
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            - Title: {alert.title}
            - Description: {alert.description}
            - Level: {alert.level.value}
            - Metric: {alert.metric_name}
            - Current Value: {alert.current_value}
            - Threshold: {alert.threshold_value}
            - Timestamp: {alert.timestamp}
            - Service: {alert.service}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config.get('host'), smtp_config.get('port'))
            server.starttls()
            server.login(smtp_config.get('username'), smtp_config.get('password'))
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                return
            
            payload = {
                "alert": asdict(alert),
                "timestamp": datetime.now().isoformat(),
                "source": "gamma_app_monitoring"
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            slack_config = self.config.get('slack', {})
            if not slack_config:
                return
            
            webhook_url = slack_config.get('webhook_url')
            if not webhook_url:
                return
            
            color = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning",
                AlertLevel.CRITICAL: "danger",
                AlertLevel.EMERGENCY: "danger"
            }.get(alert.level, "good")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {"title": "Level", "value": alert.level.value, "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold_value), "short": True}
                    ],
                    "timestamp": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_discord_notification(self, alert: Alert):
        """Send Discord notification"""
        try:
            discord_config = self.config.get('discord', {})
            if not discord_config:
                return
            
            webhook_url = discord_config.get('webhook_url')
            if not webhook_url:
                return
            
            color = {
                AlertLevel.INFO: 0x00ff00,
                AlertLevel.WARNING: 0xffff00,
                AlertLevel.CRITICAL: 0xff0000,
                AlertLevel.EMERGENCY: 0xff0000
            }.get(alert.level, 0x00ff00)
            
            payload = {
                "embeds": [{
                    "title": alert.title,
                    "description": alert.description,
                    "color": color,
                    "fields": [
                        {"name": "Level", "value": alert.level.value, "inline": True},
                        {"name": "Metric", "value": alert.metric_name, "inline": True},
                        {"name": "Current Value", "value": str(alert.current_value), "inline": True},
                        {"name": "Threshold", "value": str(alert.threshold_value), "inline": True}
                    ],
                    "timestamp": alert.timestamp.isoformat()
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Discord notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    async def _websocket_server(self):
        """WebSocket server for real-time metrics"""
        try:
            async def handle_client(websocket: WebSocketServerProtocol, path: str):
                self.websocket_clients.append(websocket)
                logger.info(f"WebSocket client connected: {websocket.remote_address}")
                
                try:
                    # Send initial metrics
                    await self._send_metrics_to_client(websocket)
                    
                    # Keep connection alive
                    async for message in websocket:
                        if message == "ping":
                            await websocket.send("pong")
                        elif message == "get_metrics":
                            await self._send_metrics_to_client(websocket)
                            
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    if websocket in self.websocket_clients:
                        self.websocket_clients.remove(websocket)
                    logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
            
            # Start WebSocket server
            server = await websockets.serve(
                handle_client, 
                "localhost", 
                8765,
                ping_interval=30,
                ping_timeout=10
            )
            
            logger.info("WebSocket server started on localhost:8765")
            
            # Keep server running
            await server.wait_closed()
            
        except Exception as e:
            logger.error(f"Error in WebSocket server: {e}")
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to all WebSocket clients"""
        try:
            if not self.websocket_clients:
                return
            
            # Prepare metrics data
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {}
            }
            
            for name, metric_deque in self.metrics.items():
                if metric_deque:
                    latest_metric = metric_deque[-1]
                    metrics_data["metrics"][name] = {
                        "value": latest_metric.value,
                        "timestamp": latest_metric.timestamp.isoformat()
                    }
            
            # Send to all connected clients
            message = json.dumps(metrics_data)
            disconnected_clients = []
            
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.append(client)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                self.websocket_clients.remove(client)
                
        except Exception as e:
            logger.error(f"Error broadcasting metrics: {e}")
    
    async def _send_metrics_to_client(self, websocket: WebSocketServerProtocol):
        """Send current metrics to a specific client"""
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {}
            }
            
            for name, metric_deque in self.metrics.items():
                if metric_deque:
                    latest_metric = metric_deque[-1]
                    metrics_data["metrics"][name] = {
                        "value": latest_metric.value,
                        "timestamp": latest_metric.timestamp.isoformat()
                    }
            
            await websocket.send(json.dumps(metrics_data))
            
        except Exception as e:
            logger.error(f"Error sending metrics to client: {e}")
    
    def register_custom_collector(self, name: str, collector: Callable):
        """Register custom metric collector"""
        self.custom_collectors[name] = collector
        logger.info(f"Custom collector registered: {name}")
    
    def create_dashboard(self, name: str, config: DashboardConfig):
        """Create monitoring dashboard"""
        self.dashboards[name] = config
        logger.info(f"Dashboard created: {name}")
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "metrics_count": len(self.metrics),
                "alerts_count": len(self.alerts),
                "active_alerts": len([a for a in self.alerts if not a.resolved]),
                "websocket_clients": len(self.websocket_clients),
                "monitoring_active": self.monitoring_active,
                "latest_metrics": {}
            }
            
            # Get latest values for key metrics
            key_metrics = [
                'system_cpu_percent', 'system_memory_percent', 'system_disk_percent',
                'application_active_connections', 'cache_hit_rate', 'error_rate'
            ]
            
            for metric_name in key_metrics:
                if metric_name in self.metrics and self.metrics[metric_name]:
                    latest_metric = self.metrics[metric_name][-1]
                    summary["latest_metrics"][metric_name] = {
                        "value": latest_metric.value,
                        "timestamp": latest_metric.timestamp.isoformat()
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}
    
    async def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        try:
            active_alerts = [a for a in self.alerts if not a.resolved]
            resolved_alerts = [a for a in self.alerts if a.resolved]
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "resolved_alerts": len(resolved_alerts),
                "alerts_by_level": {},
                "recent_alerts": []
            }
            
            # Count alerts by level
            for level in AlertLevel:
                count = len([a for a in active_alerts if a.level == level])
                summary["alerts_by_level"][level.value] = count
            
            # Get recent alerts (last 10)
            recent_alerts = sorted(
                self.alerts, 
                key=lambda x: x.timestamp, 
                reverse=True
            )[:10]
            
            summary["recent_alerts"] = [asdict(alert) for alert in recent_alerts]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {}
    
    async def close(self):
        """Close monitoring system"""
        await self.stop_monitoring()
        
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Advanced monitoring system closed")

# Global monitoring instance
monitoring_system = AdvancedMonitoringSystem()

# Convenience functions
async def start_monitoring(interval: int = 10):
    """Start monitoring"""
    await monitoring_system.start_monitoring(interval)

async def stop_monitoring():
    """Stop monitoring"""
    await monitoring_system.stop_monitoring()

def register_metric_collector(name: str, collector: Callable):
    """Register metric collector"""
    monitoring_system.register_custom_collector(name, collector)

async def get_monitoring_summary():
    """Get monitoring summary"""
    return await monitoring_system.get_metrics_summary()
















