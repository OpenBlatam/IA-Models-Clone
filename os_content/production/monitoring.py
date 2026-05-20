from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import psutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
from .config import get_production_config
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Production Monitoring for OS Content UGC Video Generator
Advanced monitoring, metrics collection, and alerting
"""



logger = structlog.get_logger("os_content.monitoring")

@dataclass
class SystemMetrics:
    """System metrics data structure"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    load_average: List[float]
    uptime: float
    process_count: int
    thread_count: int

@dataclass
class ApplicationMetrics:
    """Application metrics data structure"""
    timestamp: datetime
    request_count: int
    error_count: int
    success_rate: float
    average_response_time: float
    active_connections: int
    queue_size: int
    cache_hit_rate: float
    database_connections: int
    redis_connections: int

@dataclass
class BusinessMetrics:
    """Business metrics data structure"""
    timestamp: datetime
    videos_processed: int
    videos_failed: int
    total_processing_time: float
    average_processing_time: float
    storage_used: float
    cdn_requests: int
    user_sessions: int
    revenue_generated: float

class ProductionMonitor:
    """Production monitoring system"""
    
    def __init__(self) -> Any:
        self.config = get_production_config()
        self.metrics_history: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.monitoring_active = False
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0,
            "response_time": 2.0,
            "success_rate": 95.0
        }
    
    def _setup_prometheus_metrics(self) -> Any:
        """Setup Prometheus metrics"""
        # System metrics
        self.cpu_gauge = Gauge('os_content_cpu_usage_percent', 'CPU usage percentage')
        self.memory_gauge = Gauge('os_content_memory_usage_percent', 'Memory usage percentage')
        self.disk_gauge = Gauge('os_content_disk_usage_percent', 'Disk usage percentage')
        self.load_gauge = Gauge('os_content_load_average', 'System load average', ['period'])
        
        # Application metrics
        self.request_counter = Counter('os_content_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
        self.error_counter = Counter('os_content_errors_total', 'Total errors', ['type'])
        self.response_time_histogram = Histogram('os_content_response_time_seconds', 'Response time in seconds', ['endpoint'])
        self.active_connections_gauge = Gauge('os_content_active_connections', 'Active connections')
        self.queue_size_gauge = Gauge('os_content_queue_size', 'Queue size')
        self.cache_hit_rate_gauge = Gauge('os_content_cache_hit_rate', 'Cache hit rate')
        
        # Business metrics
        self.videos_processed_counter = Counter('os_content_videos_processed_total', 'Total videos processed')
        self.videos_failed_counter = Counter('os_content_videos_failed_total', 'Total videos failed')
        self.processing_time_histogram = Histogram('os_content_processing_time_seconds', 'Video processing time in seconds')
        self.storage_used_gauge = Gauge('os_content_storage_used_bytes', 'Storage used in bytes')
        self.cdn_requests_counter = Counter('os_content_cdn_requests_total', 'Total CDN requests')
        
        # Database metrics
        self.db_connections_gauge = Gauge('os_content_db_connections', 'Database connections')
        self.db_query_time_histogram = Histogram('os_content_db_query_time_seconds', 'Database query time in seconds')
        
        # Redis metrics
        self.redis_connections_gauge = Gauge('os_content_redis_connections', 'Redis connections')
        self.redis_memory_gauge = Gauge('os_content_redis_memory_bytes', 'Redis memory usage in bytes')
    
    async def start_monitoring(self) -> Any:
        """Start production monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Production monitoring started")
        
        # Start monitoring tasks
        asyncio.create_task(self._system_monitoring_loop())
        asyncio.create_task(self._application_monitoring_loop())
        asyncio.create_task(self._business_monitoring_loop())
        asyncio.create_task(self._alert_monitoring_loop())
    
    async def stop_monitoring(self) -> Any:
        """Stop production monitoring"""
        self.monitoring_active = False
        logger.info("Production monitoring stopped")
    
    async def _system_monitoring_loop(self) -> Any:
        """System monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_system_metrics()
                await self._process_system_metrics(metrics)
                await self._check_system_alerts(metrics)
                
                # Store metrics
                self.metrics_history.append({
                    "type": "system",
                    "timestamp": metrics.timestamp.isoformat(),
                    "data": {
                        "cpu_usage": metrics.cpu_usage,
                        "memory_usage": metrics.memory_usage,
                        "disk_usage": metrics.disk_usage,
                        "network_io": metrics.network_io,
                        "load_average": metrics.load_average,
                        "uptime": metrics.uptime,
                        "process_count": metrics.process_count,
                        "thread_count": metrics.thread_count
                    }
                })
                
                # Update Prometheus metrics
                self._update_system_prometheus_metrics(metrics)
                
                await asyncio.sleep(self.config.metrics_save_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _application_monitoring_loop(self) -> Any:
        """Application monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_application_metrics()
                await self._process_application_metrics(metrics)
                await self._check_application_alerts(metrics)
                
                # Store metrics
                self.metrics_history.append({
                    "type": "application",
                    "timestamp": metrics.timestamp.isoformat(),
                    "data": {
                        "request_count": metrics.request_count,
                        "error_count": metrics.error_count,
                        "success_rate": metrics.success_rate,
                        "average_response_time": metrics.average_response_time,
                        "active_connections": metrics.active_connections,
                        "queue_size": metrics.queue_size,
                        "cache_hit_rate": metrics.cache_hit_rate,
                        "database_connections": metrics.database_connections,
                        "redis_connections": metrics.redis_connections
                    }
                })
                
                # Update Prometheus metrics
                self._update_application_prometheus_metrics(metrics)
                
                await asyncio.sleep(self.config.metrics_save_interval)
                
            except Exception as e:
                logger.error(f"Application monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _business_monitoring_loop(self) -> Any:
        """Business metrics monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_business_metrics()
                await self._process_business_metrics(metrics)
                await self._check_business_alerts(metrics)
                
                # Store metrics
                self.metrics_history.append({
                    "type": "business",
                    "timestamp": metrics.timestamp.isoformat(),
                    "data": {
                        "videos_processed": metrics.videos_processed,
                        "videos_failed": metrics.videos_failed,
                        "total_processing_time": metrics.total_processing_time,
                        "average_processing_time": metrics.average_processing_time,
                        "storage_used": metrics.storage_used,
                        "cdn_requests": metrics.cdn_requests,
                        "user_sessions": metrics.user_sessions,
                        "revenue_generated": metrics.revenue_generated
                    }
                })
                
                # Update Prometheus metrics
                self._update_business_prometheus_metrics(metrics)
                
                await asyncio.sleep(self.config.metrics_save_interval * 2)  # Less frequent
                
            except Exception as e:
                logger.error(f"Business monitoring error: {e}")
                await asyncio.sleep(20)
    
    async def _alert_monitoring_loop(self) -> Any:
        """Alert monitoring loop"""
        while self.monitoring_active:
            try:
                await self._check_alert_conditions()
                await self._send_alerts()
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        # Load average
        load_average = list(psutil.getloadavg())
        
        # Uptime
        uptime = time.time() - psutil.boot_time()
        
        # Process and thread count
        process_count = len(psutil.pids())
        thread_count = psutil.cpu_count()
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            load_average=load_average,
            uptime=uptime,
            process_count=process_count,
            thread_count=thread_count
        )
    
    async def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application metrics"""
        # This would integrate with the actual application metrics
        # For now, return mock data
        
        return ApplicationMetrics(
            timestamp=datetime.utcnow(),
            request_count=100,
            error_count=2,
            success_rate=98.0,
            average_response_time=0.5,
            active_connections=25,
            queue_size=5,
            cache_hit_rate=85.0,
            database_connections=10,
            redis_connections=5
        )
    
    async def _collect_business_metrics(self) -> BusinessMetrics:
        """Collect business metrics"""
        # This would integrate with the actual business metrics
        # For now, return mock data
        
        return BusinessMetrics(
            timestamp=datetime.utcnow(),
            videos_processed=50,
            videos_failed=1,
            total_processing_time=1200.0,
            average_processing_time=24.0,
            storage_used=1024 * 1024 * 1024 * 10,  # 10GB
            cdn_requests=1000,
            user_sessions=100,
            revenue_generated=500.0
        )
    
    async def _process_system_metrics(self, metrics: SystemMetrics):
        """Process system metrics"""
        # Log metrics
        logger.info(
            "System metrics collected",
            cpu_usage=metrics.cpu_usage,
            memory_usage=metrics.memory_usage,
            disk_usage=metrics.disk_usage,
            load_average=metrics.load_average
        )
    
    async def _process_application_metrics(self, metrics: ApplicationMetrics):
        """Process application metrics"""
        # Log metrics
        logger.info(
            "Application metrics collected",
            request_count=metrics.request_count,
            error_count=metrics.error_count,
            success_rate=metrics.success_rate,
            response_time=metrics.average_response_time
        )
    
    async def _process_business_metrics(self, metrics: BusinessMetrics):
        """Process business metrics"""
        # Log metrics
        logger.info(
            "Business metrics collected",
            videos_processed=metrics.videos_processed,
            videos_failed=metrics.videos_failed,
            processing_time=metrics.average_processing_time,
            revenue=metrics.revenue_generated
        )
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system alerts"""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts.append({
                "type": "system",
                "severity": "warning",
                "message": f"High CPU usage: {metrics.cpu_usage}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append({
                "type": "system",
                "severity": "warning",
                "message": f"High memory usage: {metrics.memory_usage}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if metrics.disk_usage > self.alert_thresholds["disk_usage"]:
            alerts.append({
                "type": "system",
                "severity": "critical",
                "message": f"High disk usage: {metrics.disk_usage}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        for alert in alerts:
            self.alert_history.append(alert)
            logger.warning(f"System alert: {alert['message']}")
    
    async def _check_application_alerts(self, metrics: ApplicationMetrics):
        """Check application alerts"""
        alerts = []
        
        if metrics.success_rate < self.alert_thresholds["success_rate"]:
            alerts.append({
                "type": "application",
                "severity": "warning",
                "message": f"Low success rate: {metrics.success_rate}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if metrics.average_response_time > self.alert_thresholds["response_time"]:
            alerts.append({
                "type": "application",
                "severity": "warning",
                "message": f"High response time: {metrics.average_response_time}s",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        error_rate = (metrics.error_count / metrics.request_count * 100) if metrics.request_count > 0 else 0
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append({
                "type": "application",
                "severity": "critical",
                "message": f"High error rate: {error_rate}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        for alert in alerts:
            self.alert_history.append(alert)
            logger.warning(f"Application alert: {alert['message']}")
    
    async def _check_business_alerts(self, metrics: BusinessMetrics):
        """Check business alerts"""
        alerts = []
        
        failure_rate = (metrics.videos_failed / metrics.videos_processed * 100) if metrics.videos_processed > 0 else 0
        if failure_rate > 5.0:
            alerts.append({
                "type": "business",
                "severity": "warning",
                "message": f"High video failure rate: {failure_rate}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        for alert in alerts:
            self.alert_history.append(alert)
            logger.warning(f"Business alert: {alert['message']}")
    
    async def _check_alert_conditions(self) -> Any:
        """Check for alert conditions"""
        # Check for repeated alerts
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        # Group by type and severity
        alert_groups = {}
        for alert in recent_alerts:
            key = f"{alert['type']}_{alert['severity']}"
            if key not in alert_groups:
                alert_groups[key] = []
            alert_groups[key].append(alert)
        
        # Escalate if too many alerts
        for key, alerts in alert_groups.items():
            if len(alerts) > 3:
                logger.error(f"Multiple alerts detected: {key} - {len(alerts)} alerts")
    
    async def _send_alerts(self) -> Any:
        """Send alerts to external systems"""
        # Send to Slack, email, PagerDuty, etc.
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > datetime.utcnow() - timedelta(minutes=1)
        ]
        
        for alert in recent_alerts:
            await self._send_alert_notification(alert)
    
    async def _send_alert_notification(self, alert: Dict[str, Any]):
        """Send individual alert notification"""
        try:
            # This would implement actual notification sending
            # Slack, email, SMS, etc.
            logger.info(f"Alert notification sent: {alert['message']}")
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    async def _cleanup_old_alerts(self) -> Any:
        """Cleanup old alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.alert_history = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
    
    def _update_system_prometheus_metrics(self, metrics: SystemMetrics):
        """Update system Prometheus metrics"""
        self.cpu_gauge.set(metrics.cpu_usage)
        self.memory_gauge.set(metrics.memory_usage)
        self.disk_gauge.set(metrics.disk_usage)
        
        for i, load in enumerate(metrics.load_average):
            self.load_gauge.labels(period=f"{i+1}m").set(load)
    
    def _update_application_prometheus_metrics(self, metrics: ApplicationMetrics):
        """Update application Prometheus metrics"""
        self.active_connections_gauge.set(metrics.active_connections)
        self.queue_size_gauge.set(metrics.queue_size)
        self.cache_hit_rate_gauge.set(metrics.cache_hit_rate)
        self.db_connections_gauge.set(metrics.database_connections)
        self.redis_connections_gauge.set(metrics.redis_connections)
    
    def _update_business_prometheus_metrics(self, metrics: BusinessMetrics):
        """Update business Prometheus metrics"""
        self.videos_processed_counter.inc(metrics.videos_processed)
        self.videos_failed_counter.inc(metrics.videos_failed)
        self.storage_used_gauge.set(metrics.storage_used)
        self.cdn_requests_counter.inc(metrics.cdn_requests)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics_history:
            return {}
        
        # Get recent metrics (last hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        system_metrics = [m for m in recent_metrics if m["type"] == "system"]
        application_metrics = [m for m in recent_metrics if m["type"] == "application"]
        business_metrics = [m for m in recent_metrics if m["type"] == "business"]
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {},
            "application": {},
            "business": {},
            "alerts": len(self.alert_history)
        }
        
        if system_metrics:
            summary["system"] = {
                "avg_cpu_usage": sum(m["data"]["cpu_usage"] for m in system_metrics) / len(system_metrics),
                "avg_memory_usage": sum(m["data"]["memory_usage"] for m in system_metrics) / len(system_metrics),
                "avg_disk_usage": sum(m["data"]["disk_usage"] for m in system_metrics) / len(system_metrics)
            }
        
        if application_metrics:
            summary["application"] = {
                "avg_success_rate": sum(m["data"]["success_rate"] for m in application_metrics) / len(application_metrics),
                "avg_response_time": sum(m["data"]["average_response_time"] for m in application_metrics) / len(application_metrics),
                "avg_cache_hit_rate": sum(m["data"]["cache_hit_rate"] for m in application_metrics) / len(application_metrics)
            }
        
        if business_metrics:
            summary["business"] = {
                "total_videos_processed": sum(m["data"]["videos_processed"] for m in business_metrics),
                "total_videos_failed": sum(m["data"]["videos_failed"] for m in business_metrics),
                "avg_processing_time": sum(m["data"]["average_processing_time"] for m in business_metrics) / len(business_metrics)
            }
        
        return summary
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics"""
        return generate_latest()
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        self.response_time_histogram.labels(endpoint=endpoint).observe(duration)
    
    def record_error(self, error_type: str):
        """Record error metrics"""
        self.error_counter.labels(type=error_type).inc()
    
    def record_video_processing(self, duration: float, success: bool):
        """Record video processing metrics"""
        self.processing_time_histogram.observe(duration)
        if success:
            self.videos_processed_counter.inc()
        else:
            self.videos_failed_counter.inc()

# Global production monitor instance
production_monitor = ProductionMonitor() 