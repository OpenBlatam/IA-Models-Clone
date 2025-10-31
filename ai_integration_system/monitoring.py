"""
AI Integration System - Monitoring and Metrics
Comprehensive monitoring, metrics collection, and alerting system
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import psutil
import asyncio

from .config import settings
from .database import get_db_session
from .models import IntegrationRequest, IntegrationResult, IntegrationMetrics

logger = logging.getLogger(__name__)

# Prometheus metrics
registry = CollectorRegistry()

# Integration metrics
integration_requests_total = Counter(
    'integration_requests_total',
    'Total number of integration requests',
    ['platform', 'content_type', 'status'],
    registry=registry
)

integration_duration_seconds = Histogram(
    'integration_duration_seconds',
    'Time spent processing integration requests',
    ['platform', 'content_type'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=registry
)

integration_queue_size = Gauge(
    'integration_queue_size',
    'Current size of integration queue',
    registry=registry
)

active_integrations = Gauge(
    'active_integrations',
    'Number of currently active integrations',
    ['platform'],
    registry=registry
)

platform_health_status = Gauge(
    'platform_health_status',
    'Health status of platforms (1=healthy, 0=unhealthy)',
    ['platform'],
    registry=registry
)

webhook_events_total = Counter(
    'webhook_events_total',
    'Total number of webhook events received',
    ['platform', 'event_type', 'status'],
    registry=registry
)

database_connections_active = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=registry
)

system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

system_memory_usage = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage',
    registry=registry
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage',
    registry=registry
)

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    severity: str  # critical, warning, info
    title: str
    message: str
    timestamp: datetime
    platform: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class MonitoringService:
    """Main monitoring service"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.metrics_cache: Dict[str, Any] = {}
        self.last_health_check = {}
        self.alert_thresholds = {
            "integration_failure_rate": 0.1,  # 10%
            "queue_size": 100,
            "response_time": 30.0,  # seconds
            "cpu_usage": 80.0,  # percentage
            "memory_usage": 85.0,  # percentage
            "disk_usage": 90.0,  # percentage
        }
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            system_disk_usage.set(disk_percent)
            
            # Database connections
            from .database import get_connection_pool_status
            pool_status = get_connection_pool_status()
            if "checked_out" in pool_status:
                database_connections_active.set(pool_status["checked_out"])
            
            metrics = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk_percent,
                "database_connections": pool_status.get("checked_out", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check for alerts
            await self.check_system_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return {"error": str(e)}
    
    async def collect_integration_metrics(self) -> Dict[str, Any]:
        """Collect integration-specific metrics"""
        try:
            with get_db_session() as session:
                # Get recent integration statistics
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                
                # Total requests in last hour
                total_requests = session.query(IntegrationRequest).filter(
                    IntegrationRequest.created_at >= cutoff_time
                ).count()
                
                # Success rate
                successful_requests = session.query(IntegrationRequest).filter(
                    IntegrationRequest.created_at >= cutoff_time,
                    IntegrationRequest.status == "completed"
                ).count()
                
                success_rate = (successful_requests / total_requests) if total_requests > 0 else 0
                
                # Platform-specific metrics
                platform_metrics = {}
                for platform in ["salesforce", "mailchimp", "wordpress", "hubspot"]:
                    platform_requests = session.query(IntegrationRequest).filter(
                        IntegrationRequest.created_at >= cutoff_time,
                        IntegrationRequest.target_platforms.contains([platform])
                    ).count()
                    
                    platform_successful = session.query(IntegrationRequest).filter(
                        IntegrationRequest.created_at >= cutoff_time,
                        IntegrationRequest.target_platforms.contains([platform]),
                        IntegrationRequest.status == "completed"
                    ).count()
                    
                    platform_success_rate = (platform_successful / platform_requests) if platform_requests > 0 else 0
                    
                    platform_metrics[platform] = {
                        "requests": platform_requests,
                        "success_rate": platform_success_rate
                    }
                    
                    # Update Prometheus metrics
                    integration_requests_total.labels(
                        platform=platform,
                        content_type="all",
                        status="total"
                    ).inc(platform_requests)
                    
                    integration_requests_total.labels(
                        platform=platform,
                        content_type="all",
                        status="success"
                    ).inc(platform_successful)
                
                metrics = {
                    "total_requests": total_requests,
                    "success_rate": success_rate,
                    "platform_metrics": platform_metrics,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Check for alerts
                await self.check_integration_alerts(metrics)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error collecting integration metrics: {str(e)}")
            return {"error": str(e)}
    
    async def check_platform_health(self) -> Dict[str, Any]:
        """Check health of all platforms"""
        try:
            from .integration_engine import integration_engine
            
            health_results = {}
            
            for platform_name, connector in integration_engine.connectors.items():
                try:
                    # Test connection
                    start_time = time.time()
                    is_healthy = await connector.authenticate()
                    response_time = time.time() - start_time
                    
                    health_results[platform_name] = {
                        "status": "healthy" if is_healthy else "unhealthy",
                        "response_time": response_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Update Prometheus metrics
                    platform_health_status.labels(platform=platform_name).set(1 if is_healthy else 0)
                    
                    # Store in database
                    with get_db_session() as session:
                        metric = IntegrationMetrics(
                            platform=platform_name,
                            metric_type="health_check",
                            metric_value="1" if is_healthy else "0",
                            metadata={
                                "response_time": response_time,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                        session.add(metric)
                        session.commit()
                    
                except Exception as e:
                    logger.error(f"Health check failed for {platform_name}: {str(e)}")
                    health_results[platform_name] = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    platform_health_status.labels(platform=platform_name).set(0)
            
            self.last_health_check = health_results
            return health_results
            
        except Exception as e:
            logger.error(f"Error checking platform health: {str(e)}")
            return {"error": str(e)}
    
    async def check_system_alerts(self, metrics: Dict[str, Any]):
        """Check for system-level alerts"""
        try:
            # CPU usage alert
            if metrics.get("cpu_usage", 0) > self.alert_thresholds["cpu_usage"]:
                await self.create_alert(
                    "high_cpu_usage",
                    "warning",
                    "High CPU Usage",
                    f"CPU usage is {metrics['cpu_usage']:.1f}%, above threshold of {self.alert_thresholds['cpu_usage']}%"
                )
            
            # Memory usage alert
            if metrics.get("memory_usage", 0) > self.alert_thresholds["memory_usage"]:
                await self.create_alert(
                    "high_memory_usage",
                    "warning",
                    "High Memory Usage",
                    f"Memory usage is {metrics['memory_usage']:.1f}%, above threshold of {self.alert_thresholds['memory_usage']}%"
                )
            
            # Disk usage alert
            if metrics.get("disk_usage", 0) > self.alert_thresholds["disk_usage"]:
                await self.create_alert(
                    "high_disk_usage",
                    "critical",
                    "High Disk Usage",
                    f"Disk usage is {metrics['disk_usage']:.1f}%, above threshold of {self.alert_thresholds['disk_usage']}%"
                )
            
        except Exception as e:
            logger.error(f"Error checking system alerts: {str(e)}")
    
    async def check_integration_alerts(self, metrics: Dict[str, Any]):
        """Check for integration-specific alerts"""
        try:
            # Low success rate alert
            success_rate = metrics.get("success_rate", 1.0)
            if success_rate < (1.0 - self.alert_thresholds["integration_failure_rate"]):
                await self.create_alert(
                    "low_success_rate",
                    "warning",
                    "Low Integration Success Rate",
                    f"Integration success rate is {success_rate:.1%}, below threshold of {1.0 - self.alert_thresholds['integration_failure_rate']:.1%}"
                )
            
            # Platform-specific alerts
            for platform, platform_metrics in metrics.get("platform_metrics", {}).items():
                platform_success_rate = platform_metrics.get("success_rate", 1.0)
                if platform_success_rate < (1.0 - self.alert_thresholds["integration_failure_rate"]):
                    await self.create_alert(
                        f"low_success_rate_{platform}",
                        "warning",
                        f"Low Success Rate - {platform.title()}",
                        f"{platform.title()} success rate is {platform_success_rate:.1%}, below threshold",
                        platform=platform
                    )
            
        except Exception as e:
            logger.error(f"Error checking integration alerts: {str(e)}")
    
    async def create_alert(self, alert_id: str, severity: str, title: str, message: str, platform: Optional[str] = None):
        """Create a new alert"""
        try:
            # Check if alert already exists and is unresolved
            existing_alert = next(
                (alert for alert in self.alerts if alert.id == alert_id and not alert.resolved),
                None
            )
            
            if existing_alert:
                # Update existing alert
                existing_alert.message = message
                existing_alert.timestamp = datetime.utcnow()
            else:
                # Create new alert
                alert = Alert(
                    id=alert_id,
                    severity=severity,
                    title=title,
                    message=message,
                    timestamp=datetime.utcnow(),
                    platform=platform
                )
                self.alerts.append(alert)
                
                # Log alert
                logger.warning(f"ALERT [{severity.upper()}] {title}: {message}")
                
                # Send notification (implement based on your notification system)
                await self.send_alert_notification(alert)
        
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        try:
            alert = next((alert for alert in self.alerts if alert.id == alert_id and not alert.resolved), None)
            if alert:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                logger.info(f"Alert resolved: {alert.title}")
        
        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")
    
    async def send_alert_notification(self, alert: Alert):
        """Send alert notification (implement based on your notification system)"""
        try:
            # This would integrate with your notification system
            # Examples: email, Slack, PagerDuty, etc.
            
            if alert.severity == "critical":
                # Send immediate notification for critical alerts
                logger.critical(f"CRITICAL ALERT: {alert.title} - {alert.message}")
            elif alert.severity == "warning":
                # Send notification for warnings
                logger.warning(f"WARNING: {alert.title} - {alert.message}")
            
            # Store alert in database for persistence
            with get_db_session() as session:
                from .models import IntegrationLog
                log_entry = IntegrationLog(
                    platform=alert.platform or "system",
                    action="alert_created",
                    status="alert",
                    message=f"{alert.severity.upper()}: {alert.title} - {alert.message}",
                    details={
                        "alert_id": alert.id,
                        "severity": alert.severity,
                        "timestamp": alert.timestamp.isoformat()
                    }
                )
                session.add(log_entry)
                session.commit()
        
        except Exception as e:
            logger.error(f"Error sending alert notification: {str(e)}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        try:
            return {
                "system_metrics": self.metrics_cache.get("system", {}),
                "integration_metrics": self.metrics_cache.get("integration", {}),
                "platform_health": self.last_health_check,
                "active_alerts": [alert for alert in self.alerts if not alert.resolved],
                "resolved_alerts": [alert for alert in self.alerts if alert.resolved],
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {str(e)}")
            return {"error": str(e)}
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        try:
            return generate_latest(registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating Prometheus metrics: {str(e)}")
            return f"# Error generating metrics: {str(e)}\n"
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        logger.info("Starting monitoring service...")
        
        while True:
            try:
                # Collect system metrics
                system_metrics = await self.collect_system_metrics()
                self.metrics_cache["system"] = system_metrics
                
                # Collect integration metrics
                integration_metrics = await self.collect_integration_metrics()
                self.metrics_cache["integration"] = integration_metrics
                
                # Check platform health
                health_results = await self.check_platform_health()
                
                # Wait before next collection
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

# Global monitoring service instance
monitoring_service = MonitoringService()

# Health check endpoint data
def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # System health
        system_metrics = monitoring_service.metrics_cache.get("system", {})
        health_status["checks"]["system"] = {
            "cpu_usage": system_metrics.get("cpu_usage", 0),
            "memory_usage": system_metrics.get("memory_usage", 0),
            "disk_usage": system_metrics.get("disk_usage", 0),
            "status": "healthy" if all([
                system_metrics.get("cpu_usage", 0) < 90,
                system_metrics.get("memory_usage", 0) < 90,
                system_metrics.get("disk_usage", 0) < 95
            ]) else "warning"
        }
        
        # Integration health
        integration_metrics = monitoring_service.metrics_cache.get("integration", {})
        success_rate = integration_metrics.get("success_rate", 1.0)
        health_status["checks"]["integration"] = {
            "success_rate": success_rate,
            "total_requests": integration_metrics.get("total_requests", 0),
            "status": "healthy" if success_rate > 0.9 else "warning"
        }
        
        # Platform health
        health_status["checks"]["platforms"] = monitoring_service.last_health_check
        
        # Database health
        from .database import check_database_health
        health_status["checks"]["database"] = check_database_health()
        
        # Overall status
        all_healthy = all([
            health_status["checks"]["system"]["status"] == "healthy",
            health_status["checks"]["integration"]["status"] == "healthy",
            health_status["checks"]["database"]["status"] == "healthy"
        ])
        
        if not all_healthy:
            health_status["status"] = "warning"
        
        # Check for critical alerts
        critical_alerts = [alert for alert in monitoring_service.alerts 
                          if not alert.resolved and alert.severity == "critical"]
        if critical_alerts:
            health_status["status"] = "critical"
            health_status["critical_alerts"] = len(critical_alerts)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error getting health status: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Export main components
__all__ = [
    "MonitoringService",
    "monitoring_service",
    "get_health_status",
    "Alert",
    "registry"
]



























