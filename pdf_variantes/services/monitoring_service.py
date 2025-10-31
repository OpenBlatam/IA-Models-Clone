"""
PDF Variantes Monitoring and Analytics System
Comprehensive monitoring, analytics, and health checking
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json

from ..utils.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = ""

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    name: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class MonitoringSystem:
    """System monitoring and metrics collection"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Metrics storage
        self.metrics: List[Metric] = []
        self.alerts: List[Alert] = []
        
        # System metrics
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_io": {"bytes_sent": 0, "bytes_recv": 0},
            "process_count": 0,
            "uptime": 0.0
        }
        
        # Application metrics
        self.app_metrics = {
            "requests_total": 0,
            "requests_per_second": 0.0,
            "response_time_avg": 0.0,
            "error_rate": 0.0,
            "active_connections": 0,
            "cache_hit_rate": 0.0
        }
        
        # Monitoring configuration
        self.monitoring_config = {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "disk_threshold": 90.0,
            "response_time_threshold": 2.0,
            "error_rate_threshold": 5.0
        }
        
        # Start time for uptime calculation
        self.start_time = time.time()
    
    async def initialize(self):
        """Initialize monitoring system"""
        try:
            # Start background monitoring tasks
            asyncio.create_task(self._collect_system_metrics())
            asyncio.create_task(self._check_alerts())
            asyncio.create_task(self._cleanup_old_metrics())
            
            logger.info("Monitoring System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Monitoring System: {e}")
            raise
    
    async def collect_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Collect a custom metric"""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                unit=unit
            )
            
            self.metrics.append(metric)
            
            # Keep only last 1000 metrics to prevent memory issues
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
            
        except Exception as e:
            logger.error(f"Error collecting metric {name}: {e}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            health_status = "healthy"
            issues = []
            
            # Check CPU usage
            if self.system_metrics["cpu_usage"] > self.monitoring_config["cpu_threshold"]:
                health_status = "degraded"
                issues.append(f"High CPU usage: {self.system_metrics['cpu_usage']:.1f}%")
            
            # Check memory usage
            if self.system_metrics["memory_usage"] > self.monitoring_config["memory_threshold"]:
                health_status = "degraded"
                issues.append(f"High memory usage: {self.system_metrics['memory_usage']:.1f}%")
            
            # Check disk usage
            if self.system_metrics["disk_usage"] > self.monitoring_config["disk_threshold"]:
                health_status = "degraded"
                issues.append(f"High disk usage: {self.system_metrics['disk_usage']:.1f}%")
            
            # Check response time
            if self.app_metrics["response_time_avg"] > self.monitoring_config["response_time_threshold"]:
                health_status = "degraded"
                issues.append(f"High response time: {self.app_metrics['response_time_avg']:.2f}s")
            
            # Check error rate
            if self.app_metrics["error_rate"] > self.monitoring_config["error_rate_threshold"]:
                health_status = "unhealthy"
                issues.append(f"High error rate: {self.app_metrics['error_rate']:.1f}%")
            
            return {
                "status": health_status,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "issues": issues,
                "metrics": {
                    "system": self.system_metrics,
                    "application": self.app_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "issues": [f"Health check failed: {str(e)}"],
                "metrics": {}
            }
    
    async def get_metrics(self, name: Optional[str] = None, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Metric]:
        """Get metrics with optional filtering"""
        try:
            filtered_metrics = self.metrics
            
            # Filter by name
            if name:
                filtered_metrics = [m for m in filtered_metrics if m.name == name]
            
            # Filter by time range
            if start_time:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
            
            if end_time:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
            
            return filtered_metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return []
    
    async def create_alert(self, name: str, severity: str, message: str) -> str:
        """Create a new alert"""
        try:
            alert_id = f"alert_{int(time.time())}"
            
            alert = Alert(
                id=alert_id,
                name=name,
                severity=severity,
                message=message,
                timestamp=datetime.utcnow()
            )
            
            self.alerts.append(alert)
            
            # Log alert
            logger.warning(f"Alert created: {name} - {message}")
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return ""
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    logger.info(f"Alert resolved: {alert.name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    async def get_alerts(self, severity: Optional[str] = None, 
                        resolved: Optional[bool] = None) -> List[Alert]:
        """Get alerts with optional filtering"""
        try:
            filtered_alerts = self.alerts
            
            # Filter by severity
            if severity:
                filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
            
            # Filter by resolved status
            if resolved is not None:
                filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
            
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_metrics["cpu_usage"] = cpu_percent
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_metrics["memory_usage"] = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.system_metrics["disk_usage"] = (disk.used / disk.total) * 100
                
                # Network I/O
                network = psutil.net_io_counters()
                self.system_metrics["network_io"] = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                }
                
                # Process count
                self.system_metrics["process_count"] = len(psutil.pids())
                
                # Uptime
                self.system_metrics["uptime"] = time.time() - self.start_time
                
                # Collect metrics
                await self.collect_metric("system.cpu_usage", cpu_percent, unit="percent")
                await self.collect_metric("system.memory_usage", memory.percent, unit="percent")
                await self.collect_metric("system.disk_usage", self.system_metrics["disk_usage"], unit="percent")
                await self.collect_metric("system.process_count", self.system_metrics["process_count"], unit="count")
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        while True:
            try:
                # Check CPU alert
                if self.system_metrics["cpu_usage"] > self.monitoring_config["cpu_threshold"]:
                    await self.create_alert(
                        "High CPU Usage",
                        "high",
                        f"CPU usage is {self.system_metrics['cpu_usage']:.1f}%"
                    )
                
                # Check memory alert
                if self.system_metrics["memory_usage"] > self.monitoring_config["memory_threshold"]:
                    await self.create_alert(
                        "High Memory Usage",
                        "high",
                        f"Memory usage is {self.system_metrics['memory_usage']:.1f}%"
                    )
                
                # Check disk alert
                if self.system_metrics["disk_usage"] > self.monitoring_config["disk_threshold"]:
                    await self.create_alert(
                        "High Disk Usage",
                        "critical",
                        f"Disk usage is {self.system_metrics['disk_usage']:.1f}%"
                    )
                
                # Check response time alert
                if self.app_metrics["response_time_avg"] > self.monitoring_config["response_time_threshold"]:
                    await self.create_alert(
                        "High Response Time",
                        "medium",
                        f"Average response time is {self.app_metrics['response_time_avg']:.2f}s"
                    )
                
                # Check error rate alert
                if self.app_metrics["error_rate"] > self.monitoring_config["error_rate_threshold"]:
                    await self.create_alert(
                        "High Error Rate",
                        "critical",
                        f"Error rate is {self.app_metrics['error_rate']:.1f}%"
                    )
                
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _cleanup_old_metrics(self):
        """Cleanup old metrics to prevent memory issues"""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
                
                # Cleanup old alerts
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
                
            except Exception as e:
                logger.error(f"Error cleaning up old metrics: {e}")
            
            await asyncio.sleep(3600)  # Cleanup every hour

class AnalyticsService:
    """Analytics and reporting service"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.monitoring_system = MonitoringSystem(settings)
        
        # Analytics data
        self.user_analytics: Dict[str, Dict[str, Any]] = {}
        self.document_analytics: Dict[str, Dict[str, Any]] = {}
        self.feature_usage: Dict[str, int] = {}
        
        # Event tracking
        self.events: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Initialize analytics service"""
        try:
            await self.monitoring_system.initialize()
            logger.info("Analytics Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Analytics Service: {e}")
            raise
    
    async def track_event(self, event_type: str, user_id: str, data: Dict[str, Any]):
        """Track user event"""
        try:
            event = {
                "event_type": event_type,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            self.events.append(event)
            
            # Update feature usage
            if event_type in self.feature_usage:
                self.feature_usage[event_type] += 1
            else:
                self.feature_usage[event_type] = 1
            
            # Update user analytics
            if user_id not in self.user_analytics:
                self.user_analytics[user_id] = {
                    "total_events": 0,
                    "last_activity": datetime.utcnow(),
                    "feature_usage": {}
                }
            
            self.user_analytics[user_id]["total_events"] += 1
            self.user_analytics[user_id]["last_activity"] = datetime.utcnow()
            
            if event_type in self.user_analytics[user_id]["feature_usage"]:
                self.user_analytics[user_id]["feature_usage"][event_type] += 1
            else:
                self.user_analytics[user_id]["feature_usage"][event_type] = 1
            
            # Keep only last 10000 events
            if len(self.events) > 10000:
                self.events = self.events[-10000:]
            
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
    
    async def track_document_usage(self, document_id: str, action: str, user_id: str):
        """Track document usage"""
        try:
            if document_id not in self.document_analytics:
                self.document_analytics[document_id] = {
                    "views": 0,
                    "edits": 0,
                    "exports": 0,
                    "variants_generated": 0,
                    "topics_extracted": 0,
                    "brainstorm_ideas": 0,
                    "last_accessed": datetime.utcnow(),
                    "users": set()
                }
            
            doc_analytics = self.document_analytics[document_id]
            doc_analytics["last_accessed"] = datetime.utcnow()
            doc_analytics["users"].add(user_id)
            
            if action == "view":
                doc_analytics["views"] += 1
            elif action == "edit":
                doc_analytics["edits"] += 1
            elif action == "export":
                doc_analytics["exports"] += 1
            elif action == "variant_generated":
                doc_analytics["variants_generated"] += 1
            elif action == "topic_extracted":
                doc_analytics["topics_extracted"] += 1
            elif action == "brainstorm_idea":
                doc_analytics["brainstorm_ideas"] += 1
            
        except Exception as e:
            logger.error(f"Error tracking document usage: {e}")
    
    async def get_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard analytics data"""
        try:
            # Get user analytics
            user_data = self.user_analytics.get(user_id, {
                "total_events": 0,
                "last_activity": datetime.utcnow(),
                "feature_usage": {}
            })
            
            # Get system health
            system_health = await self.monitoring_system.get_system_health()
            
            # Get recent events
            recent_events = [e for e in self.events if e["user_id"] == user_id][-10:]
            
            # Get feature usage statistics
            total_events = sum(self.feature_usage.values())
            feature_stats = {
                feature: {
                    "count": count,
                    "percentage": (count / total_events * 100) if total_events > 0 else 0
                }
                for feature, count in self.feature_usage.items()
            }
            
            return {
                "user_analytics": user_data,
                "system_health": system_health,
                "recent_events": recent_events,
                "feature_usage": feature_stats,
                "total_users": len(self.user_analytics),
                "total_documents": len(self.document_analytics),
                "total_events": total_events
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    async def generate_report(self, start_date: datetime, end_date: datetime, user_id: str) -> Dict[str, Any]:
        """Generate analytics report for date range"""
        try:
            # Filter events by date range
            filtered_events = [
                e for e in self.events
                if start_date <= datetime.fromisoformat(e["timestamp"]) <= end_date
                and e["user_id"] == user_id
            ]
            
            # Calculate metrics
            total_events = len(filtered_events)
            unique_days = len(set(e["timestamp"][:10] for e in filtered_events))
            
            # Event type breakdown
            event_types = {}
            for event in filtered_events:
                event_type = event["event_type"]
                if event_type in event_types:
                    event_types[event_type] += 1
                else:
                    event_types[event_type] = 1
            
            # Daily activity
            daily_activity = {}
            for event in filtered_events:
                date = event["timestamp"][:10]
                if date in daily_activity:
                    daily_activity[date] += 1
                else:
                    daily_activity[date] = 1
            
            return {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_events": total_events,
                    "unique_days": unique_days,
                    "average_events_per_day": total_events / unique_days if unique_days > 0 else 0
                },
                "event_types": event_types,
                "daily_activity": daily_activity,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {}
    
    async def get_feature_usage_stats(self) -> Dict[str, Any]:
        """Get feature usage statistics"""
        try:
            total_usage = sum(self.feature_usage.values())
            
            return {
                "total_usage": total_usage,
                "feature_breakdown": self.feature_usage,
                "top_features": sorted(
                    self.feature_usage.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
            
        except Exception as e:
            logger.error(f"Error getting feature usage stats: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup analytics service"""
        try:
            await self.monitoring_system.cleanup()
            logger.info("Analytics Service cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up Analytics Service: {e}")

class HealthService:
    """Health monitoring service"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.monitoring_system = MonitoringSystem(settings)
        
        # Service dependencies
        self.dependencies: Dict[str, Dict[str, Any]] = {
            "database": {"status": "unknown", "last_check": None},
            "redis": {"status": "unknown", "last_check": None},
            "ai_services": {"status": "unknown", "last_check": None},
            "file_storage": {"status": "unknown", "last_check": None}
        }
    
    async def initialize(self):
        """Initialize health service"""
        try:
            await self.monitoring_system.initialize()
            logger.info("Health Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Health Service: {e}")
            raise
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Get monitoring system health
            monitoring_health = await self.monitoring_system.get_system_health()
            
            # Check service dependencies
            await self._check_dependencies()
            
            # Determine overall health
            overall_status = "healthy"
            if monitoring_health["status"] == "unhealthy":
                overall_status = "unhealthy"
            elif monitoring_health["status"] == "degraded":
                overall_status = "degraded"
            
            # Check if any dependencies are down
            for dep_name, dep_info in self.dependencies.items():
                if dep_info["status"] == "down":
                    overall_status = "unhealthy"
                    break
                elif dep_info["status"] == "degraded":
                    overall_status = "degraded"
            
            return {
                "status": overall_status,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self.monitoring_system.start_time,
                "version": self.settings.APP_VERSION,
                "environment": self.settings.ENVIRONMENT,
                "monitoring": monitoring_health,
                "dependencies": self.dependencies,
                "services": {
                    "api": "healthy",
                    "pdf_processing": "healthy",
                    "ai_services": "healthy",
                    "collaboration": "healthy",
                    "export": "healthy"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": 0,
                "version": self.settings.APP_VERSION,
                "environment": self.settings.ENVIRONMENT,
                "error": str(e)
            }
    
    async def _check_dependencies(self):
        """Check health of service dependencies"""
        try:
            current_time = datetime.utcnow()
            
            # Check database (placeholder)
            self.dependencies["database"]["status"] = "up"
            self.dependencies["database"]["last_check"] = current_time
            
            # Check Redis (placeholder)
            self.dependencies["redis"]["status"] = "up"
            self.dependencies["redis"]["last_check"] = current_time
            
            # Check AI services (placeholder)
            self.dependencies["ai_services"]["status"] = "up"
            self.dependencies["ai_services"]["last_check"] = current_time
            
            # Check file storage (placeholder)
            self.dependencies["file_storage"]["status"] = "up"
            self.dependencies["file_storage"]["last_check"] = current_time
            
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
    
    async def cleanup(self):
        """Cleanup health service"""
        try:
            await self.monitoring_system.cleanup()
            logger.info("Health Service cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up Health Service: {e}")

class NotificationService:
    """Notification service for alerts and events"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.notifications: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Initialize notification service"""
        try:
            logger.info("Notification Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Notification Service: {e}")
            raise
    
    async def send_notification(self, user_id: str, notification_type: str, 
                              title: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Send notification to user"""
        try:
            notification = {
                "id": f"notif_{int(time.time())}",
                "user_id": user_id,
                "type": notification_type,
                "title": title,
                "message": message,
                "data": data or {},
                "timestamp": datetime.utcnow().isoformat(),
                "read": False
            }
            
            self.notifications.append(notification)
            
            # Keep only last 1000 notifications
            if len(self.notifications) > 1000:
                self.notifications = self.notifications[-1000:]
            
            logger.info(f"Notification sent to user {user_id}: {title}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get notifications for user"""
        try:
            user_notifications = [
                n for n in self.notifications 
                if n["user_id"] == user_id
            ]
            
            if unread_only:
                user_notifications = [n for n in user_notifications if not n["read"]]
            
            return sorted(user_notifications, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting user notifications: {e}")
            return []
    
    async def mark_notification_read(self, notification_id: str, user_id: str) -> bool:
        """Mark notification as read"""
        try:
            for notification in self.notifications:
                if notification["id"] == notification_id and notification["user_id"] == user_id:
                    notification["read"] = True
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error marking notification as read: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup notification service"""
        try:
            logger.info("Notification Service cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up Notification Service: {e}")
