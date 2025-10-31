"""
Advanced Monitoring Service for comprehensive system monitoring and alerting
"""

import asyncio
import psutil
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text
from dataclasses import dataclass
import json

from ..models.database import SystemMetric, Alert, PerformanceLog
from ..core.exceptions import DatabaseError, ValidationError


@dataclass
class MetricThreshold:
    """Metric threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    unit: str
    description: str


class AdvancedMonitoringService:
    """Service for advanced system monitoring and alerting."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.metrics_cache = {}
        self.alert_rules = {}
        self.monitoring_enabled = True
        
        # Initialize metric thresholds
        self.thresholds = {
            "cpu_usage": MetricThreshold("cpu_usage", 70.0, 90.0, "%", "CPU usage percentage"),
            "memory_usage": MetricThreshold("memory_usage", 80.0, 95.0, "%", "Memory usage percentage"),
            "disk_usage": MetricThreshold("disk_usage", 85.0, 95.0, "%", "Disk usage percentage"),
            "response_time": MetricThreshold("response_time", 1000.0, 5000.0, "ms", "API response time"),
            "error_rate": MetricThreshold("error_rate", 5.0, 10.0, "%", "Error rate percentage"),
            "active_connections": MetricThreshold("active_connections", 1000, 2000, "count", "Active database connections"),
            "cache_hit_rate": MetricThreshold("cache_hit_rate", 80.0, 90.0, "%", "Cache hit rate"),
            "queue_size": MetricThreshold("queue_size", 100, 500, "count", "Background task queue size")
        }
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        try:
            metrics = {}
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            metrics["cpu"] = {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics["memory"] = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent,
                "swap_total_gb": round(swap.total / (1024**3), 2),
                "swap_used_gb": round(swap.used / (1024**3), 2),
                "swap_usage_percent": swap.percent
            }
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics["disk"] = {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "usage_percent": round((disk.used / disk.total) * 100, 2),
                "read_bytes_per_sec": disk_io.read_bytes if disk_io else 0,
                "write_bytes_per_sec": disk_io.write_bytes if disk_io else 0
            }
            
            # Network metrics
            network_io = psutil.net_io_counters()
            network_connections = len(psutil.net_connections())
            
            metrics["network"] = {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
                "active_connections": network_connections
            }
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            metrics["process"] = {
                "pid": process.pid,
                "memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                "memory_vms_mb": round(process_memory.vms / (1024**2), 2),
                "cpu_percent": process_cpu,
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time())
            }
            
            # Database metrics (mock implementation)
            metrics["database"] = await self._collect_database_metrics()
            
            # Cache metrics (mock implementation)
            metrics["cache"] = await self._collect_cache_metrics()
            
            # Application metrics
            metrics["application"] = await self._collect_application_metrics()
            
            # Store metrics in database
            await self._store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            raise DatabaseError(f"Failed to collect system metrics: {str(e)}")
    
    async def _collect_database_metrics(self) -> Dict[str, Any]:
        """Collect database-specific metrics."""
        try:
            # Get database connection count
            connection_count_query = select(func.count(text("1")))
            connection_count_result = await self.session.execute(connection_count_query)
            connection_count = connection_count_result.scalar()
            
            # Get database size (mock implementation)
            db_size_mb = 1024  # Mock value
            
            # Get slow query count (mock implementation)
            slow_queries = 5  # Mock value
            
            return {
                "active_connections": connection_count,
                "database_size_mb": db_size_mb,
                "slow_queries_count": slow_queries,
                "connection_pool_size": 20,  # Mock value
                "connection_pool_usage": round((connection_count / 20) * 100, 2)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect cache-specific metrics."""
        try:
            # Mock cache metrics
            return {
                "hit_rate_percent": 85.5,
                "miss_rate_percent": 14.5,
                "total_keys": 10000,
                "memory_usage_mb": 256,
                "evictions_per_second": 5,
                "expired_keys_per_second": 10
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        try:
            # Get recent performance logs
            recent_logs_query = select(PerformanceLog).where(
                PerformanceLog.timestamp >= datetime.utcnow() - timedelta(minutes=5)
            )
            recent_logs_result = await self.session.execute(recent_logs_query)
            recent_logs = recent_logs_result.scalars().all()
            
            # Calculate metrics
            if recent_logs:
                response_times = [log.response_time for log in recent_logs]
                error_count = sum(1 for log in recent_logs if log.status_code >= 400)
                
                avg_response_time = sum(response_times) / len(response_times)
                error_rate = (error_count / len(recent_logs)) * 100
            else:
                avg_response_time = 0
                error_rate = 0
            
            return {
                "requests_per_minute": len(recent_logs),
                "average_response_time_ms": round(avg_response_time, 2),
                "error_rate_percent": round(error_rate, 2),
                "active_sessions": 150,  # Mock value
                "background_tasks_running": 5,  # Mock value
                "queue_size": 25  # Mock value
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in database."""
        try:
            system_metric = SystemMetric(
                metrics_data=metrics,
                timestamp=datetime.utcnow()
            )
            
            self.session.add(system_metric)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to store metrics: {str(e)}")
    
    async def check_metric_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check metrics against thresholds and generate alerts."""
        try:
            alerts = []
            
            # Check CPU usage
            cpu_usage = metrics.get("cpu", {}).get("usage_percent", 0)
            cpu_threshold = self.thresholds["cpu_usage"]
            
            if cpu_usage >= cpu_threshold.critical_threshold:
                alerts.append({
                    "metric": "cpu_usage",
                    "value": cpu_usage,
                    "threshold": cpu_threshold.critical_threshold,
                    "severity": "critical",
                    "message": f"CPU usage is critically high: {cpu_usage}%"
                })
            elif cpu_usage >= cpu_threshold.warning_threshold:
                alerts.append({
                    "metric": "cpu_usage",
                    "value": cpu_usage,
                    "threshold": cpu_threshold.warning_threshold,
                    "severity": "warning",
                    "message": f"CPU usage is high: {cpu_usage}%"
                })
            
            # Check memory usage
            memory_usage = metrics.get("memory", {}).get("usage_percent", 0)
            memory_threshold = self.thresholds["memory_usage"]
            
            if memory_usage >= memory_threshold.critical_threshold:
                alerts.append({
                    "metric": "memory_usage",
                    "value": memory_usage,
                    "threshold": memory_threshold.critical_threshold,
                    "severity": "critical",
                    "message": f"Memory usage is critically high: {memory_usage}%"
                })
            elif memory_usage >= memory_threshold.warning_threshold:
                alerts.append({
                    "metric": "memory_usage",
                    "value": memory_usage,
                    "threshold": memory_threshold.warning_threshold,
                    "severity": "warning",
                    "message": f"Memory usage is high: {memory_usage}%"
                })
            
            # Check disk usage
            disk_usage = metrics.get("disk", {}).get("usage_percent", 0)
            disk_threshold = self.thresholds["disk_usage"]
            
            if disk_usage >= disk_threshold.critical_threshold:
                alerts.append({
                    "metric": "disk_usage",
                    "value": disk_usage,
                    "threshold": disk_threshold.critical_threshold,
                    "severity": "critical",
                    "message": f"Disk usage is critically high: {disk_usage}%"
                })
            elif disk_usage >= disk_threshold.warning_threshold:
                alerts.append({
                    "metric": "disk_usage",
                    "value": disk_usage,
                    "threshold": disk_threshold.warning_threshold,
                    "severity": "warning",
                    "message": f"Disk usage is high: {disk_usage}%"
                })
            
            # Check response time
            response_time = metrics.get("application", {}).get("average_response_time_ms", 0)
            response_threshold = self.thresholds["response_time"]
            
            if response_time >= response_threshold.critical_threshold:
                alerts.append({
                    "metric": "response_time",
                    "value": response_time,
                    "threshold": response_threshold.critical_threshold,
                    "severity": "critical",
                    "message": f"Response time is critically slow: {response_time}ms"
                })
            elif response_time >= response_threshold.warning_threshold:
                alerts.append({
                    "metric": "response_time",
                    "value": response_time,
                    "threshold": response_threshold.warning_threshold,
                    "severity": "warning",
                    "message": f"Response time is slow: {response_time}ms"
                })
            
            # Check error rate
            error_rate = metrics.get("application", {}).get("error_rate_percent", 0)
            error_threshold = self.thresholds["error_rate"]
            
            if error_rate >= error_threshold.critical_threshold:
                alerts.append({
                    "metric": "error_rate",
                    "value": error_rate,
                    "threshold": error_threshold.critical_threshold,
                    "severity": "critical",
                    "message": f"Error rate is critically high: {error_rate}%"
                })
            elif error_rate >= error_threshold.warning_threshold:
                alerts.append({
                    "metric": "error_rate",
                    "value": error_rate,
                    "threshold": error_threshold.warning_threshold,
                    "severity": "warning",
                    "message": f"Error rate is high: {error_rate}%"
                })
            
            # Store alerts in database
            for alert_data in alerts:
                await self._store_alert(alert_data)
            
            return alerts
            
        except Exception as e:
            raise DatabaseError(f"Failed to check metric thresholds: {str(e)}")
    
    async def _store_alert(self, alert_data: Dict[str, Any]):
        """Store alert in database."""
        try:
            alert = Alert(
                metric_name=alert_data["metric"],
                metric_value=alert_data["value"],
                threshold_value=alert_data["threshold"],
                severity=alert_data["severity"],
                message=alert_data["message"],
                timestamp=datetime.utcnow(),
                resolved=False
            )
            
            self.session.add(alert)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to store alert: {str(e)}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            # Collect current metrics
            metrics = await self.collect_system_metrics()
            
            # Check thresholds
            alerts = await self.check_metric_thresholds(metrics)
            
            # Calculate health score
            health_score = self._calculate_health_score(metrics, alerts)
            
            # Determine overall status
            if health_score >= 90:
                status = "excellent"
            elif health_score >= 70:
                status = "good"
            elif health_score >= 50:
                status = "fair"
            else:
                status = "poor"
            
            return {
                "status": status,
                "health_score": health_score,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "active_alerts": len([a for a in alerts if a["severity"] == "critical"]),
                "warnings": len([a for a in alerts if a["severity"] == "warning"]),
                "alerts": alerts
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get system health: {str(e)}")
    
    def _calculate_health_score(self, metrics: Dict[str, Any], alerts: List[Dict[str, Any]]) -> float:
        """Calculate overall health score."""
        try:
            score = 100.0
            
            # Deduct points for critical alerts
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]
            score -= len(critical_alerts) * 20
            
            # Deduct points for warning alerts
            warning_alerts = [a for a in alerts if a["severity"] == "warning"]
            score -= len(warning_alerts) * 10
            
            # Deduct points for high resource usage
            cpu_usage = metrics.get("cpu", {}).get("usage_percent", 0)
            if cpu_usage > 80:
                score -= (cpu_usage - 80) * 0.5
            
            memory_usage = metrics.get("memory", {}).get("usage_percent", 0)
            if memory_usage > 80:
                score -= (memory_usage - 80) * 0.5
            
            disk_usage = metrics.get("disk", {}).get("usage_percent", 0)
            if disk_usage > 80:
                score -= (disk_usage - 80) * 0.5
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 0.0
    
    async def get_metrics_history(
        self,
        hours: int = 24,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metrics history for specified time period."""
        try:
            since_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Build query
            query = select(SystemMetric).where(SystemMetric.timestamp >= since_time)
            
            if metric_name:
                # Filter by specific metric (this would require JSON querying)
                pass
            
            # Get metrics
            query = query.order_by(desc(SystemMetric.timestamp))
            metrics_result = await self.session.execute(query)
            metrics = metrics_result.scalars().all()
            
            # Format results
            metrics_history = []
            for metric in metrics:
                metrics_history.append({
                    "timestamp": metric.timestamp,
                    "metrics": metric.metrics_data
                })
            
            return {
                "period_hours": hours,
                "metric_name": metric_name,
                "data_points": len(metrics_history),
                "history": metrics_history
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get metrics history: {str(e)}")
    
    async def get_active_alerts(
        self,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get active alerts."""
        try:
            # Build query
            query = select(Alert).where(Alert.resolved == False)
            
            if severity:
                query = query.where(Alert.severity == severity)
            
            # Get alerts
            query = query.order_by(desc(Alert.timestamp)).limit(limit)
            alerts_result = await self.session.execute(query)
            alerts = alerts_result.scalars().all()
            
            # Format results
            alert_list = []
            for alert in alerts:
                alert_list.append({
                    "id": alert.id,
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                    "threshold_value": alert.threshold_value,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved
                })
            
            return {
                "alerts": alert_list,
                "total": len(alert_list),
                "severity_filter": severity
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get active alerts: {str(e)}")
    
    async def resolve_alert(self, alert_id: int) -> Dict[str, Any]:
        """Resolve an alert."""
        try:
            # Get alert
            alert_query = select(Alert).where(Alert.id == alert_id)
            alert_result = await self.session.execute(alert_query)
            alert = alert_result.scalar_one_or_none()
            
            if not alert:
                raise ValidationError("Alert not found")
            
            # Mark as resolved
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            await self.session.commit()
            
            return {
                "alert_id": alert_id,
                "resolved": True,
                "resolved_at": alert.resolved_at
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to resolve alert: {str(e)}")
    
    async def update_threshold(self, metric_name: str, warning_threshold: float, critical_threshold: float) -> Dict[str, Any]:
        """Update metric threshold."""
        try:
            if metric_name not in self.thresholds:
                raise ValidationError(f"Unknown metric: {metric_name}")
            
            old_threshold = self.thresholds[metric_name]
            self.thresholds[metric_name].warning_threshold = warning_threshold
            self.thresholds[metric_name].critical_threshold = critical_threshold
            
            return {
                "metric_name": metric_name,
                "old_warning_threshold": old_threshold.warning_threshold,
                "new_warning_threshold": warning_threshold,
                "old_critical_threshold": old_threshold.critical_threshold,
                "new_critical_threshold": critical_threshold,
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise ValidationError(f"Failed to update threshold: {str(e)}")
    
    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        try:
            # Get total metrics collected
            total_metrics_query = select(func.count(SystemMetric.id))
            total_metrics_result = await self.session.execute(total_metrics_query)
            total_metrics = total_metrics_result.scalar()
            
            # Get total alerts
            total_alerts_query = select(func.count(Alert.id))
            total_alerts_result = await self.session.execute(total_alerts_query)
            total_alerts = total_alerts_result.scalar()
            
            # Get active alerts
            active_alerts_query = select(func.count(Alert.id)).where(Alert.resolved == False)
            active_alerts_result = await self.session.execute(active_alerts_query)
            active_alerts = active_alerts_result.scalar()
            
            # Get alerts by severity
            alerts_by_severity_query = select(
                Alert.severity,
                func.count(Alert.id).label('count')
            ).group_by(Alert.severity)
            
            alerts_by_severity_result = await self.session.execute(alerts_by_severity_query)
            alerts_by_severity = dict(alerts_by_severity_result.all())
            
            return {
                "total_metrics_collected": total_metrics,
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "resolved_alerts": total_alerts - active_alerts,
                "alerts_by_severity": alerts_by_severity,
                "monitoring_enabled": self.monitoring_enabled,
                "thresholds_configured": len(self.thresholds)
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get monitoring stats: {str(e)}")
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring."""
        try:
            self.monitoring_enabled = True
            
            while self.monitoring_enabled:
                # Collect metrics
                metrics = await self.collect_system_metrics()
                
                # Check thresholds
                await self.check_metric_thresholds(metrics)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            raise DatabaseError(f"Monitoring failed: {str(e)}")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_enabled = False
    
    async def log_performance(self, endpoint: str, response_time: float, status_code: int, user_id: Optional[str] = None):
        """Log API performance metrics."""
        try:
            performance_log = PerformanceLog(
                endpoint=endpoint,
                response_time=response_time,
                status_code=status_code,
                user_id=user_id,
                timestamp=datetime.utcnow()
            )
            
            self.session.add(performance_log)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to log performance: {str(e)}")
    
    async def get_performance_analytics(
        self,
        hours: int = 24,
        endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance analytics."""
        try:
            since_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Build query
            query = select(PerformanceLog).where(PerformanceLog.timestamp >= since_time)
            
            if endpoint:
                query = query.where(PerformanceLog.endpoint == endpoint)
            
            # Get performance logs
            logs_result = await self.session.execute(query)
            logs = logs_result.scalars().all()
            
            if not logs:
                return {
                    "period_hours": hours,
                    "endpoint": endpoint,
                    "total_requests": 0,
                    "average_response_time": 0,
                    "error_rate": 0,
                    "status_codes": {}
                }
            
            # Calculate analytics
            total_requests = len(logs)
            response_times = [log.response_time for log in logs]
            average_response_time = sum(response_times) / len(response_times)
            
            error_count = sum(1 for log in logs if log.status_code >= 400)
            error_rate = (error_count / total_requests) * 100
            
            # Status code distribution
            status_codes = {}
            for log in logs:
                status_code = log.status_code
                status_codes[status_code] = status_codes.get(status_code, 0) + 1
            
            return {
                "period_hours": hours,
                "endpoint": endpoint,
                "total_requests": total_requests,
                "average_response_time": round(average_response_time, 2),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "error_rate": round(error_rate, 2),
                "status_codes": status_codes
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get performance analytics: {str(e)}")

























