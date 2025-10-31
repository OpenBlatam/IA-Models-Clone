"""
Advanced Monitoring System for OpusClip Improved
===============================================

Comprehensive monitoring with Prometheus, Grafana, and custom metrics.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import aiohttp
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
import structlog

from .schemas import get_settings
from .exceptions import MonitoringError, create_monitoring_error

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricData:
    """Metric data structure"""
    name: str
    value: float
    labels: Dict[str, str] = None
    timestamp: datetime = None
    help_text: str = ""


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str
    threshold: float
    severity: str  # "critical", "warning", "info"
    description: str
    enabled: bool = True


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable
    interval: int = 60  # seconds
    timeout: int = 30  # seconds
    enabled: bool = True


class PrometheusMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Video processing metrics
        self.video_processing_total = Counter(
            'video_processing_total',
            'Total video processing operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.video_processing_duration = Histogram(
            'video_processing_duration_seconds',
            'Video processing duration',
            ['operation', 'quality'],
            registry=self.registry
        )
        
        self.video_processing_size = Histogram(
            'video_processing_size_bytes',
            'Video processing file size',
            ['operation'],
            registry=self.registry
        )
        
        # AI metrics
        self.ai_requests_total = Counter(
            'ai_requests_total',
            'Total AI requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.ai_request_duration = Histogram(
            'ai_request_duration_seconds',
            'AI request duration',
            ['provider', 'model'],
            registry=self.registry
        )
        
        self.ai_tokens_used = Counter(
            'ai_tokens_used_total',
            'Total AI tokens used',
            ['provider', 'model', 'type'],
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage',
            ['device'],
            registry=self.registry
        )
        
        self.system_load_average = Gauge(
            'system_load_average',
            'System load average',
            ['period'],
            registry=self.registry
        )
        
        # Database metrics
        self.database_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.database_queries_total = Counter(
            'database_queries_total',
            'Total database queries',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['operation', 'table'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'cache_type', 'status'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )
        
        # Storage metrics
        self.storage_operations_total = Counter(
            'storage_operations_total',
            'Total storage operations',
            ['operation', 'storage_type', 'status'],
            registry=self.registry
        )
        
        self.storage_usage = Gauge(
            'storage_usage_bytes',
            'Storage usage',
            ['storage_type'],
            registry=self.registry
        )
        
        # Business metrics
        self.active_users = Gauge(
            'active_users_total',
            'Total active users',
            registry=self.registry
        )
        
        self.projects_created = Counter(
            'projects_created_total',
            'Total projects created',
            registry=self.registry
        )
        
        self.videos_processed = Counter(
            'videos_processed_total',
            'Total videos processed',
            ['quality'],
            registry=self.registry
        )
        
        self.clips_generated = Counter(
            'clips_generated_total',
            'Total clips generated',
            ['platform'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        self.error_rate = Gauge(
            'error_rate',
            'Error rate percentage',
            ['component'],
            registry=self.registry
        )
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float, request_size: int = 0, response_size: int = 0):
        """Record HTTP request metrics"""
        self.http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        if request_size > 0:
            self.http_request_size.labels(method=method, endpoint=endpoint).observe(request_size)
        
        if response_size > 0:
            self.http_response_size.labels(method=method, endpoint=endpoint).observe(response_size)
    
    def record_video_processing(self, operation: str, status: str, duration: float, file_size: int = 0, quality: str = "unknown"):
        """Record video processing metrics"""
        self.video_processing_total.labels(operation=operation, status=status).inc()
        self.video_processing_duration.labels(operation=operation, quality=quality).observe(duration)
        
        if file_size > 0:
            self.video_processing_size.labels(operation=operation).observe(file_size)
    
    def record_ai_request(self, provider: str, model: str, status: str, duration: float, tokens_used: int = 0, token_type: str = "unknown"):
        """Record AI request metrics"""
        self.ai_requests_total.labels(provider=provider, model=model, status=status).inc()
        self.ai_request_duration.labels(provider=provider, model=model).observe(duration)
        
        if tokens_used > 0:
            self.ai_tokens_used.labels(provider=provider, model=model, type=token_type).inc(tokens_used)
    
    def record_database_operation(self, operation: str, table: str, duration: float):
        """Record database operation metrics"""
        self.database_queries_total.labels(operation=operation, table=table).inc()
        self.database_query_duration.labels(operation=operation, table=table).observe(duration)
    
    def record_cache_operation(self, operation: str, cache_type: str, status: str):
        """Record cache operation metrics"""
        self.cache_operations_total.labels(operation=operation, cache_type=cache_type, status=status).inc()
    
    def record_storage_operation(self, operation: str, storage_type: str, status: str):
        """Record storage operation metrics"""
        self.storage_operations_total.labels(operation=operation, storage_type=storage_type, status=status).inc()
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics"""
        self.errors_total.labels(error_type=error_type, component=component).inc()
    
    def update_system_metrics(self):
        """Update system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.system_disk_usage.labels(device=partition.device).set(usage.used)
                except PermissionError:
                    pass
            
            # Load average
            load_avg = psutil.getloadavg()
            self.system_load_average.labels(period="1m").set(load_avg[0])
            self.system_load_average.labels(period="5m").set(load_avg[1])
            self.system_load_average.labels(period="15m").set(load_avg[2])
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry)


class HealthChecker:
    """Health check system"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.last_results: Dict[str, Dict[str, Any]] = {}
        self.logger = structlog.get_logger("health_checker")
    
    def register_health_check(self, name: str, check_function: Callable, interval: int = 60, timeout: int = 30):
        """Register a health check"""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout
        )
        self.last_results[name] = {
            "status": "unknown",
            "last_check": None,
            "error": None
        }
    
    async def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check"""
        if name not in self.health_checks:
            raise ValueError(f"Health check '{name}' not found")
        
        health_check = self.health_checks[name]
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                health_check.check_function(),
                timeout=health_check.timeout
            )
            
            self.last_results[name] = {
                "status": "healthy" if result else "unhealthy",
                "last_check": datetime.utcnow(),
                "error": None
            }
            
            return self.last_results[name]
            
        except asyncio.TimeoutError:
            error_msg = f"Health check '{name}' timed out after {health_check.timeout}s"
            self.last_results[name] = {
                "status": "unhealthy",
                "last_check": datetime.utcnow(),
                "error": error_msg
            }
            self.logger.error(error_msg)
            return self.last_results[name]
            
        except Exception as e:
            error_msg = f"Health check '{name}' failed: {str(e)}"
            self.last_results[name] = {
                "status": "unhealthy",
                "last_check": datetime.utcnow(),
                "error": error_msg
            }
            self.logger.error(error_msg)
            return self.last_results[name]
    
    async def run_all_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks"""
        tasks = []
        for name in self.health_checks.keys():
            tasks.append(self.run_health_check(name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update results
        for i, name in enumerate(self.health_checks.keys()):
            if isinstance(results[i], Exception):
                self.last_results[name] = {
                    "status": "unhealthy",
                    "last_check": datetime.utcnow(),
                    "error": str(results[i])
                }
        
        return self.last_results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.last_results:
            return {"status": "unknown", "checks": {}}
        
        healthy_count = sum(1 for result in self.last_results.values() if result["status"] == "healthy")
        total_count = len(self.last_results)
        
        if healthy_count == total_count:
            status = "healthy"
        elif healthy_count > 0:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "healthy_checks": healthy_count,
            "total_checks": total_count,
            "checks": self.last_results
        }


class AlertManager:
    """Alert management system"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.logger = structlog.get_logger("alert_manager")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
    
    async def check_alerts(self, metrics: Dict[str, float]):
        """Check alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Evaluate alert condition
                should_alert = self._evaluate_condition(rule.condition, metrics, rule.threshold)
                
                if should_alert:
                    await self._trigger_alert(rule)
                else:
                    await self._resolve_alert(rule_name)
                    
            except Exception as e:
                self.logger.error(f"Failed to check alert rule '{rule_name}': {e}")
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, float], threshold: float) -> bool:
        """Evaluate alert condition"""
        # Simple condition evaluation - in production, use a proper expression evaluator
        if ">" in condition:
            metric_name = condition.split(">")[0].strip()
            if metric_name in metrics:
                return metrics[metric_name] > threshold
        elif "<" in condition:
            metric_name = condition.split("<")[0].strip()
            if metric_name in metrics:
                return metrics[metric_name] < threshold
        elif "==" in condition:
            metric_name = condition.split("==")[0].strip()
            if metric_name in metrics:
                return metrics[metric_name] == threshold
        
        return False
    
    async def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert"""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        alert_data = {
            "alert_id": alert_id,
            "rule_name": rule.name,
            "severity": rule.severity,
            "description": rule.description,
            "condition": rule.condition,
            "threshold": rule.threshold,
            "triggered_at": datetime.utcnow(),
            "status": "active"
        }
        
        self.active_alerts[rule.name] = alert_data
        self.alert_history.append(alert_data)
        
        # Log alert
        self.logger.warning(
            f"Alert triggered: {rule.name}",
            severity=rule.severity,
            description=rule.description
        )
        
        # Send notification (placeholder)
        await self._send_alert_notification(alert_data)
    
    async def _resolve_alert(self, rule_name: str):
        """Resolve an alert"""
        if rule_name in self.active_alerts:
            alert_data = self.active_alerts[rule_name]
            alert_data["resolved_at"] = datetime.utcnow()
            alert_data["status"] = "resolved"
            
            del self.active_alerts[rule_name]
            
            self.logger.info(f"Alert resolved: {rule_name}")
    
    async def _send_alert_notification(self, alert_data: Dict[str, Any]):
        """Send alert notification"""
        # Placeholder for notification sending
        # In production, integrate with Slack, email, PagerDuty, etc.
        self.logger.info(f"Alert notification sent: {alert_data['rule_name']}")


class PerformanceProfiler:
    """Performance profiling system"""
    
    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
    
    def start_profile(self, name: str, metadata: Dict[str, Any] = None):
        """Start profiling a function or operation"""
        profile_id = f"{name}_{int(time.time() * 1000)}"
        
        self.active_profiles[profile_id] = {
            "name": name,
            "start_time": time.time(),
            "start_memory": psutil.Process().memory_info().rss,
            "metadata": metadata or {}
        }
        
        return profile_id
    
    def end_profile(self, profile_id: str) -> Dict[str, Any]:
        """End profiling and return results"""
        if profile_id not in self.active_profiles:
            raise ValueError(f"Profile '{profile_id}' not found")
        
        profile_data = self.active_profiles[profile_id]
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        result = {
            "profile_id": profile_id,
            "name": profile_data["name"],
            "duration": end_time - profile_data["start_time"],
            "memory_delta": end_memory - profile_data["start_memory"],
            "start_time": profile_data["start_time"],
            "end_time": end_time,
            "metadata": profile_data["metadata"]
        }
        
        # Store in profiles
        if profile_data["name"] not in self.profiles:
            self.profiles[profile_data["name"]] = []
        
        self.profiles[profile_data["name"]].append(result)
        
        # Keep only last 100 profiles per name
        if len(self.profiles[profile_data["name"]]) > 100:
            self.profiles[profile_data["name"]] = self.profiles[profile_data["name"]][-100:]
        
        del self.active_profiles[profile_id]
        return result
    
    def get_profile_stats(self, name: str) -> Dict[str, Any]:
        """Get profiling statistics for a name"""
        if name not in self.profiles or not self.profiles[name]:
            return {}
        
        durations = [p["duration"] for p in self.profiles[name]]
        memory_deltas = [p["memory_delta"] for p in self.profiles[name]]
        
        return {
            "name": name,
            "count": len(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
            "min_memory_delta": min(memory_deltas),
            "max_memory_delta": max(memory_deltas)
        }


class MonitoringSystem:
    """Main monitoring system"""
    
    def __init__(self):
        self.settings = get_settings()
        self.metrics = PrometheusMetrics()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        self.profiler = PerformanceProfiler()
        self.logger = structlog.get_logger("monitoring_system")
        
        self._initialize_health_checks()
        self._initialize_alert_rules()
    
    def _initialize_health_checks(self):
        """Initialize health checks"""
        # Database health check
        self.health_checker.register_health_check(
            "database",
            self._check_database_health,
            interval=30
        )
        
        # Redis health check
        self.health_checker.register_health_check(
            "redis",
            self._check_redis_health,
            interval=30
        )
        
        # Storage health check
        self.health_checker.register_health_check(
            "storage",
            self._check_storage_health,
            interval=60
        )
        
        # AI services health check
        self.health_checker.register_health_check(
            "ai_services",
            self._check_ai_services_health,
            interval=120
        )
    
    def _initialize_alert_rules(self):
        """Initialize alert rules"""
        # CPU usage alert
        self.alert_manager.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            condition="system_cpu_usage_percent",
            threshold=80.0,
            severity="warning",
            description="CPU usage is above 80%"
        ))
        
        # Memory usage alert
        self.alert_manager.add_alert_rule(AlertRule(
            name="high_memory_usage",
            condition="system_memory_usage_bytes",
            threshold=8 * 1024 * 1024 * 1024,  # 8GB
            severity="warning",
            description="Memory usage is above 8GB"
        ))
        
        # Error rate alert
        self.alert_manager.add_alert_rule(AlertRule(
            name="high_error_rate",
            condition="error_rate",
            threshold=5.0,
            severity="critical",
            description="Error rate is above 5%"
        ))
        
        # Video processing time alert
        self.alert_manager.add_alert_rule(AlertRule(
            name="slow_video_processing",
            condition="video_processing_duration_seconds",
            threshold=300.0,  # 5 minutes
            severity="warning",
            description="Video processing is taking too long"
        ))
    
    async def _check_database_health(self) -> bool:
        """Check database health"""
        try:
            # Placeholder - implement actual database health check
            return True
        except Exception:
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check Redis health"""
        try:
            # Placeholder - implement actual Redis health check
            return True
        except Exception:
            return False
    
    async def _check_storage_health(self) -> bool:
        """Check storage health"""
        try:
            # Placeholder - implement actual storage health check
            return True
        except Exception:
            return False
    
    async def _check_ai_services_health(self) -> bool:
        """Check AI services health"""
        try:
            # Placeholder - implement actual AI services health check
            return True
        except Exception:
            return False
    
    async def start_monitoring(self):
        """Start monitoring system"""
        self.logger.info("Starting monitoring system")
        
        # Start background tasks
        asyncio.create_task(self._update_system_metrics())
        asyncio.create_task(self._run_health_checks())
        asyncio.create_task(self._check_alerts())
    
    async def _update_system_metrics(self):
        """Update system metrics periodically"""
        while True:
            try:
                self.metrics.update_system_metrics()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                self.logger.error(f"Failed to update system metrics: {e}")
                await asyncio.sleep(60)
    
    async def _run_health_checks(self):
        """Run health checks periodically"""
        while True:
            try:
                await self.health_checker.run_all_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Failed to run health checks: {e}")
                await asyncio.sleep(30)
    
    async def _check_alerts(self):
        """Check alerts periodically"""
        while True:
            try:
                # Get current metrics
                current_metrics = {
                    "system_cpu_usage_percent": psutil.cpu_percent(),
                    "system_memory_usage_bytes": psutil.virtual_memory().used,
                    "error_rate": 0.0,  # Placeholder
                    "video_processing_duration_seconds": 0.0  # Placeholder
                }
                
                await self.alert_manager.check_alerts(current_metrics)
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Failed to check alerts: {e}")
                await asyncio.sleep(60)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        return self.metrics.get_metrics()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        return self.health_checker.get_overall_health()
    
    def get_active_alerts(self) -> Dict[str, Dict[str, Any]]:
        """Get active alerts"""
        return self.alert_manager.active_alerts
    
    def get_profile_stats(self, name: str = None) -> Dict[str, Any]:
        """Get profiling statistics"""
        if name:
            return self.profiler.get_profile_stats(name)
        else:
            return {
                name: self.profiler.get_profile_stats(name)
                for name in self.profiler.profiles.keys()
            }


# Global monitoring system
monitoring_system = MonitoringSystem()





























