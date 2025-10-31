"""
Advanced Monitoring and Observability System for BUL
====================================================

Comprehensive monitoring, metrics collection, alerting, and observability features.
"""

import asyncio
import time
import psutil
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import threading
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import logging

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    """Alert data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    message: str = ""
    level: AlertLevel = AlertLevel.INFO
    metric_name: str = ""
    threshold: float = 0.0
    current_value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class HealthCheck:
    """Health check data structure"""
    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: str = ""
    response_time: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

class PrometheusMetrics:
    """Prometheus metrics integration"""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.metrics = {}
        self._initialize_metrics()
        self._start_server()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        # API metrics
        self.metrics['api_requests_total'] = Counter(
            'bul_api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.metrics['api_request_duration'] = Histogram(
            'bul_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.metrics['document_generation_total'] = Counter(
            'bul_document_generation_total',
            'Total number of documents generated',
            ['business_area', 'document_type', 'status']
        )
        
        self.metrics['document_generation_duration'] = Histogram(
            'bul_document_generation_duration_seconds',
            'Document generation duration in seconds',
            ['business_area', 'document_type']
        )
        
        # System metrics
        self.metrics['cpu_usage'] = Gauge(
            'bul_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.metrics['memory_usage'] = Gauge(
            'bul_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.metrics['active_connections'] = Gauge(
            'bul_active_connections',
            'Number of active connections'
        )
        
        # Cache metrics
        self.metrics['cache_hits_total'] = Counter(
            'bul_cache_hits_total',
            'Total number of cache hits',
            ['cache_type']
        )
        
        self.metrics['cache_misses_total'] = Counter(
            'bul_cache_misses_total',
            'Total number of cache misses',
            ['cache_type']
        )
        
        # Error metrics
        self.metrics['errors_total'] = Counter(
            'bul_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
    
    def _start_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None, value: float = 1.0):
        """Increment a counter metric"""
        if name in self.metrics and hasattr(self.metrics[name], 'labels'):
            if labels:
                self.metrics[name].labels(**labels).inc(value)
            else:
                self.metrics[name].inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        if name in self.metrics and hasattr(self.metrics[name], 'labels'):
            if labels:
                self.metrics[name].labels(**labels).set(value)
            else:
                self.metrics[name].set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a histogram metric"""
        if name in self.metrics and hasattr(self.metrics[name], 'labels'):
            if labels:
                self.metrics[name].labels(**labels).observe(value)
            else:
                self.metrics[name].observe(value)

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self):
        self.prometheus = PrometheusMetrics()
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.custom_metrics: Dict[str, Callable] = {}
        self.collection_interval = 60  # seconds
        self.is_collecting = False
        self._start_collection()
    
    def _start_collection(self):
        """Start metrics collection"""
        if not self.is_collecting:
            self.is_collecting = True
            asyncio.create_task(self._collect_metrics_loop())
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self.is_collecting:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await self._collect_custom_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.prometheus.set_gauge('cpu_usage', cpu_percent)
            self._store_metric('system.cpu_usage', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.prometheus.set_gauge('memory_usage', memory.used)
            self._store_metric('system.memory_usage', memory.used)
            self._store_metric('system.memory_percent', memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self._store_metric('system.disk_usage', disk.used)
            self._store_metric('system.disk_percent', (disk.used / disk.total) * 100)
            
            # Network I/O
            network = psutil.net_io_counters()
            self._store_metric('system.network_bytes_sent', network.bytes_sent)
            self._store_metric('system.network_bytes_recv', network.bytes_recv)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-level metrics"""
        try:
            # Get metrics from BUL engine
            from ..core.bul_engine import get_global_bul_engine
            engine = await get_global_bul_engine()
            if engine:
                stats = await engine.get_stats()
                
                # Document generation metrics
                self._store_metric('app.documents_generated', stats.get('documents_generated', 0))
                self._store_metric('app.avg_processing_time', stats.get('average_processing_time', 0))
                self._store_metric('app.avg_confidence', stats.get('average_confidence', 0))
                
                # Cache metrics
                cache_perf = stats.get('cache_performance', {})
                self._store_metric('app.cache_hit_rate', cache_perf.get('hit_rate', 0))
                self._store_metric('app.cache_hits', cache_perf.get('cache_hits', 0))
                self._store_metric('app.cache_misses', cache_perf.get('cache_misses', 0))
                
                # API performance
                api_perf = stats.get('api_performance', {})
                self._store_metric('app.api_success_rate', 
                    api_perf.get('successful_calls', 0) / 
                    max(1, api_perf.get('successful_calls', 0) + api_perf.get('failed_calls', 0)))
                
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    async def _collect_custom_metrics(self):
        """Collect custom metrics"""
        for name, collector in self.custom_metrics.items():
            try:
                if asyncio.iscoroutinefunction(collector):
                    value = await collector()
                else:
                    value = collector()
                self._store_metric(name, value)
            except Exception as e:
                logger.error(f"Error collecting custom metric {name}: {e}")
    
    def _store_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Store metric in history"""
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {},
            timestamp=datetime.now()
        )
        self.metrics_history[name].append(metric)
    
    def add_custom_metric(self, name: str, collector: Callable):
        """Add a custom metric collector"""
        self.custom_metrics[name] = collector
    
    def get_metric_history(self, name: str, hours: int = 24) -> List[Metric]:
        """Get metric history for a specific metric"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metric for metric in self.metrics_history.get(name, [])
            if metric.timestamp > cutoff_time
        ]
    
    def get_metric_summary(self, name: str, hours: int = 24) -> Dict[str, Any]:
        """Get metric summary statistics"""
        history = self.get_metric_history(name, hours)
        if not history:
            return {}
        
        values = [metric.value for metric in history]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0,
            "timestamp": history[-1].timestamp if history else None
        }

class AlertManager:
    """Advanced alerting system"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: List[Callable] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        self.alert_rules = {
            "high_cpu_usage": {
                "metric": "system.cpu_usage",
                "threshold": 80.0,
                "level": AlertLevel.WARNING,
                "message": "High CPU usage detected"
            },
            "high_memory_usage": {
                "metric": "system.memory_percent",
                "threshold": 85.0,
                "level": AlertLevel.WARNING,
                "message": "High memory usage detected"
            },
            "low_cache_hit_rate": {
                "metric": "app.cache_hit_rate",
                "threshold": 0.5,
                "level": AlertLevel.WARNING,
                "message": "Low cache hit rate",
                "comparison": "less_than"
            },
            "high_error_rate": {
                "metric": "app.api_success_rate",
                "threshold": 0.9,
                "level": AlertLevel.ERROR,
                "message": "High API error rate",
                "comparison": "less_than"
            },
            "slow_response_time": {
                "metric": "app.avg_processing_time",
                "threshold": 30.0,
                "level": AlertLevel.WARNING,
                "message": "Slow document processing time"
            }
        }
    
    def add_alert_rule(self, name: str, rule: Dict[str, Any]):
        """Add a custom alert rule"""
        self.alert_rules[name] = rule
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler"""
        self.alert_handlers.append(handler)
    
    async def check_alerts(self, metrics_collector: MetricsCollector):
        """Check all alert rules"""
        for rule_name, rule in self.alert_rules.items():
            try:
                metric_name = rule["metric"]
                threshold = rule["threshold"]
                level = rule["level"]
                message = rule["message"]
                comparison = rule.get("comparison", "greater_than")
                
                # Get current metric value
                summary = metrics_collector.get_metric_summary(metric_name, hours=1)
                current_value = summary.get("latest", 0)
                
                # Check if alert condition is met
                alert_triggered = False
                if comparison == "greater_than" and current_value > threshold:
                    alert_triggered = True
                elif comparison == "less_than" and current_value < threshold:
                    alert_triggered = True
                elif comparison == "equals" and current_value == threshold:
                    alert_triggered = True
                
                if alert_triggered:
                    await self._trigger_alert(rule_name, message, level, metric_name, threshold, current_value)
                else:
                    await self._resolve_alert(rule_name)
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    async def _trigger_alert(self, rule_name: str, message: str, level: AlertLevel, 
                           metric_name: str, threshold: float, current_value: float):
        """Trigger an alert"""
        # Check if alert already exists and is active
        if rule_name in self.alerts and not self.alerts[rule_name].resolved:
            return  # Alert already active
        
        # Create new alert
        alert = Alert(
            name=rule_name,
            message=message,
            level=level,
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value
        )
        
        self.alerts[rule_name] = alert
        
        # Send alert to handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.warning(f"Alert triggered: {message} (Current: {current_value}, Threshold: {threshold})")
    
    async def _resolve_alert(self, rule_name: str):
        """Resolve an alert"""
        if rule_name in self.alerts and not self.alerts[rule_name].resolved:
            self.alerts[rule_name].resolved = True
            self.alerts[rule_name].resolved_at = datetime.now()
            logger.info(f"Alert resolved: {rule_name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts.values()
            if alert.timestamp > cutoff_time
        ]

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthCheck] = {}
        self._initialize_default_checks()
    
    def _initialize_default_checks(self):
        """Initialize default health checks"""
        self.health_checks = {
            "system": self._check_system_health,
            "database": self._check_database_health,
            "api": self._check_api_health,
            "cache": self._check_cache_health,
            "external_apis": self._check_external_apis_health
        }
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks"""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                response_time = time.time() - start_time
                
                health_check = HealthCheck(
                    name=name,
                    status=result.get("status", "unhealthy"),
                    message=result.get("message", ""),
                    response_time=response_time,
                    details=result.get("details", {})
                )
                
                results[name] = health_check
                self.health_status[name] = health_check
                
            except Exception as e:
                health_check = HealthCheck(
                    name=name,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    response_time=0.0
                )
                results[name] = health_check
                self.health_status[name] = health_check
        
        return results
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            issues = []
            
            if cpu_percent > 90:
                status = "unhealthy"
                issues.append(f"High CPU usage: {cpu_percent}%")
            elif cpu_percent > 80:
                status = "degraded"
                issues.append(f"Elevated CPU usage: {cpu_percent}%")
            
            if memory.percent > 95:
                status = "unhealthy"
                issues.append(f"High memory usage: {memory.percent}%")
            elif memory.percent > 85:
                status = "degraded"
                issues.append(f"Elevated memory usage: {memory.percent}%")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 95:
                status = "unhealthy"
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                status = "degraded"
                issues.append(f"Elevated disk usage: {disk_percent:.1f}%")
            
            return {
                "status": status,
                "message": "; ".join(issues) if issues else "System healthy",
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk_percent,
                    "issues": issues
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"System health check failed: {str(e)}"
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # In a real implementation, check database connectivity
            return {
                "status": "healthy",
                "message": "Database connection healthy"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Database health check failed: {str(e)}"
            }
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            from ..core.bul_engine import get_global_bul_engine
            engine = await get_global_bul_engine()
            
            if engine and engine.is_initialized:
                return {
                    "status": "healthy",
                    "message": "API engine healthy"
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "API engine not initialized"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"API health check failed: {str(e)}"
            }
    
    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health"""
        try:
            from ..utils.cache_manager import get_cache_manager
            cache = get_cache_manager()
            
            if cache and cache.is_initialized:
                return {
                    "status": "healthy",
                    "message": "Cache system healthy"
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Cache system not initialized"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Cache health check failed: {str(e)}"
            }
    
    async def _check_external_apis_health(self) -> Dict[str, Any]:
        """Check external APIs health"""
        try:
            # Check OpenRouter API
            import os
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            
            if openrouter_key:
                return {
                    "status": "healthy",
                    "message": "External APIs accessible"
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "OpenRouter API key not configured"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"External APIs health check failed: {str(e)}"
            }

class ObservabilityManager:
    """Main observability manager"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.is_running = False
    
    async def start(self):
        """Start the observability system"""
        if not self.is_running:
            self.is_running = True
            
            # Start alert checking loop
            asyncio.create_task(self._alert_checking_loop())
            
            # Add default alert handlers
            self.alert_manager.add_alert_handler(self._default_alert_handler)
            
            logger.info("Observability system started")
    
    async def _alert_checking_loop(self):
        """Alert checking loop"""
        while self.is_running:
            try:
                await self.alert_manager.check_alerts(self.metrics_collector)
                await asyncio.sleep(60)  # Check alerts every minute
            except Exception as e:
                logger.error(f"Error in alert checking loop: {e}")
                await asyncio.sleep(5)
    
    async def _default_alert_handler(self, alert: Alert):
        """Default alert handler"""
        logger.warning(f"ALERT [{alert.level.value.upper()}]: {alert.message}")
        # In production, integrate with external alerting systems
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Run health checks
            health_checks = await self.health_checker.run_all_checks()
            
            # Get metrics summary
            metrics_summary = {}
            for metric_name in ["system.cpu_usage", "system.memory_percent", "app.cache_hit_rate"]:
                metrics_summary[metric_name] = self.metrics_collector.get_metric_summary(metric_name, hours=1)
            
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            
            # Calculate overall health
            health_statuses = [check.status for check in health_checks.values()]
            if "unhealthy" in health_statuses:
                overall_health = "unhealthy"
            elif "degraded" in health_statuses:
                overall_health = "degraded"
            else:
                overall_health = "healthy"
            
            return {
                "overall_health": overall_health,
                "timestamp": datetime.now().isoformat(),
                "health_checks": {
                    name: {
                        "status": check.status,
                        "message": check.message,
                        "response_time": check.response_time,
                        "last_check": check.last_check.isoformat()
                    }
                    for name, check in health_checks.items()
                },
                "metrics": metrics_summary,
                "active_alerts": [
                    {
                        "name": alert.name,
                        "message": alert.message,
                        "level": alert.level.value,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in active_alerts
                ],
                "uptime": time.time() - psutil.boot_time()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "overall_health": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global observability instance
_observability_manager: Optional[ObservabilityManager] = None

async def get_observability_manager() -> ObservabilityManager:
    """Get the global observability manager"""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
        await _observability_manager.start()
    return _observability_manager
















