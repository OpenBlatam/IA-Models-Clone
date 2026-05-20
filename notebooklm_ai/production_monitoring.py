from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
import json
import os
import sys
import traceback
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta
import psutil
import GPUtil
from pathlib import Path
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
import redis.asyncio as redis
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import uvicorn
from production_config import get_config, ProductionConfig
            import sqlalchemy
            from sqlalchemy import create_engine
from typing import Any, List, Dict, Optional
"""
Production Monitoring System
============================

Comprehensive monitoring system for production deployment including:
- Health checks
- Metrics collection
- Performance monitoring
- Alerting
- Dashboard integration
- Log aggregation
"""


# Monitoring imports

# Local imports

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """Alert level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0

@dataclass
class Alert:
    """Alert definition"""
    id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System metrics collection"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_total: int
    disk_percent: float
    disk_used: int
    disk_total: int
    network_sent: int
    network_recv: int
    gpu_usage: Optional[float] = None
    gpu_memory_used: Optional[int] = None
    gpu_memory_total: Optional[int] = None

class MonitoringMetrics:
    """Monitoring metrics for Prometheus"""
    
    def __init__(self) -> Any:
        # System metrics
        self.cpu_usage = Gauge('system_cpu_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_bytes', 'Memory usage in bytes')
        self.memory_percent = Gauge('system_memory_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_bytes', 'Disk usage in bytes')
        self.disk_percent = Gauge('system_disk_percent', 'Disk usage percentage')
        self.network_sent = Gauge('system_network_sent_bytes', 'Network bytes sent')
        self.network_recv = Gauge('system_network_recv_bytes', 'Network bytes received')
        
        # GPU metrics
        self.gpu_usage = Gauge('system_gpu_usage_percent', 'GPU usage percentage')
        self.gpu_memory_used = Gauge('system_gpu_memory_bytes', 'GPU memory used in bytes')
        self.gpu_memory_percent = Gauge('system_gpu_memory_percent', 'GPU memory usage percentage')
        
        # Application metrics
        self.request_total = Counter('app_requests_total', 'Total requests', ['endpoint', 'method', 'status'])
        self.request_duration = Histogram('app_request_duration_seconds', 'Request duration', ['endpoint'])
        self.error_total = Counter('app_errors_total', 'Total errors', ['type', 'source'])
        self.active_connections = Gauge('app_active_connections', 'Active connections')
        
        # Worker metrics
        self.worker_tasks_processed = Counter('worker_tasks_processed_total', 'Tasks processed', ['worker_id', 'status'])
        self.worker_queue_size = Gauge('worker_queue_size', 'Queue size', ['queue_name'])
        self.worker_active = Gauge('worker_active', 'Active workers')
        
        # Health check metrics
        self.health_check_status = Gauge('health_check_status', 'Health check status', ['check_name'])
        self.health_check_duration = Histogram('health_check_duration_seconds', 'Health check duration', ['check_name'])
        
        # Alert metrics
        self.alert_total = Counter('alert_total', 'Total alerts', ['level', 'source'])
        self.alert_active = Gauge('alert_active', 'Active alerts', ['level'])

class HealthChecker:
    """Health check system"""
    
    def __init__(self, config: ProductionConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger()
        self.checks = {}
        self.last_check = {}
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> Any:
        """Register default health checks"""
        self.register_check("system", self._check_system_health)
        self.register_check("database", self._check_database_health)
        self.register_check("redis", self._check_redis_health)
        self.register_check("storage", self._check_storage_health)
        self.register_check("api", self._check_api_health)
        self.register_check("workers", self._check_workers_health)
        self.register_check("memory", self._check_memory_health)
        self.register_check("disk", self._check_disk_health)
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check"""
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks"""
        results = {}
        
        for name, check_func in self.checks.items():
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(check_func):
                    status, message, details = await check_func()
                else:
                    status, message, details = check_func()
                
                duration = time.time() - start_time
                
                results[name] = HealthCheck(
                    name=name,
                    status=status,
                    message=message,
                    details=details,
                    duration=duration
                )
                
                self.last_check[name] = results[name]
                
            except Exception as e:
                duration = time.time() - start_time
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e), "traceback": traceback.format_exc()},
                    duration=duration
                )
        
        return results
    
    def get_overall_health(self, results: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall health status"""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    # Health check implementations
    def _check_system_health(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check system health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used": memory.used,
                "memory_total": memory.total
            }
            
            if cpu_percent > 90 or memory.percent > 90:
                return HealthStatus.UNHEALTHY, "System resources critically high", details
            elif cpu_percent > 80 or memory.percent > 80:
                return HealthStatus.DEGRADED, "System resources high", details
            else:
                return HealthStatus.HEALTHY, "System healthy", details
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"System check failed: {str(e)}", {"error": str(e)}
    
    async def _check_database_health(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check database health"""
        try:
            # Test database connection
            
            engine = create_engine(self.config.get_database_url())
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text("SELECT 1"))
                result.fetchone()
            
            return HealthStatus.HEALTHY, "Database connection healthy", {"connection": "ok"}
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Database check failed: {str(e)}", {"error": str(e)}
    
    async def _check_redis_health(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check Redis health"""
        try:
            redis_client = redis.from_url(self.config.get_redis_url())
            await redis_client.ping()
            
            # Test basic operations
            await redis_client.set("health_check", "ok")
            value = await redis_client.get("health_check")
            await redis_client.delete("health_check")
            
            if value == b"ok":
                return HealthStatus.HEALTHY, "Redis connection healthy", {"connection": "ok"}
            else:
                return HealthStatus.DEGRADED, "Redis operations failed", {"connection": "partial"}
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Redis check failed: {str(e)}", {"error": str(e)}
    
    def _check_storage_health(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check storage health"""
        try:
            storage_path = Path(self.config.storage.local_path)
            
            if not storage_path.exists():
                storage_path.mkdir(parents=True, exist_ok=True)
            
            # Test write/read
            test_file = storage_path / "health_check.txt"
            test_file.write_text("ok")
            content = test_file.read_text()
            test_file.unlink()
            
            if content == "ok":
                return HealthStatus.HEALTHY, "Storage healthy", {"path": str(storage_path)}
            else:
                return HealthStatus.DEGRADED, "Storage operations failed", {"path": str(storage_path)}
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Storage check failed: {str(e)}", {"error": str(e)}
    
    async async def _check_api_health(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check API health"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.config.api.host}:{self.config.api.port}/health", timeout=5)
                
                if response.status_code == 200:
                    return HealthStatus.HEALTHY, "API healthy", {"status_code": response.status_code}
                else:
                    return HealthStatus.DEGRADED, f"API returned status {response.status_code}", {"status_code": response.status_code}
                    
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"API check failed: {str(e)}", {"error": str(e)}
    
    async def _check_workers_health(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check workers health"""
        try:
            # This would check the worker pool status
            # For now, return healthy
            return HealthStatus.HEALTHY, "Workers healthy", {"workers": "ok"}
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Workers check failed: {str(e)}", {"error": str(e)}
    
    def _check_memory_health(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check memory health"""
        try:
            memory = psutil.virtual_memory()
            
            details = {
                "percent": memory.percent,
                "used": memory.used,
                "total": memory.total,
                "available": memory.available
            }
            
            if memory.percent > 95:
                return HealthStatus.UNHEALTHY, "Memory critically low", details
            elif memory.percent > 85:
                return HealthStatus.DEGRADED, "Memory usage high", details
            else:
                return HealthStatus.HEALTHY, "Memory healthy", details
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Memory check failed: {str(e)}", {"error": str(e)}
    
    def _check_disk_health(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check disk health"""
        try:
            disk = psutil.disk_usage('/')
            
            details = {
                "percent": disk.percent,
                "used": disk.used,
                "total": disk.total,
                "free": disk.free
            }
            
            if disk.percent > 95:
                return HealthStatus.UNHEALTHY, "Disk space critically low", details
            elif disk.percent > 85:
                return HealthStatus.DEGRADED, "Disk usage high", details
            else:
                return HealthStatus.HEALTHY, "Disk healthy", details
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Disk check failed: {str(e)}", {"error": str(e)}

class AlertManager:
    """Alert management system"""
    
    def __init__(self, config: ProductionConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger()
        self.alerts = {}
        self.alert_rules = {}
        self.alert_handlers = {}
        
        # Register default alert handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> Any:
        """Register default alert handlers"""
        self.register_handler("log", self._log_alert)
        self.register_handler("email", self._email_alert)
        self.register_handler("slack", self._slack_alert)
        self.register_handler("webhook", self._webhook_alert)
    
    def register_handler(self, name: str, handler: Callable):
        """Register an alert handler"""
        self.alert_handlers[name] = handler
        self.logger.info(f"Registered alert handler: {name}")
    
    def create_alert(self, level: AlertLevel, title: str, message: str, source: str, metadata: Dict[str, Any] = None) -> Alert:
        """Create a new alert"""
        alert_id = f"{source}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            level=level,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        self.logger.warning(f"Alert created: {title} - {message}")
        
        # Trigger alert handlers
        self._trigger_alert_handlers(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by level"""
        return [alert for alert in self.alerts.values() if alert.level == level]
    
    def _trigger_alert_handlers(self, alert: Alert):
        """Trigger alert handlers"""
        for name, handler in self.alert_handlers.items():
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(alert))
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler {name} failed: {e}")
    
    # Alert handlers
    def _log_alert(self, alert: Alert):
        """Log alert handler"""
        log_level = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.ERROR: self.logger.error,
            AlertLevel.CRITICAL: self.logger.critical
        }
        
        log_func = log_level.get(alert.level, self.logger.warning)
        log_func(f"ALERT [{alert.level.value.upper()}] {alert.title}: {alert.message}")
    
    def _email_alert(self, alert: Alert):
        """Email alert handler"""
        # Implement email sending logic
        self.logger.info(f"Email alert would be sent: {alert.title}")
    
    def _slack_alert(self, alert: Alert):
        """Slack alert handler"""
        # Implement Slack notification logic
        self.logger.info(f"Slack alert would be sent: {alert.title}")
    
    def _webhook_alert(self, alert: Alert):
        """Webhook alert handler"""
        # Implement webhook notification logic
        self.logger.info(f"Webhook alert would be sent: {alert.title}")

class MetricsCollector:
    """Metrics collection system"""
    
    def __init__(self, config: ProductionConfig, metrics: MonitoringMetrics):
        
    """__init__ function."""
self.config = config
        self.metrics = metrics
        self.logger = structlog.get_logger()
        self.collection_interval = 30  # seconds
        self.is_running = False
    
    async def start(self) -> Any:
        """Start metrics collection"""
        self.is_running = True
        self.logger.info("Starting metrics collection")
        
        while self.is_running:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def stop(self) -> Any:
        """Stop metrics collection"""
        self.is_running = False
        self.logger.info("Stopped metrics collection")
    
    async def collect_metrics(self) -> Any:
        """Collect system and application metrics"""
        try:
            # System metrics
            await self._collect_system_metrics()
            
            # GPU metrics
            await self._collect_gpu_metrics()
            
            # Application metrics (if available)
            await self._collect_application_metrics()
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
    
    async def _collect_system_metrics(self) -> Any:
        """Collect system metrics"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.cpu_usage.set(cpu_percent)
            
            # Memory
            memory = psutil.virtual_memory()
            self.metrics.memory_usage.set(memory.used)
            self.metrics.memory_percent.set(memory.percent)
            
            # Disk
            disk = psutil.disk_usage('/')
            self.metrics.disk_usage.set(disk.used)
            self.metrics.disk_percent.set(disk.percent)
            
            # Network
            network = psutil.net_io_counters()
            self.metrics.network_sent.set(network.bytes_sent)
            self.metrics.network_recv.set(network.bytes_recv)
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    async def _collect_gpu_metrics(self) -> Any:
        """Collect GPU metrics"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    self.metrics.gpu_usage.set(gpu.load * 100)
                    self.metrics.gpu_memory_used.set(gpu.memoryUsed * 1024 * 1024)  # Convert to bytes
                    self.metrics.gpu_memory_percent.set((gpu.memoryUsed / gpu.memoryTotal) * 100)
                    
        except Exception as e:
            # GPU metrics are optional
            pass
    
    async def _collect_application_metrics(self) -> Any:
        """Collect application-specific metrics"""
        try:
            # This would collect metrics from the application
            # For now, just log that we're collecting
            pass
            
        except Exception as e:
            self.logger.error(f"Failed to collect application metrics: {e}")

class MonitoringDashboard:
    """Monitoring dashboard API"""
    
    def __init__(self, config: ProductionConfig, health_checker: HealthChecker, alert_manager: AlertManager, metrics: MonitoringMetrics):
        
    """__init__ function."""
self.config = config
        self.health_checker = health_checker
        self.alert_manager = alert_manager
        self.metrics = metrics
        self.logger = structlog.get_logger()
        self.app = FastAPI(title="Monitoring Dashboard", version="1.0.0")
        
        self._setup_routes()
    
    def _setup_routes(self) -> Any:
        """Setup dashboard routes"""
        
        @self.app.get("/")
        async def dashboard_root():
            """Dashboard root"""
            return {
                "title": "Production Monitoring Dashboard",
                "version": "1.0.0",
                "endpoints": {
                    "health": "/health",
                    "metrics": "/metrics",
                    "alerts": "/alerts",
                    "system": "/system"
                }
            }
        
        @self.app.get("/health")
        async def dashboard_health():
            """Comprehensive health status"""
            health_results = await self.health_checker.run_health_checks()
            overall_health = self.health_checker.get_overall_health(health_results)
            
            return {
                "overall": overall_health.value,
                "timestamp": time.time(),
                "checks": {
                    name: {
                        "status": check.status.value,
                        "message": check.message,
                        "duration": check.duration,
                        "details": check.details
                    }
                    for name, check in health_results.items()
                }
            }
        
        @self.app.get("/metrics")
        async def dashboard_metrics():
            """Prometheus metrics"""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
        
        @self.app.get("/alerts")
        async def dashboard_alerts():
            """Active alerts"""
            active_alerts = self.alert_manager.get_active_alerts()
            
            return {
                "total": len(active_alerts),
                "alerts": [
                    {
                        "id": alert.id,
                        "level": alert.level.value,
                        "title": alert.title,
                        "message": alert.message,
                        "source": alert.source,
                        "timestamp": alert.timestamp,
                        "acknowledged": alert.acknowledged,
                        "metadata": alert.metadata
                    }
                    for alert in active_alerts
                ]
            }
        
        @self.app.get("/system")
        async def dashboard_system():
            """System information"""
            try:
                # System info
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network info
                network = psutil.net_io_counters()
                
                # Process info
                process = psutil.Process()
                
                return {
                    "system": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used": memory.used,
                        "memory_total": memory.total,
                        "disk_percent": disk.percent,
                        "disk_used": disk.used,
                        "disk_total": disk.total,
                        "network_sent": network.bytes_sent,
                        "network_recv": network.bytes_recv
                    },
                    "process": {
                        "pid": process.pid,
                        "memory_info": {
                            "rss": process.memory_info().rss,
                            "vms": process.memory_info().vms
                        },
                        "cpu_percent": process.cpu_percent(),
                        "num_threads": process.num_threads(),
                        "create_time": process.create_time()
                    },
                    "timestamp": time.time()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")
        
        @self.app.post("/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge an alert"""
            success = self.alert_manager.acknowledge_alert(alert_id)
            if success:
                return {"success": True, "message": "Alert acknowledged"}
            else:
                raise HTTPException(status_code=404, detail="Alert not found")
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve an alert"""
            success = self.alert_manager.resolve_alert(alert_id)
            if success:
                return {"success": True, "message": "Alert resolved"}
            else:
                raise HTTPException(status_code=404, detail="Alert not found")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI app"""
        return self.app

class ProductionMonitoring:
    """Main production monitoring system"""
    
    def __init__(self, config: ProductionConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger()
        self.metrics = MonitoringMetrics()
        self.health_checker = HealthChecker(config)
        self.alert_manager = AlertManager(config)
        self.metrics_collector = MetricsCollector(config, self.metrics)
        self.dashboard = MonitoringDashboard(config, self.health_checker, self.alert_manager, self.metrics)
        
        self.is_running = False
    
    async def start(self) -> Any:
        """Start the monitoring system"""
        self.logger.info("🚀 Starting Production Monitoring System")
        self.is_running = True
        
        try:
            # Start metrics collection
            metrics_task = asyncio.create_task(self.metrics_collector.start())
            
            # Start dashboard
            dashboard_config = uvicorn.Config(
                app=self.dashboard.get_app(),
                host="0.0.0.0",
                port=8002,
                log_level="info"
            )
            dashboard_server = uvicorn.Server(dashboard_config)
            
            # Run both tasks
            await asyncio.gather(
                metrics_task,
                dashboard_server.serve(),
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"Monitoring system error: {e}")
            raise
    
    async def stop(self) -> Any:
        """Stop the monitoring system"""
        self.logger.info("🛑 Stopping Production Monitoring System")
        self.is_running = False
        
        try:
            await self.metrics_collector.stop()
            self.logger.info("✅ Monitoring system stopped")
        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {e}")

async def main():
    """Main function for monitoring deployment"""
    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger()
    logger.info("🚀 Starting Production Monitoring System")
    
    try:
        # Get configuration
        config = get_config()
        
        # Validate configuration
        if not config.validate():
            logger.error("Invalid configuration")
            sys.exit(1)
        
        # Create directories
        config.create_directories()
        
        # Create monitoring system
        monitoring = ProductionMonitoring(config)
        
        # Start monitoring
        await monitoring.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Monitoring system error: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup
        if 'monitoring' in locals():
            await monitoring.stop()
        logger.info("✅ Monitoring system shutdown completed")

match __name__:
    case "__main__":
    asyncio.run(main()) 