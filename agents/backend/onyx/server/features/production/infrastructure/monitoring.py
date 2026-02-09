from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
import psutil
import GPUtil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from prometheus_client import (
from prometheus_client.asyncio import start_http_server
from typing import Any, List, Dict, Optional
"""
Monitoring Infrastructure
=========================

Prometheus monitoring service with comprehensive metrics and health checks.
"""


# Prometheus metrics
    Counter, Gauge, Histogram, Summary, generate_latest,
    CONTENT_TYPE_LATEST, CollectorRegistry
)

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System metrics data structure."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self) -> Any:
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class HealthStatus:
    """Health status data structure."""
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, Dict[str, Any]]
    timestamp: datetime = None
    
    def __post_init__(self) -> Any:
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PrometheusMonitoringService:
    """Prometheus-based monitoring service with comprehensive metrics."""
    
    def __init__(
        self,
        port: int = 9090,
        enable_http_server: bool = True,
        collect_system_metrics: bool = True,
        metrics_interval: int = 30
    ):
        
    """__init__ function."""
self.port = port
        self.enable_http_server = enable_http_server
        self.collect_system_metrics = collect_system_metrics
        self.metrics_interval = metrics_interval
        
        # Prometheus registry
        self.registry = CollectorRegistry()
        
        # Metrics definitions
        self._define_metrics()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
        
        # Health checks
        self.health_checks = {
            "system": self._check_system_health,
            "memory": self._check_memory_health,
            "disk": self._check_disk_health,
            "network": self._check_network_health,
            "gpu": self._check_gpu_health
        }
        
        # Performance tracking
        self.performance_data = {
            "response_times": [],
            "error_counts": {},
            "request_counts": {},
            "system_metrics": []
        }
        
        logger.info("PrometheusMonitoringService initialized")
    
    def _define_metrics(self) -> Any:
        """Define Prometheus metrics."""
        # Request metrics
        self.request_counter = Counter(
            'copywriting_requests_total',
            'Total number of copywriting requests',
            ['status', 'style', 'tone'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'copywriting_request_duration_seconds',
            'Request duration in seconds',
            ['status', 'style'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            ['mount_point'],
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'system_gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_memory = Gauge(
            'system_gpu_memory_usage_percent',
            'GPU memory usage percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        # AI model metrics
        self.ai_model_requests = Counter(
            'ai_model_requests_total',
            'Total AI model requests',
            ['model_name', 'status'],
            registry=self.registry
        )
        
        self.ai_model_duration = Histogram(
            'ai_model_duration_seconds',
            'AI model inference duration',
            ['model_name'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_counter = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Business metrics
        self.active_users = Gauge(
            'active_users_total',
            'Number of active users',
            registry=self.registry
        )
        
        self.content_generated = Counter(
            'content_generated_total',
            'Total content generated',
            ['content_type', 'quality_score'],
            registry=self.registry
        )
    
    async def initialize(self) -> Any:
        """Initialize monitoring service."""
        try:
            # Start HTTP server for metrics
            if self.enable_http_server:
                await start_http_server(self.port, registry=self.registry)
                logger.info(f"Prometheus metrics server started on port {self.port}")
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._running = True
            logger.info("Monitoring service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring service: {e}")
            raise
    
    async def _start_background_tasks(self) -> Any:
        """Start background monitoring tasks."""
        # System metrics collector
        if self.collect_system_metrics:
            task = asyncio.create_task(self._collect_system_metrics())
            self._background_tasks.append(task)
        
        # Performance data processor
        task = asyncio.create_task(self._process_performance_data())
        self._background_tasks.append(task)
        
        # Health check monitor
        task = asyncio.create_task(self._monitor_health())
        self._background_tasks.append(task)
        
        logger.info("Background monitoring tasks started")
    
    async def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a custom metric."""
        try:
            # Map metric names to Prometheus metrics
            if name == "request_count":
                status = tags.get("status", "unknown")
                style = tags.get("style", "unknown")
                tone = tags.get("tone", "unknown")
                self.request_counter.labels(status=status, style=style, tone=tone).inc()
            
            elif name == "request_duration":
                status = tags.get("status", "unknown")
                style = tags.get("style", "unknown")
                self.request_duration.labels(status=status, style=style).observe(value)
            
            elif name == "cache_hit":
                cache_type = tags.get("cache_type", "default")
                self.cache_hits.labels(cache_type=cache_type).inc()
            
            elif name == "cache_miss":
                cache_type = tags.get("cache_type", "default")
                self.cache_misses.labels(cache_type=cache_type).inc()
            
            elif name == "ai_model_request":
                model_name = tags.get("model_name", "unknown")
                status = tags.get("status", "unknown")
                self.ai_model_requests.labels(model_name=model_name, status=status).inc()
            
            elif name == "ai_model_duration":
                model_name = tags.get("model_name", "unknown")
                self.ai_model_duration.labels(model_name=model_name).observe(value)
            
            elif name == "error":
                error_type = tags.get("error_type", "unknown")
                component = tags.get("component", "unknown")
                self.error_counter.labels(error_type=error_type, component=component).inc()
            
            elif name == "content_generated":
                content_type = tags.get("content_type", "unknown")
                quality_score = tags.get("quality_score", "unknown")
                self.content_generated.labels(content_type=content_type, quality_score=quality_score).inc()
            
            # Store in performance data
            self.performance_data["request_counts"][name] = self.performance_data["request_counts"].get(name, 0) + 1
            
        except Exception as e:
            logger.error(f"Error recording metric {name}: {e}")
    
    async def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        try:
            if name == "request_count":
                status = tags.get("status", "unknown")
                style = tags.get("style", "unknown")
                tone = tags.get("tone", "unknown")
                self.request_counter.labels(status=status, style=style, tone=tone).inc(value)
            
            elif name == "error_count":
                error_type = tags.get("error_type", "unknown")
                component = tags.get("component", "unknown")
                self.error_counter.labels(error_type=error_type, component=component).inc(value)
            
            elif name == "cache_hit":
                cache_type = tags.get("cache_type", "default")
                self.cache_hits.labels(cache_type=cache_type).inc(value)
            
            elif name == "cache_miss":
                cache_type = tags.get("cache_type", "default")
                self.cache_misses.labels(cache_type=cache_type).inc(value)
            
        except Exception as e:
            logger.error(f"Error incrementing counter {name}: {e}")
    
    async def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record timing information."""
        try:
            if name == "request_duration":
                status = tags.get("status", "unknown")
                style = tags.get("style", "unknown")
                self.request_duration.labels(status=status, style=style).observe(duration)
            
            elif name == "ai_model_duration":
                model_name = tags.get("model_name", "unknown")
                self.ai_model_duration.labels(model_name=model_name).observe(duration)
            
            # Store in performance data
            self.performance_data["response_times"].append({
                "name": name,
                "duration": duration,
                "timestamp": datetime.now(),
                "tags": tags or {}
            })
            
            # Keep only recent data
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.performance_data["response_times"] = [
                rt for rt in self.performance_data["response_times"]
                if rt["timestamp"] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error recording timing {name}: {e}")
    
    async def record_event(self, name: str, data: Dict[str, Any]):
        """Record an event."""
        try:
            if name == "copywriting_request":
                status = data.get("status", "unknown")
                style = data.get("style", "unknown")
                tone = data.get("tone", "unknown")
                self.request_counter.labels(status=status, style=style, tone=tone).inc()
                
                if "duration" in data:
                    self.request_duration.labels(status=status, style=style).observe(data["duration"])
            
            elif name == "content_generated":
                content_type = data.get("content_type", "unknown")
                quality_score = data.get("quality_score", "unknown")
                self.content_generated.labels(content_type=content_type, quality_score=quality_score).inc()
            
            elif name == "error":
                error_type = data.get("error_type", "unknown")
                component = data.get("component", "unknown")
                self.error_counter.labels(error_type=error_type, component=component).inc()
            
        except Exception as e:
            logger.error(f"Error recording event {name}: {e}")
    
    async def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get current metrics."""
        try:
            metrics = {}
            
            # Get Prometheus metrics
            prometheus_metrics = generate_latest(self.registry).decode('utf-8')
            metrics["prometheus"] = prometheus_metrics
            
            # Get custom metrics
            metrics["custom"] = {
                "performance_data": self.performance_data,
                "system_metrics": await self._get_current_system_metrics()
            }
            
            # Filter by metric names if specified
            if metric_names:
                filtered_metrics = {}
                for name in metric_names:
                    if name in metrics["custom"]:
                        filtered_metrics[name] = metrics["custom"][name]
                metrics["custom"] = filtered_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        try:
            health_checks = {}
            overall_status = "healthy"
            
            # Run all health checks
            for check_name, check_func in self.health_checks.items():
                try:
                    check_result = await check_func()
                    health_checks[check_name] = check_result
                    
                    if check_result.get("status") == "unhealthy":
                        overall_status = "unhealthy"
                    elif check_result.get("status") == "degraded" and overall_status == "healthy":
                        overall_status = "degraded"
                        
                except Exception as e:
                    health_checks[check_name] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    overall_status = "unhealthy"
            
            return {
                "status": overall_status,
                "checks": health_checks,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _collect_system_metrics(self) -> Any:
        """Collect system metrics periodically."""
        while self._running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.percent)
                
                # Disk usage
                disk_partitions = psutil.disk_partitions()
                for partition in disk_partitions:
                    try:
                        disk_usage = psutil.disk_usage(partition.mountpoint)
                        usage_percent = (disk_usage.used / disk_usage.total) * 100
                        self.disk_usage.labels(mount_point=partition.mountpoint).set(usage_percent)
                    except Exception as e:
                        logger.warning(f"Error getting disk usage for {partition.mountpoint}: {e}")
                
                # GPU usage (if available)
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        self.gpu_usage.labels(gpu_id=str(i)).set(gpu.load * 100)
                        self.gpu_memory.labels(gpu_id=str(i)).set(gpu.memoryUtil * 100)
                except Exception as e:
                    logger.debug(f"GPU metrics not available: {e}")
                
                # Store system metrics
                system_metrics = SystemMetrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    disk_usage=usage_percent if 'usage_percent' in locals() else 0.0,
                    network_io=self._get_network_io(),
                    gpu_usage=gpu.load * 100 if 'gpu' in locals() else None,
                    gpu_memory=gpu.memoryUtil * 100 if 'gpu' in locals() else None
                )
                
                self.performance_data["system_metrics"].append(asdict(system_metrics))
                
                # Keep only recent metrics
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.performance_data["system_metrics"] = [
                    sm for sm in self.performance_data["system_metrics"]
                    if datetime.fromisoformat(sm["timestamp"]) > cutoff_time
                ]
                
                await asyncio.sleep(self.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.metrics_interval)
    
    async def _process_performance_data(self) -> Any:
        """Process and analyze performance data."""
        while self._running:
            try:
                # Calculate averages and trends
                if self.performance_data["response_times"]:
                    avg_response_time = sum(rt["duration"] for rt in self.performance_data["response_times"]) / len(self.performance_data["response_times"])
                    logger.debug(f"Average response time: {avg_response_time:.3f}s")
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Error processing performance data: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_health(self) -> Any:
        """Monitor system health periodically."""
        while self._running:
            try:
                health_status = await self.get_health_status()
                
                if health_status["status"] == "unhealthy":
                    logger.warning(f"System health degraded: {health_status}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring health: {e}")
                await asyncio.sleep(30)
    
    async def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            metrics = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available,
                "memory_total": memory.total,
                "disk_usage": {},
                "network_io": self._get_network_io(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Disk usage
            disk_partitions = psutil.disk_partitions()
            for partition in disk_partitions:
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    metrics["disk_usage"][partition.mountpoint] = {
                        "total": disk_usage.total,
                        "used": disk_usage.used,
                        "free": disk_usage.free,
                        "percent": (disk_usage.used / disk_usage.total) * 100
                    }
                except Exception:
                    continue
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting current system metrics: {e}")
            return {"error": str(e)}
    
    def _get_network_io(self) -> Dict[str, float]:
        """Get network I/O statistics."""
        try:
            network_io = psutil.net_io_counters()
            return {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            }
        except Exception:
            return {}
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if cpu_percent > 90:
                status = "degraded"
            if cpu_percent > 95:
                status = "unhealthy"
            
            return {
                "status": status,
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health."""
        try:
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 85:
                status = "degraded"
            if memory.percent > 95:
                status = "unhealthy"
            
            return {
                "status": status,
                "memory_usage": memory.percent,
                "available_memory": memory.available,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk health."""
        try:
            disk_partitions = psutil.disk_partitions()
            disk_status = {}
            
            for partition in disk_partitions:
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    usage_percent = (disk_usage.used / disk_usage.total) * 100
                    
                    status = "healthy"
                    if usage_percent > 85:
                        status = "degraded"
                    if usage_percent > 95:
                        status = "unhealthy"
                    
                    disk_status[partition.mountpoint] = {
                        "status": status,
                        "usage_percent": usage_percent
                    }
                except Exception:
                    continue
            
            overall_status = "healthy"
            if any(disk["status"] == "unhealthy" for disk in disk_status.values()):
                overall_status = "unhealthy"
            elif any(disk["status"] == "degraded" for disk in disk_status.values()):
                overall_status = "degraded"
            
            return {
                "status": overall_status,
                "disks": disk_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network health."""
        try:
            # Simple network connectivity check
            network_io = psutil.net_io_counters()
            
            return {
                "status": "healthy",
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU health."""
        try:
            gpus = GPUtil.getGPUs()
            gpu_status = {}
            
            for i, gpu in enumerate(gpus):
                status = "healthy"
                if gpu.load > 0.95:
                    status = "degraded"
                if gpu.memoryUtil > 0.95:
                    status = "degraded"
                
                gpu_status[f"gpu_{i}"] = {
                    "status": status,
                    "load": gpu.load * 100,
                    "memory_util": gpu.memoryUtil * 100,
                    "temperature": gpu.temperature
                }
            
            overall_status = "healthy"
            if any(gpu["status"] == "degraded" for gpu in gpu_status.values()):
                overall_status = "degraded"
            
            return {
                "status": overall_status,
                "gpus": gpu_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "healthy",  # GPU not available is not unhealthy
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup(self) -> Any:
        """Cleanup monitoring service."""
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            logger.info("Monitoring service cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up monitoring service: {e}") 