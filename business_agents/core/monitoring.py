"""
Monitoring and Observability
============================

Comprehensive monitoring, metrics collection, and observability system.
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import threading
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from prometheus_client.core import CollectorRegistry

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime = field(default_factory=datetime.now)

class MetricsCollector:
    """Collects and stores system metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.system_metrics_history: deque = deque(maxlen=100)  # Keep last 100 system snapshots
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a custom metric."""
        with self._lock:
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics_history[name].append(metric_point)
    
    def record_system_metrics(self):
        """Record current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv
            )
            
            with self._lock:
                self.system_metrics_history.append(system_metrics)
            
            # Record as individual metrics
            self.record_metric("system.cpu_percent", cpu_percent)
            self.record_metric("system.memory_percent", memory_percent)
            self.record_metric("system.memory_used_mb", memory_used_mb)
            self.record_metric("system.disk_usage_percent", disk_usage_percent)
            self.record_metric("system.disk_free_gb", disk_free_gb)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
    
    def get_metric_summary(self, name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric over a time period."""
        with self._lock:
            if name not in self.metrics_history:
                return {}
            
            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            recent_metrics = [
                m for m in self.metrics_history[name]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {}
            
            values = [m.value for m in recent_metrics]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else None,
                "duration_minutes": duration_minutes
            }
    
    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get current system metrics summary."""
        with self._lock:
            if not self.system_metrics_history:
                return {}
            
            latest = self.system_metrics_history[-1]
            
            return {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_used_mb": latest.memory_used_mb,
                "memory_available_mb": latest.memory_available_mb,
                "disk_usage_percent": latest.disk_usage_percent,
                "disk_free_gb": latest.disk_free_gb,
                "timestamp": latest.timestamp.isoformat()
            }

class PrometheusMetrics:
    """Prometheus metrics integration."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'business_agents_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'business_agents_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Agent metrics
        self.agent_executions = Counter(
            'business_agents_agent_executions_total',
            'Total agent executions',
            ['agent_id', 'capability', 'status'],
            registry=self.registry
        )
        
        self.agent_execution_duration = Histogram(
            'business_agents_agent_execution_duration_seconds',
            'Agent execution duration in seconds',
            ['agent_id', 'capability'],
            registry=self.registry
        )
        
        # Workflow metrics
        self.workflow_executions = Counter(
            'business_agents_workflow_executions_total',
            'Total workflow executions',
            ['workflow_id', 'status'],
            registry=self.registry
        )
        
        self.workflow_duration = Histogram(
            'business_agents_workflow_duration_seconds',
            'Workflow execution duration in seconds',
            ['workflow_id'],
            registry=self.registry
        )
        
        # Document metrics
        self.document_generations = Counter(
            'business_agents_document_generations_total',
            'Total document generations',
            ['document_type', 'format', 'status'],
            registry=self.registry
        )
        
        self.document_generation_duration = Histogram(
            'business_agents_document_generation_duration_seconds',
            'Document generation duration in seconds',
            ['document_type', 'format'],
            registry=self.registry
        )
        
        # System metrics
        self.active_agents = Gauge(
            'business_agents_active_agents',
            'Number of active agents',
            registry=self.registry
        )
        
        self.active_workflows = Gauge(
            'business_agents_active_workflows',
            'Number of active workflows',
            registry=self.registry
        )
        
        self.system_cpu_percent = Gauge(
            'business_agents_system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_percent = Gauge(
            'business_agents_system_memory_percent',
            'System memory usage percentage',
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_agent_execution(self, agent_id: str, capability: str, status: str, duration: float):
        """Record agent execution metrics."""
        self.agent_executions.labels(
            agent_id=agent_id,
            capability=capability,
            status=status
        ).inc()
        
        self.agent_execution_duration.labels(
            agent_id=agent_id,
            capability=capability
        ).observe(duration)
    
    def record_workflow_execution(self, workflow_id: str, status: str, duration: float):
        """Record workflow execution metrics."""
        self.workflow_executions.labels(
            workflow_id=workflow_id,
            status=status
        ).inc()
        
        self.workflow_duration.labels(
            workflow_id=workflow_id
        ).observe(duration)
    
    def record_document_generation(self, document_type: str, format: str, status: str, duration: float):
        """Record document generation metrics."""
        self.document_generations.labels(
            document_type=document_type,
            format=format,
            status=status
        ).inc()
        
        self.document_generation_duration.labels(
            document_type=document_type,
            format=format
        ).observe(duration)
    
    def update_system_metrics(self, cpu_percent: float, memory_percent: float):
        """Update system metrics."""
        self.system_cpu_percent.set(cpu_percent)
        self.system_memory_percent.set(memory_percent)
    
    def update_active_counts(self, active_agents: int, active_workflows: int):
        """Update active counts."""
        self.active_agents.set(active_agents)
        self.active_workflows.set(active_workflows)

class MonitoringService:
    """Central monitoring service."""
    
    def __init__(self, metrics_port: int = 9090):
        self.metrics_collector = MetricsCollector()
        self.prometheus_metrics = PrometheusMetrics()
        self.metrics_port = metrics_port
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start monitoring service."""
        if self._running:
            return
        
        self._running = True
        
        # Start Prometheus metrics server
        try:
            start_http_server(self.metrics_port, registry=self.prometheus_metrics.registry)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {str(e)}")
        
        # Start background monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring service started")
    
    async def stop(self):
        """Stop monitoring service."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                # Collect system metrics
                self.metrics_collector.record_system_metrics()
                
                # Update Prometheus metrics
                system_summary = self.metrics_collector.get_system_metrics_summary()
                if system_summary:
                    self.prometheus_metrics.update_system_metrics(
                        system_summary["cpu_percent"],
                        system_summary["memory_percent"]
                    )
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def record_request_metrics(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        self.prometheus_metrics.record_request(method, endpoint, status_code, duration)
        self.metrics_collector.record_metric(
            "requests.duration",
            duration,
            {"method": method, "endpoint": endpoint, "status": str(status_code)}
        )
    
    def record_agent_metrics(self, agent_id: str, capability: str, status: str, duration: float):
        """Record agent execution metrics."""
        self.prometheus_metrics.record_agent_execution(agent_id, capability, status, duration)
        self.metrics_collector.record_metric(
            "agents.execution_duration",
            duration,
            {"agent_id": agent_id, "capability": capability, "status": status}
        )
    
    def record_workflow_metrics(self, workflow_id: str, status: str, duration: float):
        """Record workflow execution metrics."""
        self.prometheus_metrics.record_workflow_execution(workflow_id, status, duration)
        self.metrics_collector.record_metric(
            "workflows.execution_duration",
            duration,
            {"workflow_id": workflow_id, "status": status}
        )
    
    def record_document_metrics(self, document_type: str, format: str, status: str, duration: float):
        """Record document generation metrics."""
        self.prometheus_metrics.record_document_generation(document_type, format, status, duration)
        self.metrics_collector.record_metric(
            "documents.generation_duration",
            duration,
            {"document_type": document_type, "format": format, "status": status}
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "system": self.metrics_collector.get_system_metrics_summary(),
            "requests": self.metrics_collector.get_metric_summary("requests.duration"),
            "agents": self.metrics_collector.get_metric_summary("agents.execution_duration"),
            "workflows": self.metrics_collector.get_metric_summary("workflows.execution_duration"),
            "documents": self.metrics_collector.get_metric_summary("documents.generation_duration")
        }

# Global monitoring service instance
monitoring_service = MonitoringService()

def monitor_execution(operation_type: str):
    """Decorator for monitoring function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                monitoring_service.metrics_collector.record_metric(
                    f"{operation_type}.execution_duration",
                    duration,
                    {"status": status}
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                monitoring_service.metrics_collector.record_metric(
                    f"{operation_type}.execution_duration",
                    duration,
                    {"status": status}
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
