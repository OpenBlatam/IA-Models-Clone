"""
NLP Metrics and Monitoring System
=================================

Sistema avanzado de métricas y monitoreo para el sistema NLP.
Incluye métricas de rendimiento, calidad, uso y alertas.
"""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque
import threading
import psutil
import torch

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Tipos de métricas."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AlertLevel(str, Enum):
    """Niveles de alerta."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Métrica individual."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    """Alerta del sistema."""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: str
    threshold: float
    current_value: float
    resolved: bool = False

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento."""
    request_count: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    p50_processing_time: float = 0.0
    p95_processing_time: float = 0.0
    p99_processing_time: float = 0.0
    error_count: int = 0
    success_rate: float = 100.0
    throughput_per_second: float = 0.0

@dataclass
class QualityMetrics:
    """Métricas de calidad."""
    sentiment_accuracy: float = 0.0
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    keyword_relevance: float = 0.0
    topic_coherence: float = 0.0
    readability_consistency: float = 0.0

@dataclass
class SystemMetrics:
    """Métricas del sistema."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: float = 0.0
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    model_load_time: float = 0.0

class NLPMonitoringSystem:
    """Sistema de monitoreo avanzado para NLP."""
    
    def __init__(self, retention_days: int = 30):
        """Initialize monitoring system."""
        self.retention_days = retention_days
        self.metrics_history: deque = deque(maxlen=retention_days * 24 * 60)  # 1 minute intervals
        self.alerts: List[Alert] = []
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()
        self.system_metrics = SystemMetrics()
        
        # Thread-safe collections
        self._lock = threading.RLock()
        self._processing_times: deque = deque(maxlen=1000)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._task_counts: Dict[str, int] = defaultdict(int)
        
        # Background monitoring
        self._monitoring_task = None
        self._running = False
        
        # Alert thresholds
        self.thresholds = {
            'processing_time_ms': 5000,  # 5 seconds
            'error_rate_percent': 5.0,   # 5%
            'memory_usage_percent': 80.0, # 80%
            'cpu_usage_percent': 90.0,   # 90%
            'cache_hit_rate_percent': 70.0, # 70%
            'gpu_memory_usage_percent': 90.0 # 90%
        }
    
    async def start_monitoring(self):
        """Start background monitoring."""
        if not self._running:
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitor_system())
            logger.info("NLP monitoring system started")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        if self._running:
            self._running = False
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("NLP monitoring system stopped")
    
    async def _monitor_system(self):
        """Background system monitoring."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await self._check_alerts()
                await asyncio.sleep(60)  # Monitor every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU and Memory
            self.system_metrics.cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.system_metrics.memory_usage = memory.percent
            
            # GPU metrics (if available)
            if torch.cuda.is_available():
                self.system_metrics.gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                self.system_metrics.gpu_memory_usage = gpu_memory * 100
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_metrics.disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.system_metrics.network_io = net_io.bytes_sent + net_io.bytes_recv
            
            # Update metrics history
            await self._record_metric('system.cpu_usage', self.system_metrics.cpu_usage)
            await self._record_metric('system.memory_usage', self.system_metrics.memory_usage)
            await self._record_metric('system.gpu_usage', self.system_metrics.gpu_usage)
            await self._record_metric('system.disk_usage', self.system_metrics.disk_usage)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def record_request(
        self,
        task: str,
        processing_time: float,
        success: bool = True,
        error_type: str = None,
        language: str = "en",
        text_length: int = 0
    ):
        """Record NLP request metrics."""
        with self._lock:
            # Update performance metrics
            self.performance_metrics.request_count += 1
            self.performance_metrics.total_processing_time += processing_time
            self._processing_times.append(processing_time)
            
            if not success:
                self.performance_metrics.error_count += 1
                if error_type:
                    self._error_counts[error_type] += 1
            
            self._task_counts[task] += 1
            
            # Calculate statistics
            if self._processing_times:
                times = list(self._processing_times)
                self.performance_metrics.average_processing_time = statistics.mean(times)
                self.performance_metrics.min_processing_time = min(times)
                self.performance_metrics.max_processing_time = max(times)
                
                if len(times) >= 2:
                    self.performance_metrics.p50_processing_time = statistics.median(times)
                    if len(times) >= 20:
                        self.performance_metrics.p95_processing_time = statistics.quantiles(times, n=20)[18]
                        self.performance_metrics.p99_processing_time = statistics.quantiles(times, n=100)[98]
            
            # Calculate success rate
            if self.performance_metrics.request_count > 0:
                self.performance_metrics.success_rate = (
                    (self.performance_metrics.request_count - self.performance_metrics.error_count) /
                    self.performance_metrics.request_count * 100
                )
            
            # Calculate throughput
            current_time = time.time()
            # This is a simplified calculation - in practice, you'd use a sliding window
            
        # Record detailed metrics
        await self._record_metric('requests.total', 1, tags={'task': task, 'language': language})
        await self._record_metric('requests.processing_time', processing_time, tags={'task': task})
        await self._record_metric('requests.text_length', text_length, tags={'task': task})
        
        if success:
            await self._record_metric('requests.success', 1, tags={'task': task})
        else:
            await self._record_metric('requests.error', 1, tags={'task': task, 'error_type': error_type or 'unknown'})
    
    async def record_quality_metrics(
        self,
        task: str,
        accuracy: float = None,
        precision: float = None,
        recall: float = None,
        f1: float = None,
        coherence: float = None
    ):
        """Record quality metrics."""
        if accuracy is not None:
            await self._record_metric('quality.accuracy', accuracy, tags={'task': task})
        
        if precision is not None:
            await self._record_metric('quality.precision', precision, tags={'task': task})
        
        if recall is not None:
            await self._record_metric('quality.recall', recall, tags={'task': task})
        
        if f1 is not None:
            await self._record_metric('quality.f1', f1, tags={'task': task})
        
        if coherence is not None:
            await self._record_metric('quality.coherence', coherence, tags={'task': task})
    
    async def _record_metric(
        self,
        name: str,
        value: float,
        tags: Dict[str, str] = None,
        metric_type: MetricType = MetricType.GAUGE
    ):
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=metric_type
        )
        
        self.metrics_history.append(metric)
    
    async def _check_alerts(self):
        """Check for alert conditions."""
        # Check processing time
        if self.performance_metrics.average_processing_time > self.thresholds['processing_time_ms']:
            await self._create_alert(
                'high_processing_time',
                AlertLevel.WARNING,
                f"Average processing time is {self.performance_metrics.average_processing_time:.2f}ms",
                'processing_time_ms',
                self.thresholds['processing_time_ms'],
                self.performance_metrics.average_processing_time
            )
        
        # Check error rate
        if self.performance_metrics.success_rate < (100 - self.thresholds['error_rate_percent']):
            await self._create_alert(
                'high_error_rate',
                AlertLevel.ERROR,
                f"Error rate is {100 - self.performance_metrics.success_rate:.2f}%",
                'error_rate_percent',
                self.thresholds['error_rate_percent'],
                100 - self.performance_metrics.success_rate
            )
        
        # Check memory usage
        if self.system_metrics.memory_usage > self.thresholds['memory_usage_percent']:
            await self._create_alert(
                'high_memory_usage',
                AlertLevel.WARNING,
                f"Memory usage is {self.system_metrics.memory_usage:.2f}%",
                'memory_usage_percent',
                self.thresholds['memory_usage_percent'],
                self.system_metrics.memory_usage
            )
        
        # Check CPU usage
        if self.system_metrics.cpu_usage > self.thresholds['cpu_usage_percent']:
            await self._create_alert(
                'high_cpu_usage',
                AlertLevel.WARNING,
                f"CPU usage is {self.system_metrics.cpu_usage:.2f}%",
                'cpu_usage_percent',
                self.thresholds['cpu_usage_percent'],
                self.system_metrics.cpu_usage
            )
    
    async def _create_alert(
        self,
        alert_id: str,
        level: AlertLevel,
        message: str,
        metric_name: str,
        threshold: float,
        current_value: float
    ):
        """Create an alert."""
        # Check if alert already exists and is not resolved
        existing_alert = next(
            (a for a in self.alerts if a.id == alert_id and not a.resolved),
            None
        )
        
        if not existing_alert:
            alert = Alert(
                id=alert_id,
                level=level,
                message=message,
                timestamp=datetime.now(),
                metric_name=metric_name,
                threshold=threshold,
                current_value=current_value
            )
            self.alerts.append(alert)
            logger.warning(f"Alert created: {message}")
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        with self._lock:
            return {
                'request_count': self.performance_metrics.request_count,
                'total_processing_time': self.performance_metrics.total_processing_time,
                'average_processing_time': self.performance_metrics.average_processing_time,
                'min_processing_time': self.performance_metrics.min_processing_time,
                'max_processing_time': self.performance_metrics.max_processing_time,
                'p50_processing_time': self.performance_metrics.p50_processing_time,
                'p95_processing_time': self.performance_metrics.p95_processing_time,
                'p99_processing_time': self.performance_metrics.p99_processing_time,
                'error_count': self.performance_metrics.error_count,
                'success_rate': self.performance_metrics.success_rate,
                'throughput_per_second': self.performance_metrics.throughput_per_second,
                'task_distribution': dict(self._task_counts),
                'error_distribution': dict(self._error_counts)
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            'cpu_usage': self.system_metrics.cpu_usage,
            'memory_usage': self.system_metrics.memory_usage,
            'gpu_usage': self.system_metrics.gpu_usage,
            'gpu_memory_usage': self.system_metrics.gpu_memory_usage,
            'disk_usage': self.system_metrics.disk_usage,
            'network_io': self.system_metrics.network_io,
            'active_connections': self.system_metrics.active_connections,
            'cache_hit_rate': self.system_metrics.cache_hit_rate,
            'model_load_time': self.system_metrics.model_load_time
        }
    
    def get_alerts(self, level: AlertLevel = None, resolved: bool = None) -> List[Dict[str, Any]]:
        """Get alerts."""
        filtered_alerts = self.alerts
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        
        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
        
        return [
            {
                'id': alert.id,
                'level': alert.level.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metric_name': alert.metric_name,
                'threshold': alert.threshold,
                'current_value': alert.current_value,
                'resolved': alert.resolved
            }
            for alert in filtered_alerts
        ]
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)
        
        summary = {}
        for name, values in metric_groups.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'median': statistics.median(values)
                }
        
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_score = 100
        
        # Deduct points for issues
        if self.performance_metrics.success_rate < 95:
            health_score -= 20
        
        if self.system_metrics.memory_usage > 80:
            health_score -= 15
        
        if self.system_metrics.cpu_usage > 90:
            health_score -= 15
        
        if self.performance_metrics.average_processing_time > 5000:
            health_score -= 10
        
        # Count unresolved alerts
        unresolved_alerts = len([a for a in self.alerts if not a.resolved])
        health_score -= min(unresolved_alerts * 5, 30)
        
        # Determine status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "warning"
        elif health_score >= 50:
            status = "degraded"
        else:
            status = "critical"
        
        return {
            'status': status,
            'health_score': max(0, health_score),
            'unresolved_alerts': unresolved_alerts,
            'uptime': 'N/A',  # Would need to track start time
            'last_updated': datetime.now().isoformat()
        }

# Global monitoring instance
nlp_monitoring = NLPMonitoringSystem()












