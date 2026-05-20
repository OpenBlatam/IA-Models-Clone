"""
Metrics Service - Advanced Implementation
========================================

Advanced metrics service with comprehensive system and application metrics collection.
"""

from __future__ import annotations
import logging
import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class MetricCategory(str, Enum):
    """Metric category enumeration"""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    API = "api"
    WEBSOCKET = "websocket"
    BACKGROUND = "background"
    SCHEDULER = "scheduler"


@dataclass
class Metric:
    """Metric data class"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    category: MetricCategory
    labels: Dict[str, str]
    timestamp: datetime
    metadata: Dict[str, Any]


class MetricsService:
    """Advanced metrics service with comprehensive metrics collection"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.summaries: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # System metrics collection
        self.system_metrics_enabled = True
        self.application_metrics_enabled = True
        self.collection_interval = 60  # seconds
        self.retention_period = 24 * 60 * 60  # 24 hours in seconds
        self.max_metrics_per_name = 1000
        
        # Background collection
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_metrics_collected": 0,
            "metrics_by_type": {metric_type.value: 0 for metric_type in MetricType},
            "metrics_by_category": {category.value: 0 for category in MetricCategory},
            "collection_errors": 0,
            "last_collection": None
        }
    
    async def start(self):
        """Start metrics collection"""
        try:
            if not self.is_collecting:
                self.is_collecting = True
                self.collection_task = asyncio.create_task(self._collection_loop())
                logger.info("Metrics service started")
        
        except Exception as e:
            logger.error(f"Failed to start metrics service: {e}")
            raise
    
    async def stop(self):
        """Stop metrics collection"""
        try:
            if self.is_collecting:
                self.is_collecting = False
                
                if self.collection_task:
                    self.collection_task.cancel()
                    try:
                        await self.collection_task
                    except asyncio.CancelledError:
                        pass
                
                logger.info("Metrics service stopped")
        
        except Exception as e:
            logger.error(f"Failed to stop metrics service: {e}")
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        category: MetricCategory = MetricCategory.APPLICATION,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment counter metric"""
        try:
            with self.lock:
                self.counters[name] += value
                
                metric = Metric(
                    name=name,
                    value=self.counters[name],
                    metric_type=MetricType.COUNTER,
                    category=category,
                    labels=labels or {},
                    timestamp=datetime.utcnow(),
                    metadata={}
                )
                
                self._store_metric(metric)
                self._update_statistics(metric)
        
        except Exception as e:
            logger.error(f"Failed to increment counter: {e}")
    
    def set_gauge(
        self,
        name: str,
        value: float,
        category: MetricCategory = MetricCategory.APPLICATION,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set gauge metric"""
        try:
            with self.lock:
                self.gauges[name] = value
                
                metric = Metric(
                    name=name,
                    value=value,
                    metric_type=MetricType.GAUGE,
                    category=category,
                    labels=labels or {},
                    timestamp=datetime.utcnow(),
                    metadata={}
                )
                
                self._store_metric(metric)
                self._update_statistics(metric)
        
        except Exception as e:
            logger.error(f"Failed to set gauge: {e}")
    
    def record_histogram(
        self,
        name: str,
        value: float,
        category: MetricCategory = MetricCategory.APPLICATION,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record histogram metric"""
        try:
            with self.lock:
                self.histograms[name].append(value)
                
                # Keep only recent values
                if len(self.histograms[name]) > self.max_metrics_per_name:
                    self.histograms[name] = self.histograms[name][-self.max_metrics_per_name:]
                
                metric = Metric(
                    name=name,
                    value=value,
                    metric_type=MetricType.HISTOGRAM,
                    category=category,
                    labels=labels or {},
                    timestamp=datetime.utcnow(),
                    metadata={
                        "count": len(self.histograms[name]),
                        "sum": sum(self.histograms[name]),
                        "min": min(self.histograms[name]),
                        "max": max(self.histograms[name]),
                        "avg": sum(self.histograms[name]) / len(self.histograms[name])
                    }
                )
                
                self._store_metric(metric)
                self._update_statistics(metric)
        
        except Exception as e:
            logger.error(f"Failed to record histogram: {e}")
    
    def record_timer(
        self,
        name: str,
        duration: float,
        category: MetricCategory = MetricCategory.APPLICATION,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record timer metric"""
        try:
            with self.lock:
                self.timers[name].append(duration)
                
                # Keep only recent values
                if len(self.timers[name]) > self.max_metrics_per_name:
                    self.timers[name] = self.timers[name][-self.max_metrics_per_name:]
                
                metric = Metric(
                    name=name,
                    value=duration,
                    metric_type=MetricType.TIMER,
                    category=category,
                    labels=labels or {},
                    timestamp=datetime.utcnow(),
                    metadata={
                        "count": len(self.timers[name]),
                        "sum": sum(self.timers[name]),
                        "min": min(self.timers[name]),
                        "max": max(self.timers[name]),
                        "avg": sum(self.timers[name]) / len(self.timers[name])
                    }
                )
                
                self._store_metric(metric)
                self._update_statistics(metric)
        
        except Exception as e:
            logger.error(f"Failed to record timer: {e}")
    
    def record_summary(
        self,
        name: str,
        value: float,
        category: MetricCategory = MetricCategory.APPLICATION,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record summary metric"""
        try:
            with self.lock:
                if name not in self.summaries:
                    self.summaries[name] = {
                        "count": 0,
                        "sum": 0.0,
                        "min": float('inf'),
                        "max": float('-inf')
                    }
                
                summary = self.summaries[name]
                summary["count"] += 1
                summary["sum"] += value
                summary["min"] = min(summary["min"], value)
                summary["max"] = max(summary["max"], value)
                
                metric = Metric(
                    name=name,
                    value=value,
                    metric_type=MetricType.SUMMARY,
                    category=category,
                    labels=labels or {},
                    timestamp=datetime.utcnow(),
                    metadata=summary.copy()
                )
                
                self._store_metric(metric)
                self._update_statistics(metric)
        
        except Exception as e:
            logger.error(f"Failed to record summary: {e}")
    
    def get_metric(
        self,
        name: str,
        metric_type: Optional[MetricType] = None,
        category: Optional[MetricCategory] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get metric data"""
        try:
            with self.lock:
                metrics = self.metrics.get(name, [])
                
                # Filter by type and category
                if metric_type:
                    metrics = [m for m in metrics if m.metric_type == metric_type]
                if category:
                    metrics = [m for m in metrics if m.category == category]
                
                # Sort by timestamp (newest first)
                metrics.sort(key=lambda x: x.timestamp, reverse=True)
                
                # Convert to dict format
                result = []
                for metric in metrics[:limit]:
                    result.append({
                        "name": metric.name,
                        "value": metric.value,
                        "type": metric.metric_type.value,
                        "category": metric.category.value,
                        "labels": metric.labels,
                        "timestamp": metric.timestamp.isoformat(),
                        "metadata": metric.metadata
                    })
                
                return result
        
        except Exception as e:
            logger.error(f"Failed to get metric: {e}")
            return []
    
    def get_metric_summary(
        self,
        name: str,
        metric_type: Optional[MetricType] = None,
        category: Optional[MetricCategory] = None,
        time_range: Optional[timedelta] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metric summary"""
        try:
            with self.lock:
                metrics = self.metrics.get(name, [])
                
                # Filter by type and category
                if metric_type:
                    metrics = [m for m in metrics if m.metric_type == metric_type]
                if category:
                    metrics = [m for m in metrics if m.category == category]
                
                # Filter by time range
                if time_range:
                    cutoff = datetime.utcnow() - time_range
                    metrics = [m for m in metrics if m.timestamp >= cutoff]
                
                if not metrics:
                    return None
                
                values = [m.value for m in metrics]
                
                return {
                    "name": name,
                    "type": metrics[0].metric_type.value,
                    "category": metrics[0].category.value,
                    "count": len(values),
                    "sum": sum(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "first_timestamp": min(m.timestamp for m in metrics).isoformat(),
                    "last_timestamp": max(m.timestamp for m in metrics).isoformat()
                }
        
        except Exception as e:
            logger.error(f"Failed to get metric summary: {e}")
            return None
    
    def get_all_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        category: Optional[MetricCategory] = None,
        time_range: Optional[timedelta] = None,
        limit: int = 1000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics with filtering"""
        try:
            with self.lock:
                result = {}
                
                for name, metrics in self.metrics.items():
                    filtered_metrics = metrics
                    
                    # Filter by type and category
                    if metric_type:
                        filtered_metrics = [m for m in filtered_metrics if m.metric_type == metric_type]
                    if category:
                        filtered_metrics = [m for m in filtered_metrics if m.category == category]
                    
                    # Filter by time range
                    if time_range:
                        cutoff = datetime.utcnow() - time_range
                        filtered_metrics = [m for m in filtered_metrics if m.timestamp >= cutoff]
                    
                    if filtered_metrics:
                        # Sort by timestamp (newest first)
                        filtered_metrics.sort(key=lambda x: x.timestamp, reverse=True)
                        
                        # Convert to dict format
                        result[name] = []
                        for metric in filtered_metrics[:limit]:
                            result[name].append({
                                "name": metric.name,
                                "value": metric.value,
                                "type": metric.metric_type.value,
                                "category": metric.category.value,
                                "labels": metric.labels,
                                "timestamp": metric.timestamp.isoformat(),
                                "metadata": metric.metadata
                            })
                
                return result
        
        except Exception as e:
            logger.error(f"Failed to get all metrics: {e}")
            return {}
    
    def get_metrics_stats(self) -> Dict[str, Any]:
        """Get metrics service statistics"""
        try:
            with self.lock:
                return {
                    "is_collecting": self.is_collecting,
                    "total_metrics_collected": self.stats["total_metrics_collected"],
                    "metrics_by_type": self.stats["metrics_by_type"],
                    "metrics_by_category": self.stats["metrics_by_category"],
                    "collection_errors": self.stats["collection_errors"],
                    "last_collection": self.stats["last_collection"],
                    "counters_count": len(self.counters),
                    "gauges_count": len(self.gauges),
                    "histograms_count": len(self.histograms),
                    "timers_count": len(self.timers),
                    "summaries_count": len(self.summaries),
                    "collection_interval": self.collection_interval,
                    "retention_period": self.retention_period,
                    "max_metrics_per_name": self.max_metrics_per_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Failed to get metrics stats: {e}")
            return {"error": str(e)}
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        try:
            while self.is_collecting:
                try:
                    if self.system_metrics_enabled:
                        await self._collect_system_metrics()
                    
                    if self.application_metrics_enabled:
                        await self._collect_application_metrics()
                    
                    self.stats["last_collection"] = datetime.utcnow().isoformat()
                    
                except Exception as e:
                    self.stats["collection_errors"] += 1
                    logger.error(f"Metrics collection error: {e}")
                
                await asyncio.sleep(self.collection_interval)
        
        except asyncio.CancelledError:
            logger.info("Metrics collection loop cancelled")
        except Exception as e:
            logger.error(f"Metrics collection loop error: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("system.cpu.percent", cpu_percent, MetricCategory.CPU)
            
            cpu_count = psutil.cpu_count()
            self.set_gauge("system.cpu.count", cpu_count, MetricCategory.CPU)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system.memory.total", memory.total, MetricCategory.MEMORY)
            self.set_gauge("system.memory.available", memory.available, MetricCategory.MEMORY)
            self.set_gauge("system.memory.percent", memory.percent, MetricCategory.MEMORY)
            self.set_gauge("system.memory.used", memory.used, MetricCategory.MEMORY)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.set_gauge("system.disk.total", disk.total, MetricCategory.DISK)
            self.set_gauge("system.disk.used", disk.used, MetricCategory.DISK)
            self.set_gauge("system.disk.free", disk.free, MetricCategory.DISK)
            self.set_gauge("system.disk.percent", (disk.used / disk.total) * 100, MetricCategory.DISK)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.set_gauge("system.network.bytes_sent", network.bytes_sent, MetricCategory.NETWORK)
            self.set_gauge("system.network.bytes_recv", network.bytes_recv, MetricCategory.NETWORK)
            self.set_gauge("system.network.packets_sent", network.packets_sent, MetricCategory.NETWORK)
            self.set_gauge("system.network.packets_recv", network.packets_recv, MetricCategory.NETWORK)
            
            # Process metrics
            process = psutil.Process()
            self.set_gauge("system.process.memory_percent", process.memory_percent(), MetricCategory.MEMORY)
            self.set_gauge("system.process.cpu_percent", process.cpu_percent(), MetricCategory.CPU)
            self.set_gauge("system.process.num_threads", process.num_threads(), MetricCategory.SYSTEM)
        
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application metrics"""
        try:
            # Application uptime
            uptime = time.time() - psutil.Process().create_time()
            self.set_gauge("application.uptime", uptime, MetricCategory.APPLICATION)
            
            # Python metrics
            import gc
            gc_stats = gc.get_stats()
            self.set_gauge("application.python.gc.collections", sum(stat['collections'] for stat in gc_stats), MetricCategory.APPLICATION)
            
            # Thread metrics
            import threading
            self.set_gauge("application.threads.active", threading.active_count(), MetricCategory.APPLICATION)
        
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
    
    def _store_metric(self, metric: Metric):
        """Store metric with retention management"""
        try:
            self.metrics[metric.name].append(metric)
            
            # Cleanup old metrics
            cutoff = datetime.utcnow() - timedelta(seconds=self.retention_period)
            self.metrics[metric.name] = [
                m for m in self.metrics[metric.name]
                if m.timestamp >= cutoff
            ]
            
            # Limit metrics per name
            if len(self.metrics[metric.name]) > self.max_metrics_per_name:
                self.metrics[metric.name] = self.metrics[metric.name][-self.max_metrics_per_name:]
        
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
    
    def _update_statistics(self, metric: Metric):
        """Update metrics statistics"""
        try:
            self.stats["total_metrics_collected"] += 1
            self.stats["metrics_by_type"][metric.metric_type.value] += 1
            self.stats["metrics_by_category"][metric.category.value] += 1
        
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")


# Global metrics service instance
metrics_service = MetricsService()

