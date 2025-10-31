#!/usr/bin/env python3
"""
Metrics Collector for Enhanced HeyGen AI
Collects and exposes metrics for Prometheus monitoring.
"""

import asyncio
import time
import psutil
import structlog
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST
)
import threading
from datetime import datetime, timedelta

logger = structlog.get_logger()

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"

@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str]
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None

class MetricsCollector:
    """Collects and manages metrics for the HeyGen AI system."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.collectors: Dict[str, Callable] = {}
        self.is_running = False
        
        # Background collection task
        self.collection_task: Optional[asyncio.Task] = None
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Start background collection
        self._start_background_collection()
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # =============================================================================
        # System Metrics
        # =============================================================================
        
        # CPU and Memory
        self.metrics["cpu_usage_percent"] = Gauge(
            "heygen_cpu_usage_percent",
            "CPU usage percentage",
            ["component"]
        )
        
        self.metrics["memory_usage_bytes"] = Gauge(
            "heygen_memory_usage_bytes",
            "Memory usage in bytes",
            ["component"]
        )
        
        self.metrics["memory_usage_percent"] = Gauge(
            "heygen_memory_usage_percent",
            "Memory usage percentage",
            ["component"]
        )
        
        # Disk and Network
        self.metrics["disk_usage_percent"] = Gauge(
            "heygen_disk_usage_percent",
            "Disk usage percentage",
            ["mount_point"]
        )
        
        self.metrics["network_bytes_sent"] = Counter(
            "heygen_network_bytes_sent_total",
            "Total network bytes sent",
            ["interface"]
        )
        
        self.metrics["network_bytes_recv"] = Counter(
            "heygen_network_bytes_recv_total",
            "Total network bytes received",
            ["interface"]
        )
        
        # =============================================================================
        # Application Metrics
        # =============================================================================
        
        # Request counts
        self.metrics["http_requests_total"] = Counter(
            "heygen_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"]
        )
        
        self.metrics["http_request_duration_seconds"] = Histogram(
            "heygen_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        # Video generation metrics
        self.metrics["video_generation_requests_total"] = Counter(
            "heygen_video_generation_requests_total",
            "Total video generation requests",
            ["status", "quality_preset", "resolution"]
        )
        
        self.metrics["video_generation_duration_seconds"] = Histogram(
            "heygen_video_generation_duration_seconds",
            "Video generation duration in seconds",
            ["quality_preset", "resolution"],
            buckets=[10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0]
        )
        
        self.metrics["video_generation_queue_size"] = Gauge(
            "heygen_video_generation_queue_size",
            "Current video generation queue size"
        )
        
        # Voice synthesis metrics
        self.metrics["voice_synthesis_requests_total"] = Counter(
            "heygen_voice_synthesis_requests_total",
            "Total voice synthesis requests",
            ["status", "voice_engine", "language"]
        )
        
        self.metrics["voice_synthesis_duration_seconds"] = Histogram(
            "heygen_voice_synthesis_duration_seconds",
            "Voice synthesis duration in seconds",
            ["voice_engine", "language"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        # Avatar generation metrics
        self.metrics["avatar_generation_requests_total"] = Counter(
            "heygen_avatar_generation_requests_total",
            "Total avatar generation requests",
            ["status", "model", "resolution"]
        )
        
        self.metrics["avatar_generation_duration_seconds"] = Histogram(
            "heygen_avatar_generation_duration_seconds",
            "Avatar generation duration in seconds",
            ["model", "resolution"],
            buckets=[5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0]
        )
        
        # =============================================================================
        # Cache Metrics
        # =============================================================================
        
        self.metrics["cache_hits_total"] = Counter(
            "heygen_cache_hits_total",
            "Total cache hits",
            ["cache_type"]
        )
        
        self.metrics["cache_misses_total"] = Counter(
            "heygen_cache_misses_total",
            "Total cache misses",
            ["cache_type"]
        )
        
        self.metrics["cache_size_bytes"] = Gauge(
            "heygen_cache_size_bytes",
            "Current cache size in bytes",
            ["cache_type"]
        )
        
        self.metrics["cache_entries_total"] = Gauge(
            "heygen_cache_entries_total",
            "Total number of cache entries",
            ["cache_type"]
        )
        
        # =============================================================================
        # Queue Metrics
        # =============================================================================
        
        self.metrics["queue_tasks_total"] = Counter(
            "heygen_queue_tasks_total",
            "Total tasks processed by queue",
            ["task_type", "status"]
        )
        
        self.metrics["queue_duration_seconds"] = Histogram(
            "heygen_queue_duration_seconds",
            "Task processing duration in seconds",
            ["task_type"],
            buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0]
        )
        
        self.metrics["queue_size"] = Gauge(
            "heygen_queue_size",
            "Current queue size",
            ["queue_type"]
        )
        
        self.metrics["queue_workers_active"] = Gauge(
            "heygen_queue_workers_active",
            "Number of active queue workers"
        )
        
        # =============================================================================
        # Webhook Metrics
        # =============================================================================
        
        self.metrics["webhook_events_total"] = Counter(
            "heygen_webhook_events_total",
            "Total webhook events sent",
            ["event_type", "status"]
        )
        
        self.metrics["webhook_delivery_duration_seconds"] = Histogram(
            "heygen_webhook_delivery_duration_seconds",
            "Webhook delivery duration in seconds",
            ["endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.metrics["webhook_endpoints_active"] = Gauge(
            "heygen_webhook_endpoints_active",
            "Number of active webhook endpoints"
        )
        
        # =============================================================================
        # Rate Limiting Metrics
        # =============================================================================
        
        self.metrics["rate_limit_requests_total"] = Counter(
            "heygen_rate_limit_requests_total",
            "Total rate limit checks",
            ["user_tier", "allowed"]
        )
        
        self.metrics["rate_limit_violations_total"] = Counter(
            "heygen_rate_limit_violations_total",
            "Total rate limit violations",
            ["user_tier"]
        )
        
        self.metrics["rate_limit_blocked_users"] = Gauge(
            "heygen_rate_limit_blocked_users",
            "Number of currently blocked users"
        )
        
        # =============================================================================
        # Model Metrics
        # =============================================================================
        
        self.metrics["model_inference_requests_total"] = Counter(
            "heygen_model_inference_requests_total",
            "Total model inference requests",
            ["model_type", "status"]
        )
        
        self.metrics["model_inference_duration_seconds"] = Histogram(
            "heygen_model_inference_duration_seconds",
            "Model inference duration in seconds",
            ["model_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.metrics["model_memory_usage_bytes"] = Gauge(
            "heygen_model_memory_usage_bytes",
            "Model memory usage in bytes",
            ["model_type"]
        )
        
        # =============================================================================
        # Error Metrics
        # =============================================================================
        
        self.metrics["errors_total"] = Counter(
            "heygen_errors_total",
            "Total errors",
            ["component", "error_type"]
        )
        
        # =============================================================================
        # Business Metrics
        # =============================================================================
        
        self.metrics["videos_generated_total"] = Counter(
            "heygen_videos_generated_total",
            "Total videos generated",
            ["quality_preset", "resolution"]
        )
        
        self.metrics["active_users"] = Gauge(
            "heygen_active_users",
            "Number of active users",
            ["time_window"]
        )
        
        self.metrics["revenue_total"] = Counter(
            "heygen_revenue_total",
            "Total revenue generated",
            ["currency", "tier"]
        )
        
        # =============================================================================
        # System Info
        # =============================================================================
        
        self.metrics["system_info"] = Info(
            "heygen_system",
            "System information"
        )
        
        # Set system info
        self.metrics["system_info"].info({
            "version": "2.0.0",
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            "platform": psutil.sys.platform,
            "architecture": psutil.sys.maxsize > 2**32 and "64bit" or "32bit"
        })
    
    def _start_background_collection(self):
        """Start background metric collection."""
        self.is_running = True
        self.collection_task = asyncio.create_task(self._collect_metrics())
    
    async def _collect_metrics(self):
        """Background task to collect metrics."""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await asyncio.sleep(15)  # Collect every 15 seconds
                
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics["cpu_usage_percent"].labels(component="system").set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics["memory_usage_bytes"].labels(component="system").set(memory.used)
            self.metrics["memory_usage_percent"].labels(component="system").set(memory.percent)
            
            # Disk metrics
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.metrics["disk_usage_percent"].labels(mount_point=partition.mountpoint).set(
                        (usage.used / usage.total) * 100
                    )
                except (PermissionError, FileNotFoundError):
                    continue
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.metrics["network_bytes_sent"].labels(interface="total").inc(
                net_io.bytes_sent - getattr(self, '_last_net_sent', net_io.bytes_sent)
            )
            self.metrics["network_bytes_recv"].labels(interface="total").inc(
                net_io.bytes_recv - getattr(self, '_last_net_recv', net_io.bytes_recv)
            )
            
            self._last_net_sent = net_io.bytes_sent
            self._last_net_recv = net_io.bytes_recv
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-level metrics."""
        try:
            # This would typically collect metrics from various components
            # For now, we'll just log that collection happened
            pass
            
        except Exception as e:
            logger.error(f"Application metrics collection error: {e}")
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        try:
            self.metrics["http_requests_total"].labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            self.metrics["http_request_duration_seconds"].labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Failed to record HTTP request metrics: {e}")
    
    def record_video_generation(self, status: str, quality_preset: str, resolution: str, duration: float):
        """Record video generation metrics."""
        try:
            self.metrics["video_generation_requests_total"].labels(
                status=status,
                quality_preset=quality_preset,
                resolution=resolution
            ).inc()
            
            if duration > 0:
                self.metrics["video_generation_duration_seconds"].labels(
                    quality_preset=quality_preset,
                    resolution=resolution
                ).observe(duration)
            
            if status == "completed":
                self.metrics["videos_generated_total"].labels(
                    quality_preset=quality_preset,
                    resolution=resolution
                ).inc()
                
        except Exception as e:
            logger.error(f"Failed to record video generation metrics: {e}")
    
    def record_voice_synthesis(self, status: str, voice_engine: str, language: str, duration: float):
        """Record voice synthesis metrics."""
        try:
            self.metrics["voice_synthesis_requests_total"].labels(
                status=status,
                voice_engine=voice_engine,
                language=language
            ).inc()
            
            if duration > 0:
                self.metrics["voice_synthesis_duration_seconds"].labels(
                    voice_engine=voice_engine,
                    language=language
                ).observe(duration)
                
        except Exception as e:
            logger.error(f"Failed to record voice synthesis metrics: {e}")
    
    def record_avatar_generation(self, status: str, model: str, resolution: str, duration: float):
        """Record avatar generation metrics."""
        try:
            self.metrics["avatar_generation_requests_total"].labels(
                status=status,
                model=model,
                resolution=resolution
            ).inc()
            
            if duration > 0:
                self.metrics["avatar_generation_duration_seconds"].labels(
                    model=model,
                    resolution=resolution
                ).observe(duration)
                
        except Exception as e:
            logger.error(f"Failed to record avatar generation metrics: {e}")
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation metrics."""
        try:
            if hit:
                self.metrics["cache_hits_total"].labels(cache_type=cache_type).inc()
            else:
                self.metrics["cache_misses_total"].labels(cache_type=cache_type).inc()
                
        except Exception as e:
            logger.error(f"Failed to record cache metrics: {e}")
    
    def update_cache_size(self, cache_type: str, size_bytes: int, entries: int):
        """Update cache size metrics."""
        try:
            self.metrics["cache_size_bytes"].labels(cache_type=cache_type).set(size_bytes)
            self.metrics["cache_entries_total"].labels(cache_type=cache_type).set(entries)
            
        except Exception as e:
            logger.error(f"Failed to update cache size metrics: {e}")
    
    def record_queue_task(self, task_type: str, status: str, duration: float = 0):
        """Record queue task metrics."""
        try:
            self.metrics["queue_tasks_total"].labels(
                task_type=task_type,
                status=status
            ).inc()
            
            if duration > 0:
                self.metrics["queue_duration_seconds"].labels(task_type=task_type).observe(duration)
                
        except Exception as e:
            logger.error(f"Failed to record queue metrics: {e}")
    
    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size metrics."""
        try:
            self.metrics["queue_size"].labels(queue_type=queue_type).set(size)
            
        except Exception as e:
            logger.error(f"Failed to update queue size metrics: {e}")
    
    def record_webhook_event(self, event_type: str, status: str):
        """Record webhook event metrics."""
        try:
            self.metrics["webhook_events_total"].labels(
                event_type=event_type,
                status=status
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record webhook metrics: {e}")
    
    def record_rate_limit_check(self, user_tier: str, allowed: bool):
        """Record rate limit check metrics."""
        try:
            self.metrics["rate_limit_requests_total"].labels(
                user_tier=user_tier,
                allowed=str(allowed).lower()
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record rate limit metrics: {e}")
    
    def record_error(self, component: str, error_type: str):
        """Record error metrics."""
        try:
            self.metrics["errors_total"].labels(
                component=component,
                error_type=error_type
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record error metrics: {e}")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return ""
    
    def get_metrics_content_type(self) -> str:
        """Get the content type for metrics."""
        return CONTENT_TYPE_LATEST
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        try:
            summary = {}
            
            # Collect gauge values
            for name, metric in self.metrics.items():
                if hasattr(metric, '_value'):
                    if hasattr(metric, '_labels'):
                        # Labeled metric
                        for label_values in metric._metrics.keys():
                            label_str = ','.join(f'{k}="{v}"' for k, v in label_values)
                            summary[f"{name}{{{label_str}}}"] = metric._metrics[label_values]._value._value
                    else:
                        # Simple metric
                        summary[name] = metric._value._value
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the metrics collector."""
        self.is_running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics collector shutdown complete")

# Global metrics collector instance
metrics_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
    return metrics_collector

async def shutdown_metrics_collector():
    """Shutdown global metrics collector."""
    global metrics_collector
    if metrics_collector:
        await metrics_collector.shutdown()
        metrics_collector = None

