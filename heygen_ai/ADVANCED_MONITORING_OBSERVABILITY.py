#!/usr/bin/env python3
"""
📊 HeyGen AI - Advanced Monitoring & Observability System
=========================================================

This module implements a comprehensive monitoring and observability system
that provides real-time insights, distributed tracing, metrics collection,
and intelligent alerting for the HeyGen AI system.
"""

import asyncio
import logging
import time
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import hashlib
import secrets
import base64
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import aiohttp
import asyncio
from aiohttp import web, WSMsgType
import ssl
import certifi
import psutil
import GPUtil
import threading
import queue
import traceback
import sys
import os
from collections import defaultdict, deque
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class TraceType(str, Enum):
    """Trace types"""
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    AI_INFERENCE = "ai_inference"
    AI_TRAINING = "ai_training"
    FILE_OPERATION = "file_operation"
    NETWORK_OPERATION = "network_operation"
    CUSTOM = "custom"

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trace:
    """Distributed trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    trace_type: TraceType = TraceType.CUSTOM
    status: str = "success"
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LogEntry:
    """Log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    source: str = ""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown: int = 300  # seconds
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """System health status"""
    overall_health: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    active_connections: int = 0
    error_rate: float = 0.0
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self):
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_definitions: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize metrics collector"""
        self.initialized = True
        logger.info("✅ Metrics Collector initialized")
    
    async def register_metric(self, name: str, metric_type: MetricType, 
                            description: str = "", labels: List[str] = None):
        """Register a new metric"""
        self.metric_definitions[name] = {
            'type': metric_type,
            'description': description,
            'labels': labels or [],
            'created_at': datetime.now()
        }
        logger.info(f"✅ Metric registered: {name}")
    
    async def record_metric(self, name: str, value: float, 
                          labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a metric value"""
        if not self.initialized:
            return False
        
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=self.metric_definitions.get(name, {}).get('type', MetricType.GAUGE),
                timestamp=datetime.now(),
                labels=labels or {},
                metadata=metadata or {}
            )
            
            # Store in buffer
            self.metrics_buffer[name].append(metric)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record metric {name}: {e}")
            return False
    
    async def get_metric_summary(self, name: str, time_window: int = 300) -> Dict[str, Any]:
        """Get metric summary for time window"""
        if name not in self.metrics_buffer:
            return {}
        
        # Get recent metrics
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        recent_metrics = [m for m in self.metrics_buffer[name] 
                         if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'time_window': time_window
        }
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics summary"""
        all_metrics = {}
        
        for name in self.metrics_buffer:
            summary = await self.get_metric_summary(name)
            if summary:
                all_metrics[name] = summary
        
        return all_metrics

class DistributedTracer:
    """Distributed tracing system"""
    
    def __init__(self):
        self.active_traces: Dict[str, Trace] = {}
        self.completed_traces: List[Trace] = []
        self.trace_buffer_size = 10000
        self.initialized = False
    
    async def initialize(self):
        """Initialize distributed tracer"""
        self.initialized = True
        logger.info("✅ Distributed Tracer initialized")
    
    def start_trace(self, operation_name: str, trace_type: TraceType = TraceType.CUSTOM,
                   parent_trace_id: str = None, tags: Dict[str, str] = None) -> str:
        """Start a new trace"""
        if not self.initialized:
            return None
        
        try:
            trace_id = str(uuid.uuid4())
            span_id = str(uuid.uuid4())
            
            trace = Trace(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_trace_id,
                operation_name=operation_name,
                trace_type=trace_type,
                tags=tags or {}
            )
            
            self.active_traces[trace_id] = trace
            
            return trace_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start trace: {e}")
            return None
    
    def end_trace(self, trace_id: str, status: str = "success", 
                 tags: Dict[str, str] = None, logs: List[Dict[str, Any]] = None):
        """End a trace"""
        if not self.initialized or trace_id not in self.active_traces:
            return False
        
        try:
            trace = self.active_traces[trace_id]
            trace.end_time = datetime.now()
            trace.duration = (trace.end_time - trace.start_time).total_seconds()
            trace.status = status
            
            if tags:
                trace.tags.update(tags)
            
            if logs:
                trace.logs.extend(logs)
            
            # Move to completed traces
            self.completed_traces.append(trace)
            del self.active_traces[trace_id]
            
            # Maintain buffer size
            if len(self.completed_traces) > self.trace_buffer_size:
                self.completed_traces = self.completed_traces[-self.trace_buffer_size:]
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to end trace {trace_id}: {e}")
            return False
    
    def add_trace_log(self, trace_id: str, level: LogLevel, message: str, 
                     metadata: Dict[str, Any] = None):
        """Add log to trace"""
        if not self.initialized or trace_id not in self.active_traces:
            return False
        
        try:
            trace = self.active_traces[trace_id]
            log_entry = {
                'timestamp': datetime.now(),
                'level': level.value,
                'message': message,
                'metadata': metadata or {}
            }
            trace.logs.append(log_entry)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add log to trace {trace_id}: {e}")
            return False
    
    async def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace summary"""
        # Check active traces first
        if trace_id in self.active_traces:
            trace = self.active_traces[trace_id]
        else:
            # Check completed traces
            trace = next((t for t in self.completed_traces if t.trace_id == trace_id), None)
        
        if not trace:
            return None
        
        return {
            'trace_id': trace.trace_id,
            'span_id': trace.span_id,
            'parent_span_id': trace.parent_span_id,
            'operation_name': trace.operation_name,
            'start_time': trace.start_time.isoformat(),
            'end_time': trace.end_time.isoformat() if trace.end_time else None,
            'duration': trace.duration,
            'trace_type': trace.trace_type.value,
            'status': trace.status,
            'tags': trace.tags,
            'log_count': len(trace.logs)
        }
    
    async def get_traces_by_type(self, trace_type: TraceType, 
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get traces by type"""
        traces = [t for t in self.completed_traces if t.trace_type == trace_type]
        traces = traces[-limit:]  # Get most recent
        
        return [await self.get_trace_summary(t.trace_id) for t in traces]

class LogAggregator:
    """Advanced log aggregation system"""
    
    def __init__(self):
        self.log_buffer: deque = deque(maxlen=50000)
        self.log_levels: Dict[LogLevel, int] = {level: 0 for level in LogLevel}
        self.initialized = False
    
    async def initialize(self):
        """Initialize log aggregator"""
        self.initialized = True
        logger.info("✅ Log Aggregator initialized")
    
    async def log(self, level: LogLevel, message: str, source: str = "",
                 trace_id: str = None, span_id: str = None, 
                 tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Log a message"""
        if not self.initialized:
            return False
        
        try:
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                message=message,
                source=source,
                trace_id=trace_id,
                span_id=span_id,
                tags=tags or {},
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.log_buffer.append(log_entry)
            
            # Update level counter
            self.log_levels[level] += 1
            
            # Also log to standard logger
            getattr(logger, level.value)(f"[{source}] {message}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to log message: {e}")
            return False
    
    async def get_logs(self, level: LogLevel = None, source: str = None,
                      trace_id: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get logs with filters"""
        logs = list(self.log_buffer)
        
        # Apply filters
        if level:
            logs = [l for l in logs if l.level == level]
        
        if source:
            logs = [l for l in logs if l.source == source]
        
        if trace_id:
            logs = [l for l in logs if l.trace_id == trace_id]
        
        # Limit results
        logs = logs[-limit:]
        
        return [{
            'timestamp': l.timestamp.isoformat(),
            'level': l.level.value,
            'message': l.message,
            'source': l.source,
            'trace_id': l.trace_id,
            'span_id': l.span_id,
            'tags': l.tags,
            'metadata': l.metadata
        } for l in logs]
    
    async def get_log_statistics(self) -> Dict[str, Any]:
        """Get log statistics"""
        return {
            'total_logs': len(self.log_buffer),
            'log_levels': {level.value: count for level, count in self.log_levels.items()},
            'oldest_log': self.log_buffer[0].timestamp.isoformat() if self.log_buffer else None,
            'newest_log': self.log_buffer[-1].timestamp.isoformat() if self.log_buffer else None
        }

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.initialized = False
        self.monitoring_thread = None
        self.stop_monitoring = False
    
    async def initialize(self):
        """Initialize system monitor"""
        self.initialized = True
        logger.info("✅ System Monitor initialized")
    
    async def get_system_health(self) -> SystemHealth:
        """Get current system health"""
        if not self.initialized:
            return None
        
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # GPU usage (if available)
            gpu_usage = None
            gpu_memory = None
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_usage = gpu.load * 100
                    gpu_memory = gpu.memoryUtil * 100
            except:
                pass
            
            # Determine overall health
            overall_health = "healthy"
            if cpu_usage > 90 or memory_usage > 90 or disk_usage > 90:
                overall_health = "critical"
            elif cpu_usage > 80 or memory_usage > 80 or disk_usage > 80:
                overall_health = "warning"
            elif cpu_usage > 70 or memory_usage > 70 or disk_usage > 70:
                overall_health = "degraded"
            
            return SystemHealth(
                overall_health=overall_health,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get system health: {e}")
            return None
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        if not self.initialized:
            return
        
        self.stop_monitoring = False
        
        async def monitor_loop():
            while not self.stop_monitoring:
                try:
                    health = await self.get_system_health()
                    if health:
                        # Log system health
                        await self._log_system_health(health)
                    
                    await asyncio.sleep(interval)
                except Exception as e:
                    logger.error(f"❌ Monitoring loop error: {e}")
                    await asyncio.sleep(interval)
        
        asyncio.create_task(monitor_loop())
        logger.info(f"✅ System monitoring started with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.stop_monitoring = True
        logger.info("✅ System monitoring stopped")
    
    async def _log_system_health(self, health: SystemHealth):
        """Log system health metrics"""
        # This would typically send to metrics collector
        logger.info(f"System Health: {health.overall_health} - "
                   f"CPU: {health.cpu_usage:.1f}%, "
                   f"Memory: {health.memory_usage:.1f}%, "
                   f"Disk: {health.disk_usage:.1f}%")

class AlertManager:
    """Advanced alert management system"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize alert manager"""
        self.initialized = True
        logger.info("✅ Alert Manager initialized")
    
    async def create_alert(self, alert: Alert) -> bool:
        """Create a new alert"""
        try:
            self.alerts[alert.alert_id] = alert
            logger.info(f"✅ Alert created: {alert.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create alert: {e}")
            return False
    
    async def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check all alerts against current metrics"""
        if not self.initialized:
            return []
        
        triggered_alerts = []
        
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            # Check cooldown
            if alert.last_triggered and \
               (datetime.now() - alert.last_triggered).seconds < alert.cooldown:
                continue
            
            # Check condition
            if self._evaluate_condition(metrics, alert):
                await self._trigger_alert(alert, metrics)
                triggered_alerts.append({
                    'alert_id': alert.alert_id,
                    'name': alert.name,
                    'severity': alert.severity.value,
                    'condition': alert.condition,
                    'threshold': alert.threshold,
                    'timestamp': datetime.now().isoformat()
                })
        
        return triggered_alerts
    
    def _evaluate_condition(self, metrics: Dict[str, Any], alert: Alert) -> bool:
        """Evaluate alert condition"""
        try:
            # Parse condition (simplified)
            if '>' in alert.condition:
                metric_name, threshold = alert.condition.split('>')
                metric_name = metric_name.strip()
                threshold = float(threshold.strip())
                
                if metric_name in metrics:
                    return metrics[metric_name] > threshold
            
            elif '<' in alert.condition:
                metric_name, threshold = alert.condition.split('<')
                metric_name = metric_name.strip()
                threshold = float(threshold.strip())
                
                if metric_name in metrics:
                    return metrics[metric_name] < threshold
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate alert condition: {e}")
            return False
    
    async def _trigger_alert(self, alert: Alert, metrics: Dict[str, Any]):
        """Trigger an alert"""
        alert.last_triggered = datetime.now()
        
        # Log alert
        logger.warning(f"🚨 ALERT TRIGGERED: {alert.name} - {alert.condition}")
        
        # Add to history
        self.alert_history.append({
            'alert_id': alert.alert_id,
            'name': alert.name,
            'severity': alert.severity.value,
            'condition': alert.condition,
            'threshold': alert.threshold,
            'triggered_at': alert.last_triggered.isoformat(),
            'metrics': metrics
        })
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

class AdvancedMonitoringSystem:
    """Main monitoring and observability system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.distributed_tracer = DistributedTracer()
        self.log_aggregator = LogAggregator()
        self.system_monitor = SystemMonitor()
        self.alert_manager = AlertManager()
        self.initialized = False
    
    async def initialize(self):
        """Initialize monitoring system"""
        try:
            logger.info("📊 Initializing Advanced Monitoring & Observability System...")
            
            # Initialize components
            await self.metrics_collector.initialize()
            await self.distributed_tracer.initialize()
            await self.log_aggregator.initialize()
            await self.system_monitor.initialize()
            await self.alert_manager.initialize()
            
            # Register default metrics
            await self._register_default_metrics()
            
            # Create default alerts
            await self._create_default_alerts()
            
            # Start system monitoring
            await self.system_monitor.start_monitoring()
            
            self.initialized = True
            logger.info("✅ Advanced Monitoring & Observability System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring system: {e}")
            raise
    
    async def _register_default_metrics(self):
        """Register default metrics"""
        default_metrics = [
            ("cpu_usage", MetricType.GAUGE, "CPU usage percentage"),
            ("memory_usage", MetricType.GAUGE, "Memory usage percentage"),
            ("disk_usage", MetricType.GAUGE, "Disk usage percentage"),
            ("response_time", MetricType.HISTOGRAM, "Response time in milliseconds"),
            ("error_rate", MetricType.GAUGE, "Error rate percentage"),
            ("active_connections", MetricType.GAUGE, "Number of active connections"),
            ("requests_per_second", MetricType.COUNTER, "Requests per second"),
            ("ai_inference_time", MetricType.HISTOGRAM, "AI inference time in milliseconds"),
            ("ai_training_time", MetricType.HISTOGRAM, "AI training time in seconds"),
            ("gpu_usage", MetricType.GAUGE, "GPU usage percentage")
        ]
        
        for name, metric_type, description in default_metrics:
            await self.metrics_collector.register_metric(name, metric_type, description)
    
    async def _create_default_alerts(self):
        """Create default alerts"""
        default_alerts = [
            Alert(
                alert_id="high_cpu_usage",
                name="High CPU Usage",
                condition="cpu_usage > 80",
                threshold=80.0,
                severity=AlertSeverity.WARNING
            ),
            Alert(
                alert_id="high_memory_usage",
                name="High Memory Usage",
                condition="memory_usage > 85",
                threshold=85.0,
                severity=AlertSeverity.WARNING
            ),
            Alert(
                alert_id="high_disk_usage",
                name="High Disk Usage",
                condition="disk_usage > 90",
                threshold=90.0,
                severity=AlertSeverity.ERROR
            ),
            Alert(
                alert_id="high_error_rate",
                name="High Error Rate",
                condition="error_rate > 5",
                threshold=5.0,
                severity=AlertSeverity.CRITICAL
            ),
            Alert(
                alert_id="slow_response_time",
                name="Slow Response Time",
                condition="response_time > 1000",
                threshold=1000.0,
                severity=AlertSeverity.WARNING
            )
        ]
        
        for alert in default_alerts:
            await self.alert_manager.create_alert(alert)
    
    async def record_metric(self, name: str, value: float, 
                          labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a metric"""
        if not self.initialized:
            return False
        
        return await self.metrics_collector.record_metric(name, value, labels, metadata)
    
    def start_trace(self, operation_name: str, trace_type: TraceType = TraceType.CUSTOM,
                   parent_trace_id: str = None, tags: Dict[str, str] = None) -> str:
        """Start a new trace"""
        if not self.initialized:
            return None
        
        return self.distributed_tracer.start_trace(operation_name, trace_type, parent_trace_id, tags)
    
    def end_trace(self, trace_id: str, status: str = "success", 
                 tags: Dict[str, str] = None, logs: List[Dict[str, Any]] = None):
        """End a trace"""
        if not self.initialized:
            return False
        
        return self.distributed_tracer.end_trace(trace_id, status, tags, logs)
    
    async def log(self, level: LogLevel, message: str, source: str = "",
                 trace_id: str = None, span_id: str = None, 
                 tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Log a message"""
        if not self.initialized:
            return False
        
        return await self.log_aggregator.log(level, message, source, trace_id, span_id, tags, metadata)
    
    async def get_system_health(self) -> SystemHealth:
        """Get current system health"""
        if not self.initialized:
            return None
        
        return await self.system_monitor.get_system_health()
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.initialized:
            return {}
        
        return await self.metrics_collector.get_all_metrics()
    
    async def get_logs(self, level: LogLevel = None, source: str = None,
                      trace_id: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get logs with filters"""
        if not self.initialized:
            return []
        
        return await self.log_aggregator.get_logs(level, source, trace_id, limit)
    
    async def get_traces(self, trace_type: TraceType = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get traces"""
        if not self.initialized:
            return []
        
        if trace_type:
            return await self.distributed_tracer.get_traces_by_type(trace_type, limit)
        else:
            # Get all traces
            all_traces = []
            for t_type in TraceType:
                traces = await self.distributed_tracer.get_traces_by_type(t_type, limit)
                all_traces.extend(traces)
            return all_traces[-limit:]
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        if not self.initialized:
            return {}
        
        try:
            # Get system health
            health = await self.get_system_health()
            
            # Get metrics summary
            metrics = await self.get_metrics_summary()
            
            # Get recent logs
            logs = await self.get_logs(limit=100)
            
            # Get recent traces
            traces = await self.get_traces(limit=50)
            
            # Get log statistics
            log_stats = await self.log_aggregator.get_log_statistics()
            
            return {
                'system_health': health.__dict__ if health else {},
                'metrics': metrics,
                'logs': logs,
                'traces': traces,
                'log_statistics': log_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get dashboard data: {e}")
            return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'initialized': self.initialized,
            'metrics_collector_ready': self.metrics_collector.initialized,
            'distributed_tracer_ready': self.distributed_tracer.initialized,
            'log_aggregator_ready': self.log_aggregator.initialized,
            'system_monitor_ready': self.system_monitor.initialized,
            'alert_manager_ready': self.alert_manager.initialized,
            'timestamp': datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown monitoring system"""
        if self.system_monitor.initialized:
            await self.system_monitor.stop_monitoring()
        
        self.initialized = False
        logger.info("✅ Advanced Monitoring & Observability System shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the advanced monitoring system"""
    print("📊 HeyGen AI - Advanced Monitoring & Observability Demo")
    print("=" * 70)
    
    # Initialize system
    monitoring = AdvancedMonitoringSystem()
    
    try:
        # Initialize the system
        print("\n🚀 Initializing Advanced Monitoring & Observability System...")
        await monitoring.initialize()
        print("✅ Advanced Monitoring & Observability System initialized successfully")
        
        # Get system status
        print("\n📊 System Status:")
        status = await monitoring.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Record some metrics
        print("\n📈 Recording Metrics...")
        
        for i in range(20):
            await monitoring.record_metric("cpu_usage", np.random.uniform(20, 80))
            await monitoring.record_metric("memory_usage", np.random.uniform(30, 70))
            await monitoring.record_metric("response_time", np.random.exponential(100))
            await monitoring.record_metric("requests_per_second", np.random.poisson(50))
            
            # Add some errors
            if i % 5 == 0:
                await monitoring.record_metric("error_rate", np.random.uniform(0, 10))
            else:
                await monitoring.record_metric("error_rate", np.random.uniform(0, 2))
        
        print("  ✅ Recorded 80 metrics")
        
        # Create some traces
        print("\n🔍 Creating Traces...")
        
        trace_ids = []
        for i in range(5):
            trace_id = monitoring.start_trace(f"ai_inference_{i}", TraceType.AI_INFERENCE)
            if trace_id:
                trace_ids.append(trace_id)
                
                # Simulate some work
                await asyncio.sleep(0.1)
                
                # Add some logs
                await monitoring.log(LogLevel.INFO, f"Starting AI inference {i}", "ai_engine", trace_id)
                await monitoring.log(LogLevel.DEBUG, f"Processing data for inference {i}", "ai_engine", trace_id)
                
                # End trace
                monitoring.end_trace(trace_id, "success")
        
        print(f"  ✅ Created {len(trace_ids)} traces")
        
        # Log some messages
        print("\n📝 Logging Messages...")
        
        for i in range(10):
            level = np.random.choice([LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR])
            await monitoring.log(level, f"Test message {i}", "demo")
        
        print("  ✅ Logged 10 messages")
        
        # Get system health
        print("\n🏥 System Health:")
        health = await monitoring.get_system_health()
        if health:
            print(f"  Overall Health: {health.overall_health}")
            print(f"  CPU Usage: {health.cpu_usage:.1f}%")
            print(f"  Memory Usage: {health.memory_usage:.1f}%")
            print(f"  Disk Usage: {health.disk_usage:.1f}%")
            if health.gpu_usage:
                print(f"  GPU Usage: {health.gpu_usage:.1f}%")
                print(f"  GPU Memory: {health.gpu_memory:.1f}%")
        
        # Get metrics summary
        print("\n📊 Metrics Summary:")
        metrics = await monitoring.get_metrics_summary()
        for name, summary in metrics.items():
            print(f"  {name}:")
            print(f"    Count: {summary['count']}")
            print(f"    Mean: {summary['mean']:.2f}")
            print(f"    Min: {summary['min']:.2f}")
            print(f"    Max: {summary['max']:.2f}")
            print(f"    P95: {summary['p95']:.2f}")
        
        # Get recent logs
        print("\n📝 Recent Logs:")
        logs = await monitoring.get_logs(limit=5)
        for log in logs:
            print(f"  [{log['level']}] {log['message']}")
        
        # Get recent traces
        print("\n🔍 Recent Traces:")
        traces = await monitoring.get_traces(limit=3)
        for trace in traces:
            print(f"  {trace['operation_name']}: {trace['duration']:.3f}s ({trace['status']})")
        
        # Get dashboard data
        print("\n📋 Dashboard Data:")
        dashboard_data = await monitoring.get_dashboard_data()
        print(f"  System Health: {dashboard_data.get('system_health', {}).get('overall_health', 'unknown')}")
        print(f"  Metrics Count: {len(dashboard_data.get('metrics', {}))}")
        print(f"  Logs Count: {len(dashboard_data.get('logs', []))}")
        print(f"  Traces Count: {len(dashboard_data.get('traces', []))}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Shutdown
        await monitoring.shutdown()
        print("\n✅ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())


