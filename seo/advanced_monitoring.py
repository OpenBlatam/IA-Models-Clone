#!/usr/bin/env python3
"""
Advanced Monitoring and Observability System
Real-time metrics, profiling, and intelligent alerting
"""

import time
import threading
import asyncio
import psutil
import GPUtil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import tracemalloc
import line_profiler
import cProfile
import pstats
import io
import warnings

warnings.filterwarnings("ignore")

@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'tags': self.tags
        }

@dataclass
class MetricSeries:
    """Time series of metric data points."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_point(self, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Add a new metric point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        self.points.append(point)
    
    def get_recent(self, seconds: int = 300) -> List[MetricPoint]:
        """Get recent metric points within specified time window."""
        cutoff = time.time() - seconds
        return [p for p in self.points if p.timestamp >= cutoff]
    
    def get_statistics(self, seconds: int = 300) -> Dict[str, float]:
        """Get statistical summary of recent metrics."""
        recent = self.get_recent(seconds)
        if not recent:
            return {}
        
        values = [p.value for p in recent]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = [p.to_dict() for p in self.points]
        df = pd.DataFrame(data)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

class MetricsCollector:
    """High-performance metrics collection system."""
    
    def __init__(self, max_series: int = 1000):
        self.metrics: Dict[str, MetricSeries] = {}
        self.max_series = max_series
        self._lock = threading.RLock()
        self._collectors: List[Callable] = []
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None
    
    def register_metric(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a new metric series."""
        with self._lock:
            if len(self.metrics) >= self.max_series:
                # Remove oldest metric if at capacity
                oldest = min(self.metrics.keys(), key=lambda k: self.metrics[k].points[0].timestamp if self.metrics[k].points else 0)
                del self.metrics[oldest]
            
            self.metrics[name] = MetricSeries(name, metadata=metadata or {})
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self._lock:
            if name not in self.metrics:
                self.register_metric(name)
            
            self.metrics[name].add_point(value, tags)
    
    def record_batch(self, metrics: Dict[str, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Record multiple metrics at once."""
        with self._lock:
            for name, value in metrics.items():
                self.record(name, value, tags)
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get a metric series by name."""
        with self._lock:
            return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """Get all registered metrics."""
        with self._lock:
            return self.metrics.copy()
    
    def add_collector(self, collector: Callable) -> None:
        """Add a custom metrics collector function."""
        self._collectors.append(collector)
    
    def start_collection(self, interval: float = 1.0) -> None:
        """Start automatic metrics collection."""
        if self._running:
            return
        
        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval,),
            daemon=True
        )
        self._collection_thread.start()
    
    def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join()
    
    def _collection_loop(self, interval: float) -> None:
        """Main collection loop."""
        while self._running:
            try:
                for collector in self._collectors:
                    try:
                        collector(self)
                    except Exception as e:
                        logging.error(f"Metrics collector failed: {e}")
                
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Metrics collection loop failed: {e}")
                time.sleep(interval)

class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        self.last_cpu_times = psutil.cpu_times()
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()
    
    def collect_system_metrics(self, collector: MetricsCollector) -> None:
        """Collect system-level metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_times = psutil.cpu_times()
        cpu_freq = psutil.cpu_freq()
        
        collector.record('system.cpu.usage_percent', cpu_percent)
        collector.record('system.cpu.user_time', cpu_times.user)
        collector.record('system.cpu.system_time', cpu_times.system)
        collector.record('system.cpu.idle_time', cpu_times.idle)
        
        if cpu_freq:
            collector.record('system.cpu.frequency_mhz', cpu_freq.current)
            collector.record('system.cpu.frequency_min_mhz', cpu_freq.min)
            collector.record('system.cpu.frequency_max_mhz', cpu_freq.max)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        collector.record('system.memory.total_gb', memory.total / (1024**3))
        collector.record('system.memory.available_gb', memory.available / (1024**3))
        collector.record('system.memory.used_gb', memory.used / (1024**3))
        collector.record('system.memory.usage_percent', memory.percent)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        collector.record('system.disk.total_gb', disk.total / (1024**3))
        collector.record('system.disk.used_gb', disk.used / (1024**3))
        collector.record('system.disk.free_gb', disk.free / (1024**3))
        collector.record('system.disk.usage_percent', (disk.used / disk.total) * 100)
        
        # Network metrics
        net_io = psutil.net_io_counters()
        collector.record('system.network.bytes_sent', net_io.bytes_sent)
        collector.record('system.network.bytes_recv', net_io.bytes_recv)
        collector.record('system.network.packets_sent', net_io.packets_sent)
        collector.record('system.network.packets_recv', net_io.packets_recv)
        
        # GPU metrics (if available)
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                collector.record(f'system.gpu.{i}.memory_used_mb', gpu.memoryUsed)
                collector.record(f'system.gpu.{i}.memory_total_mb', gpu.memoryTotal)
                collector.record(f'system.gpu.{i}.memory_utilization_percent', gpu.memoryUtil * 100)
                collector.record(f'system.gpu.{i}.gpu_utilization_percent', gpu.load * 100)
                collector.record(f'system.gpu.{i}.temperature_celsius', gpu.temperature)
        except:
            pass  # GPU monitoring not available
    
    def collect_process_metrics(self, collector: MetricsCollector, pid: Optional[int] = None) -> None:
        """Collect process-specific metrics."""
        process = psutil.Process(pid) if pid else psutil.Process()
        
        try:
            # Process CPU and memory
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            collector.record('process.cpu.usage_percent', cpu_percent)
            collector.record('process.memory.rss_mb', memory_info.rss / (1024**2))
            collector.record('process.memory.vms_mb', memory_info.vms / (1024**2))
            collector.record('process.memory.usage_percent', memory_percent)
            
            # Process threads and file handles
            collector.record('process.threads.count', process.num_threads())
            collector.record('process.files.count', len(process.open_files()))
            collector.record('process.connections.count', len(process.connections()))
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

class PerformanceProfiler:
    """Advanced performance profiling and analysis."""
    
    def __init__(self):
        self.profilers: Dict[str, Any] = {}
        self.traces: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()
    
    @contextmanager
    def profile_function(self, name: str, enable_line_profiler: bool = False):
        """Context manager for profiling a function."""
        if enable_line_profiler:
            profiler = line_profiler.LineProfiler()
            self.profilers[name] = profiler
            try:
                yield profiler
            finally:
                if name in self.profilers:
                    del self.profilers[name]
        else:
            profiler = cProfile.Profile()
            self.profilers[name] = profiler
            try:
                profiler.enable()
                yield profiler
            finally:
                profiler.disable()
                if name in self.profilers:
                    del self.profilers[name]
    
    @contextmanager
    def memory_trace(self, name: str):
        """Context manager for memory tracing."""
        tracemalloc.start()
        try:
            yield
        finally:
            current, peak = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            
            with self._lock:
                self.traces[name] = {
                    'current_mb': current / (1024**2),
                    'peak_mb': peak / (1024**2),
                    'snapshot': snapshot
                }
    
    def get_profile_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get profiling statistics for a named profile."""
        if name not in self.profilers:
            return None
        
        profiler = self.profilers[name]
        
        if isinstance(profiler, line_profiler.LineProfiler):
            # Line profiler stats
            stats = profiler.get_stats()
            return {
                'type': 'line_profiler',
                'timings': stats.timings,
                'unit': stats.unit
            }
        else:
            # cProfile stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            return {
                'type': 'cprofile',
                'stats': s.getvalue()
            }
    
    def get_memory_trace(self, name: str) -> Optional[Dict[str, Any]]:
        """Get memory trace for a named trace."""
        with self._lock:
            return self.traces.get(name)

class AlertManager:
    """Intelligent alerting system with rule-based triggers."""
    
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self.alerts: deque = deque(maxlen=1000)
        self.handlers: List[Callable] = []
        self._lock = threading.RLock()
    
    def add_rule(self, name: str, condition: Callable, severity: str = 'warning', 
                 cooldown: float = 300) -> None:
        """Add an alert rule."""
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'cooldown': cooldown,
            'last_triggered': 0
        }
        
        with self._lock:
            self.rules.append(rule)
    
    def add_handler(self, handler: Callable) -> None:
        """Add an alert handler."""
        with self._lock:
            self.handlers.append(handler)
    
    def check_alerts(self, metrics: MetricsCollector) -> None:
        """Check all alert rules against current metrics."""
        current_time = time.time()
        
        with self._lock:
            for rule in self.rules:
                # Check cooldown
                if current_time - rule['last_triggered'] < rule['cooldown']:
                    continue
                
                try:
                    if rule['condition'](metrics):
                        alert = {
                            'timestamp': current_time,
                            'rule_name': rule['name'],
                            'severity': rule['severity'],
                            'message': f"Alert triggered: {rule['name']}"
                        }
                        
                        self.alerts.append(alert)
                        rule['last_triggered'] = current_time
                        
                        # Notify handlers
                        for handler in self.handlers:
                            try:
                                handler(alert)
                            except Exception as e:
                                logging.error(f"Alert handler failed: {e}")
                
                except Exception as e:
                    logging.error(f"Alert rule check failed: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts within specified time window."""
        cutoff = time.time() - (hours * 3600)
        
        with self._lock:
            return [a for a in self.alerts if a['timestamp'] >= cutoff]

class MetricsVisualizer:
    """Real-time metrics visualization and analysis."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time System Metrics', fontsize=16)
        
        # Set up subplots
        self.axes[0, 0].set_title('CPU Usage')
        self.axes[0, 1].set_title('Memory Usage')
        self.axes[1, 0].set_title('Network I/O')
        self.axes[1, 1].set_title('Disk Usage')
        
        plt.ion()  # Enable interactive mode
    
    def update_plots(self) -> None:
        """Update all metric plots."""
        try:
            # CPU Usage
            cpu_metric = self.metrics.get_metric('system.cpu.usage_percent')
            if cpu_metric:
                cpu_data = cpu_metric.to_dataframe()
                if not cpu_data.empty:
                    self.axes[0, 0].clear()
                    self.axes[0, 0].plot(cpu_data['datetime'], cpu_data['value'])
                    self.axes[0, 0].set_title('CPU Usage (%)')
                    self.axes[0, 0].set_ylim(0, 100)
                    self.axes[0, 0].grid(True)
            
            # Memory Usage
            mem_metric = self.metrics.get_metric('system.memory.usage_percent')
            if mem_metric:
                mem_data = mem_metric.to_dataframe()
                if not mem_data.empty:
                    self.axes[0, 1].clear()
                    self.axes[0, 1].plot(mem_data['datetime'], mem_data['value'])
                    self.axes[0, 1].set_title('Memory Usage (%)')
                    self.axes[0, 1].set_ylim(0, 100)
                    self.axes[0, 1].grid(True)
            
            # Network I/O
            net_sent = self.metrics.get_metric('system.network.bytes_sent')
            net_recv = self.metrics.get_metric('system.network.bytes_recv')
            if net_sent and net_recv:
                sent_data = net_sent.to_dataframe()
                recv_data = net_recv.to_dataframe()
                if not sent_data.empty and not recv_data.empty:
                    self.axes[1, 0].clear()
                    self.axes[1, 0].plot(sent_data['datetime'], sent_data['value'], label='Sent')
                    self.axes[1, 0].plot(recv_data['datetime'], recv_data['value'], label='Received')
                    self.axes[1, 0].set_title('Network I/O (bytes)')
                    self.axes[1, 0].legend()
                    self.axes[1, 0].grid(True)
            
            # Disk Usage
            disk_metric = self.metrics.get_metric('system.disk.usage_percent')
            if disk_metric:
                disk_data = disk_metric.to_dataframe()
                if not disk_data.empty:
                    self.axes[1, 1].clear()
                    self.axes[1, 1].plot(disk_data['datetime'], disk_data['value'])
                    self.axes[1, 1].set_title('Disk Usage (%)')
                    self.axes[1, 1].set_ylim(0, 100)
                    self.axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.pause(0.01)
            
        except Exception as e:
            logging.error(f"Plot update failed: {e}")
    
    def save_plots(self, filename: str) -> None:
        """Save current plots to file."""
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        except Exception as e:
            logging.error(f"Failed to save plots: {e}")
    
    def close(self) -> None:
        """Close the visualization."""
        plt.ioff()
        plt.close(self.fig)

class MonitoringSystem:
    """Complete monitoring system orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = MetricsCollector()
        self.system_monitor = SystemMonitor()
        self.profiler = PerformanceProfiler()
        self.alert_manager = AlertManager()
        self.visualizer: Optional[MetricsVisualizer] = None
        
        # Set up automatic collection
        self.metrics.add_collector(self.system_monitor.collect_system_metrics)
        self.metrics.add_collector(self.system_monitor.collect_process_metrics)
        
        # Set up default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self) -> None:
        """Set up default system alert rules."""
        # High CPU usage
        self.alert_manager.add_rule(
            'high_cpu_usage',
            lambda m: m.get_metric('system.cpu.usage_percent') and 
                     m.get_metric('system.cpu.usage_percent').get_statistics(60)['mean'] > 90,
            'warning',
            300
        )
        
        # High memory usage
        self.alert_manager.add_rule(
            'high_memory_usage',
            lambda m: m.get_metric('system.memory.usage_percent') and 
                     m.get_metric('system.memory.usage_percent').get_statistics(60)['mean'] > 85,
            'warning',
            300
        )
        
        # High disk usage
        self.alert_manager.add_rule(
            'high_disk_usage',
            lambda m: m.get_metric('system.disk.usage_percent') and 
                     m.get_metric('system.disk.usage_percent').get_statistics(300)['mean'] > 90,
            'critical',
            600
        )
    
    def start(self, collection_interval: float = 1.0, enable_visualization: bool = False) -> None:
        """Start the monitoring system."""
        self.metrics.start_collection(collection_interval)
        
        if enable_visualization:
            self.visualizer = MetricsVisualizer(self.metrics)
        
        # Start alert checking in background
        threading.Thread(
            target=self._alert_check_loop,
            daemon=True
        ).start()
    
    def stop(self) -> None:
        """Stop the monitoring system."""
        self.metrics.stop_collection()
        if self.visualizer:
            self.visualizer.close()
    
    def _alert_check_loop(self) -> None:
        """Background loop for checking alerts."""
        while True:
            try:
                self.alert_manager.check_alerts(self.metrics)
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logging.error(f"Alert check loop failed: {e}")
                time.sleep(5)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        health = {
            'status': 'healthy',
            'metrics': {},
            'alerts': [],
            'timestamp': time.time()
        }
        
        # Check key metrics
        cpu_metric = self.metrics.get_metric('system.cpu.usage_percent')
        if cpu_metric:
            cpu_stats = cpu_metric.get_statistics(300)
            health['metrics']['cpu'] = cpu_stats
            if cpu_stats.get('mean', 0) > 80:
                health['status'] = 'warning'
        
        mem_metric = self.metrics.get_metric('system.memory.usage_percent')
        if mem_metric:
            mem_stats = mem_metric.get_statistics(300)
            health['metrics']['memory'] = mem_stats
            if mem_stats.get('mean', 0) > 85:
                health['status'] = 'warning'
        
        # Get recent alerts
        health['alerts'] = self.alert_manager.get_recent_alerts(1)  # Last hour
        
        if any(a['severity'] == 'critical' for a in health['alerts']):
            health['status'] = 'critical'
        
        return health
    
    def export_metrics(self, filename: str, format: str = 'json') -> None:
        """Export metrics to file."""
        try:
            all_metrics = self.metrics.get_all_metrics()
            export_data = {}
            
            for name, series in all_metrics.items():
                export_data[name] = {
                    'metadata': series.metadata,
                    'data': [p.to_dict() for p in series.points]
                }
            
            if format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == 'pickle':
                with open(filename, 'wb') as f:
                    pickle.dump(export_data, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logging.error(f"Failed to export metrics: {e}")

# Convenience functions
def create_monitoring_system(config: Optional[Dict[str, Any]] = None) -> MonitoringSystem:
    """Create and configure a monitoring system."""
    return MonitoringSystem(config)

def start_monitoring(collection_interval: float = 1.0, enable_visualization: bool = False) -> MonitoringSystem:
    """Quick start monitoring with default settings."""
    monitoring = create_monitoring_system()
    monitoring.start(collection_interval, enable_visualization)
    return monitoring


