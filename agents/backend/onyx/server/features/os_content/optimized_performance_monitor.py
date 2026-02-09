from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import psutil
import numpy as np
from collections import deque, defaultdict
import json
import sqlite3
from pathlib import Path
import gc
import os
import signal
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest

from typing import Any, List, Dict, Optional
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float
    acknowledged: bool = False

@dataclass
class PerformanceConfig:
    collection_interval: float = 1.0  # 1 second
    retention_period: int = 86400  # 24 hours
    max_data_points: int = 10000
    enable_prometheus: bool = True
    enable_alerting: bool = True
    enable_storage: bool = True
    storage_path: str = "performance_data"
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)

class TimeSeriesData:
    """Thread-safe time series data structure"""
    
    def __init__(self, max_points: int = 10000):
        
    """__init__ function."""
self.max_points = max_points
        self.data = deque(maxlen=max_points)
        self.lock = threading.RLock()
    
    def add_point(self, timestamp: float, value: float, labels: Dict[str, str] = None):
        
    """add_point function."""
with self.lock:
            self.data.append({
                'timestamp': timestamp,
                'value': value,
                'labels': labels or {}
            })
    
    def get_data(self, start_time: float = None, end_time: float = None) -> List[Dict]:
        with self.lock:
            if start_time is None and end_time is None:
                return list(self.data)
            
            filtered_data = []
            for point in self.data:
                if start_time and point['timestamp'] < start_time:
                    continue
                if end_time and point['timestamp'] > end_time:
                    continue
                filtered_data.append(point)
            
            return filtered_data
    
    def get_latest(self) -> Optional[Dict]:
        with self.lock:
            return self.data[-1] if self.data else None
    
    def get_statistics(self) -> Dict[str, float]:
        with self.lock:
            if not self.data:
                return {}
            
            values = [point['value'] for point in self.data]
            return {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'count': len(values)
            }

class OptimizedPerformanceMonitor:
    def __init__(self, config: PerformanceConfig = None):
        
    """__init__ function."""
self.config = config or PerformanceConfig()
        
        # Data storage
        self.metrics: Dict[str, TimeSeriesData] = defaultdict(
            lambda: TimeSeriesData(self.config.max_data_points)
        )
        
        # Alerts
        self.alerts: List[Alert] = []
        self.alert_lock = threading.RLock()
        
        # Prometheus metrics
        self.prometheus_metrics = {}
        if self.config.enable_prometheus:
            self._init_prometheus_metrics()
        
        # Storage
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Monitoring tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Shutdown flag
        self._shutdown = False
        
        logger.info("OptimizedPerformanceMonitor initialized")

    def _init_prometheus_metrics(self) -> Any:
        """Initialize Prometheus metrics"""
        self.prometheus_metrics = {
            'cpu_usage': Gauge('cpu_usage_percent', 'CPU usage percentage'),
            'memory_usage': Gauge('memory_usage_percent', 'Memory usage percentage'),
            'disk_usage': Gauge('disk_usage_percent', 'Disk usage percentage'),
            'network_io': Counter('network_io_bytes', 'Network I/O bytes'),
            'process_count': Gauge('process_count', 'Number of processes'),
            'response_time': Histogram('response_time_seconds', 'Response time in seconds'),
            'error_rate': Counter('error_count', 'Error count'),
            'throughput': Summary('throughput_ops', 'Operations per second')
        }

    async def start(self) -> Any:
        """Start performance monitoring"""
        try:
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitor_loop())
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Performance monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting performance monitor: {e}")
            raise

    async def stop(self) -> Any:
        """Stop performance monitoring"""
        try:
            self._shutdown = True
            
            # Cancel monitoring tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            
            # Wait for tasks to finish
            if self.monitoring_task or self.cleanup_task:
                await asyncio.gather(
                    self.monitoring_task, self.cleanup_task,
                    return_exceptions=True
                )
            
            # Save final data
            if self.config.enable_storage:
                await self._save_data()
            
            logger.info("Performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping performance monitor: {e}")

    async def _monitor_loop(self) -> Any:
        """Main monitoring loop"""
        while not self._shutdown:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
                # Check alerts
                if self.config.enable_alerting:
                    await self._check_alerts()
                
                # Update Prometheus metrics
                if self.config.enable_prometheus:
                    await self._update_prometheus_metrics()
                
                await asyncio.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.collection_interval)

    async def _collect_system_metrics(self) -> Any:
        """Collect system-level metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        self._add_metric('system.cpu.usage', cpu_percent, timestamp)
        self._add_metric('system.cpu.count', cpu_count, timestamp)
        if cpu_freq:
            self._add_metric('system.cpu.frequency', cpu_freq.current, timestamp)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric('system.memory.usage', memory.percent, timestamp)
        self._add_metric('system.memory.available', memory.available, timestamp)
        self._add_metric('system.memory.total', memory.total, timestamp)
        self._add_metric('system.memory.used', memory.used, timestamp)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self._add_metric('system.disk.usage', (disk.used / disk.total) * 100, timestamp)
        self._add_metric('system.disk.available', disk.free, timestamp)
        self._add_metric('system.disk.total', disk.total, timestamp)
        
        # Network metrics
        network = psutil.net_io_counters()
        self._add_metric('system.network.bytes_sent', network.bytes_sent, timestamp)
        self._add_metric('system.network.bytes_recv', network.bytes_recv, timestamp)
        self._add_metric('system.network.packets_sent', network.packets_sent, timestamp)
        self._add_metric('system.network.packets_recv', network.packets_recv, timestamp)
        
        # Process metrics
        process_count = len(psutil.pids())
        self._add_metric('system.processes.count', process_count, timestamp)
        
        # Load average (Unix-like systems)
        try:
            load_avg = os.getloadavg()
            self._add_metric('system.load.1min', load_avg[0], timestamp)
            self._add_metric('system.load.5min', load_avg[1], timestamp)
            self._add_metric('system.load.15min', load_avg[2], timestamp)
        except AttributeError:
            pass  # Windows doesn't have load average

    async def _collect_application_metrics(self) -> Any:
        """Collect application-specific metrics"""
        timestamp = time.time()
        
        # Current process metrics
        process = psutil.Process()
        
        # Process CPU and memory
        try:
            process_cpu = process.cpu_percent()
            process_memory = process.memory_info()
            process_memory_percent = process.memory_percent()
            
            self._add_metric('application.process.cpu', process_cpu, timestamp)
            self._add_metric('application.process.memory.rss', process_memory.rss, timestamp)
            self._add_metric('application.process.memory.vms', process_memory.vms, timestamp)
            self._add_metric('application.process.memory.percent', process_memory_percent, timestamp)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # Garbage collection metrics
        gc_stats = gc.get_stats()
        for gen in gc_stats:
            self._add_metric(f'application.gc.{gen["generation"]}.collections', 
                           gen['collections'], timestamp)
            self._add_metric(f'application.gc.{gen["generation"]}.collected', 
                           gen['collected'], timestamp)
        
        # Thread count
        thread_count = process.num_threads()
        self._add_metric('application.threads.count', thread_count, timestamp)
        
        # File descriptors (Unix-like systems)
        try:
            fd_count = process.num_fds()
            self._add_metric('application.files.descriptors', fd_count, timestamp)
        except (psutil.NoSuchProcess, AttributeError):
            pass

    def _add_metric(self, name: str, value: float, timestamp: float, labels: Dict[str, str] = None):
        """Add a metric to the time series data"""
        self.metrics[name].add_point(timestamp, value, labels)

    async def _check_alerts(self) -> Any:
        """Check metrics against alert thresholds"""
        current_time = time.time()
        
        for metric_name, thresholds in self.config.alert_thresholds.items():
            if metric_name not in self.metrics:
                continue
            
            latest_data = self.metrics[metric_name].get_latest()
            if not latest_data:
                continue
            
            current_value = latest_data['value']
            
            for level, threshold in thresholds.items():
                alert_level = AlertLevel(level.upper())
                
                # Check if threshold is exceeded
                if self._is_threshold_exceeded(current_value, threshold):
                    # Check if alert already exists
                    existing_alert = self._get_existing_alert(metric_name, alert_level)
                    
                    if not existing_alert:
                        # Create new alert
                        alert = Alert(
                            id=f"{metric_name}_{alert_level.value}_{int(current_time)}",
                            level=alert_level,
                            message=f"{metric_name} exceeded {alert_level.value} threshold: {current_value} > {threshold}",
                            metric_name=metric_name,
                            threshold=threshold,
                            current_value=current_value,
                            timestamp=current_time
                        )
                        
                        self._add_alert(alert)
                        logger.warning(f"Alert triggered: {alert.message}")

    def _is_threshold_exceeded(self, value: float, threshold: float) -> bool:
        """Check if a value exceeds a threshold"""
        return value > threshold

    def _get_existing_alert(self, metric_name: str, level: AlertLevel) -> Optional[Alert]:
        """Get existing alert for metric and level"""
        with self.alert_lock:
            for alert in self.alerts:
                if (alert.metric_name == metric_name and 
                    alert.level == level and 
                    not alert.acknowledged):
                    return alert
        return None

    def _add_alert(self, alert: Alert):
        """Add a new alert"""
        with self.alert_lock:
            self.alerts.append(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    async def _update_prometheus_metrics(self) -> Any:
        """Update Prometheus metrics"""
        try:
            # Update system metrics
            cpu_data = self.metrics['system.cpu.usage'].get_latest()
            if cpu_data:
                self.prometheus_metrics['cpu_usage'].set(cpu_data['value'])
            
            memory_data = self.metrics['system.memory.usage'].get_latest()
            if memory_data:
                self.prometheus_metrics['memory_usage'].set(memory_data['value'])
            
            disk_data = self.metrics['system.disk.usage'].get_latest()
            if disk_data:
                self.prometheus_metrics['disk_usage'].set(disk_data['value'])
            
            process_data = self.metrics['system.processes.count'].get_latest()
            if process_data:
                self.prometheus_metrics['process_count'].set(process_data['value'])
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")

    async def _cleanup_loop(self) -> Any:
        """Cleanup old data"""
        while not self._shutdown:
            try:
                current_time = time.time()
                cutoff_time = current_time - self.config.retention_period
                
                # Clean up old metrics
                for metric_name, time_series in self.metrics.items():
                    old_data = time_series.get_data(end_time=cutoff_time)
                    # TimeSeriesData uses deque with maxlen, so old data is automatically removed
                
                # Clean up old alerts
                with self.alert_lock:
                    self.alerts = [
                        alert for alert in self.alerts
                        if alert.timestamp > cutoff_time or not alert.acknowledged
                    ]
                
                # Save data periodically
                if self.config.enable_storage:
                    await self._save_data()
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)

    async def _save_data(self) -> Any:
        """Save metrics data to storage"""
        try:
            # Save metrics to SQLite
            db_path = self.storage_path / "metrics.db"
            
            async with asyncio.Lock():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Create tables if they don't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp REAL NOT NULL,
                        labels TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        threshold REAL NOT NULL,
                        current_value REAL NOT NULL,
                        timestamp REAL NOT NULL,
                        acknowledged BOOLEAN NOT NULL
                    )
                ''')
                
                # Save metrics
                for metric_name, time_series in self.metrics.items():
                    data = time_series.get_data()
                    for point in data:
                        cursor.execute('''
                            INSERT OR REPLACE INTO metrics (name, value, timestamp, labels)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            metric_name,
                            point['value'],
                            point['timestamp'],
                            json.dumps(point['labels'])
                        ))
                
                # Save alerts
                with self.alert_lock:
                    for alert in self.alerts:
                        cursor.execute('''
                            INSERT OR REPLACE INTO alerts 
                            (id, level, message, metric_name, threshold, current_value, timestamp, acknowledged)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            alert.id,
                            alert.level.value,
                            alert.message,
                            alert.metric_name,
                            alert.threshold,
                            alert.current_value,
                            alert.timestamp,
                            alert.acknowledged
                        ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def get_metric(self, name: str, start_time: float = None, end_time: float = None) -> List[Dict]:
        """Get metric data for a specific metric"""
        if name not in self.metrics:
            return []
        
        return self.metrics[name].get_data(start_time, end_time)

    def get_metric_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a specific metric"""
        if name not in self.metrics:
            return {}
        
        return self.metrics[name].get_statistics()

    def get_all_metrics(self) -> Dict[str, List[Dict]]:
        """Get all metrics data"""
        return {
            name: time_series.get_data()
            for name, time_series in self.metrics.items()
        }

    def get_alerts(self, level: Optional[AlertLevel] = None, acknowledged: Optional[bool] = None) -> List[Alert]:
        """Get alerts with optional filtering"""
        with self.alert_lock:
            alerts = self.alerts.copy()
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        if acknowledged is not None:
            alerts = [alert for alert in alerts if alert.acknowledged == acknowledged]
        
        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        with self.alert_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return True
        return False

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function for alerts"""
        self.alert_callbacks.append(callback)

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        if not self.config.enable_prometheus:
            return ""
        
        return generate_latest()

    def generate_report(self, output_path: str = None) -> str:
        """Generate a performance report"""
        try:
            # Create report data
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'alerts': [],
                'summary': {}
            }
            
            # Add metrics statistics
            for metric_name in self.metrics:
                stats = self.get_metric_statistics(metric_name)
                report_data['metrics'][metric_name] = stats
            
            # Add active alerts
            active_alerts = self.get_alerts(acknowledged=False)
            report_data['alerts'] = [
                {
                    'id': alert.id,
                    'level': alert.level.value,
                    'message': alert.message,
                    'metric_name': alert.metric_name,
                    'timestamp': datetime.fromtimestamp(alert.timestamp).isoformat()
                }
                for alert in active_alerts
            ]
            
            # Generate summary
            report_data['summary'] = {
                'total_metrics': len(self.metrics),
                'active_alerts': len(active_alerts),
                'monitoring_duration': time.time() - min(
                    (ts.get_latest()['timestamp'] for ts in self.metrics.values() if ts.get_latest()),
                    default=time.time()
                )
            }
            
            # Convert to JSON
            report_json = json.dumps(report_data, indent=2)
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(report_json)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            return report_json
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""

    def plot_metrics(self, metric_names: List[str], output_path: str = None):
        """Generate plots for specified metrics"""
        try:
            fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))
            if len(metric_names) == 1:
                axes = [axes]
            
            for i, metric_name in enumerate(metric_names):
                if metric_name not in self.metrics:
                    continue
                
                data = self.metrics[metric_name].get_data()
                if not data:
                    continue
                
                timestamps = [point['timestamp'] for point in data]
                values = [point['value'] for point in data]
                
                # Convert timestamps to datetime
                dates = [datetime.fromtimestamp(ts) for ts in timestamps]
                
                axes[i].plot(dates, values)
                axes[i].set_title(f'{metric_name}')
                axes[i].set_ylabel('Value')
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")

# Usage example
async def main():
    
    """main function."""
# Configure alert thresholds
    alert_thresholds = {
        'system.cpu.usage': {
            'warning': 70.0,
            'error': 85.0,
            'critical': 95.0
        },
        'system.memory.usage': {
            'warning': 80.0,
            'error': 90.0,
            'critical': 95.0
        },
        'system.disk.usage': {
            'warning': 85.0,
            'error': 90.0,
            'critical': 95.0
        }
    }
    
    # Initialize monitor
    config = PerformanceConfig(
        collection_interval=2.0,
        retention_period=3600,  # 1 hour
        enable_prometheus=True,
        enable_alerting=True,
        enable_storage=True,
        alert_thresholds=alert_thresholds
    )
    
    monitor = OptimizedPerformanceMonitor(config)
    
    # Add alert callback
    def alert_handler(alert: Alert):
        
    """alert_handler function."""
print(f"ALERT: {alert.level.value.upper()} - {alert.message}")
    
    monitor.add_alert_callback(alert_handler)
    
    try:
        # Start monitoring
        await monitor.start()
        
        # Monitor for some time
        await asyncio.sleep(30)
        
        # Generate report
        report = monitor.generate_report("performance_report.json")
        print("Performance report generated")
        
        # Plot some metrics
        monitor.plot_metrics(['system.cpu.usage', 'system.memory.usage'], "metrics_plot.png")
        print("Metrics plot generated")
        
        # Get current statistics
        cpu_stats = monitor.get_metric_statistics('system.cpu.usage')
        print(f"CPU statistics: {cpu_stats}")
        
        # Get active alerts
        alerts = monitor.get_alerts(acknowledged=False)
        print(f"Active alerts: {len(alerts)}")
        
    finally:
        await monitor.stop()

match __name__:
    case "__main__":
    asyncio.run(main()) 