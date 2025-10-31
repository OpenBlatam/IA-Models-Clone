#!/usr/bin/env python3
"""
📊 HeyGen AI - Comprehensive Monitoring & Analytics System
=========================================================

This module provides advanced monitoring and analytics for the HeyGen AI system:
- Real-time performance monitoring
- Advanced analytics and insights
- Predictive analytics and forecasting
- Automated alerting and notifications
- Comprehensive reporting and dashboards
"""

import asyncio
import time
import psutil
import logging
import json
import sqlite3
import threading
import queue
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
import hashlib
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertLevel(str, Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    unit: str = ""

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    name: str
    level: AlertLevel
    status: AlertStatus
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """System health status"""
    overall_score: float = 0.0
    cpu_health: float = 0.0
    memory_health: float = 0.0
    disk_health: float = 0.0
    network_health: float = 0.0
    gpu_health: float = 0.0
    application_health: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_buffer = deque(maxlen=10000)
        self.collection_active = False
        self._lock = threading.RLock()
        self._collection_thread = None
    
    def start_collection(self):
        """Start metrics collection"""
        if self.collection_active:
            return
        
        self.collection_active = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.collection_active = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.collection_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        with self._lock:
            current_time = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            self._add_metric("cpu_usage_percent", cpu_percent, MetricType.GAUGE, {"type": "system"})
            self._add_metric("cpu_count", cpu_count, MetricType.GAUGE, {"type": "system"})
            if cpu_freq:
                self._add_metric("cpu_frequency_mhz", cpu_freq.current, MetricType.GAUGE, {"type": "system"})
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            self._add_metric("memory_usage_percent", memory.percent, MetricType.GAUGE, {"type": "system"})
            self._add_metric("memory_available_gb", memory.available / (1024**3), MetricType.GAUGE, {"type": "system"})
            self._add_metric("memory_used_gb", memory.used / (1024**3), MetricType.GAUGE, {"type": "system"})
            self._add_metric("swap_usage_percent", swap.percent, MetricType.GAUGE, {"type": "system"})
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            self._add_metric("disk_usage_percent", (disk_usage.used / disk_usage.total) * 100, MetricType.GAUGE, {"type": "system"})
            self._add_metric("disk_free_gb", disk_usage.free / (1024**3), MetricType.GAUGE, {"type": "system"})
            if disk_io:
                self._add_metric("disk_read_bytes", disk_io.read_bytes, MetricType.COUNTER, {"type": "system"})
                self._add_metric("disk_write_bytes", disk_io.write_bytes, MetricType.COUNTER, {"type": "system"})
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if net_io:
                self._add_metric("network_bytes_sent", net_io.bytes_sent, MetricType.COUNTER, {"type": "system"})
                self._add_metric("network_bytes_recv", net_io.bytes_recv, MetricType.COUNTER, {"type": "system"})
                self._add_metric("network_packets_sent", net_io.packets_sent, MetricType.COUNTER, {"type": "system"})
                self._add_metric("network_packets_recv", net_io.packets_recv, MetricType.COUNTER, {"type": "system"})
            
            # Process metrics
            process = psutil.Process()
            self._add_metric("process_cpu_percent", process.cpu_percent(), MetricType.GAUGE, {"type": "application"})
            self._add_metric("process_memory_mb", process.memory_info().rss / (1024**2), MetricType.GAUGE, {"type": "application"})
            self._add_metric("process_threads", process.num_threads(), MetricType.GAUGE, {"type": "application"})
            self._add_metric("process_fds", process.num_fds() if hasattr(process, 'num_fds') else 0, MetricType.GAUGE, {"type": "application"})
    
    def _add_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None):
        """Add metric to buffer"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            timestamp=datetime.now()
        )
        self.metrics_buffer.append(metric)
    
    def add_custom_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, 
                         labels: Dict[str, str] = None, unit: str = ""):
        """Add custom metric"""
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels or {},
                timestamp=datetime.now(),
                unit=unit
            )
            self.metrics_buffer.append(metric)
    
    def get_metrics(self, name: Optional[str] = None, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Metric]:
        """Get metrics with optional filtering"""
        with self._lock:
            metrics = list(self.metrics_buffer)
            
            # Filter by name
            if name:
                metrics = [m for m in metrics if m.name == name]
            
            # Filter by time range
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            return sorted(metrics, key=lambda m: m.timestamp)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest metric values"""
        with self._lock:
            latest_metrics = {}
            for metric in reversed(self.metrics_buffer):
                if metric.name not in latest_metrics:
                    latest_metrics[metric.name] = metric.value
            return latest_metrics

class AlertingSystem:
    """Advanced alerting system"""
    
    def __init__(self):
        self.alerts = {}
        self.alert_rules = self._load_alert_rules()
        self.alert_handlers = {}
        self.alert_history = deque(maxlen=1000)
        self._lock = threading.RLock()
    
    def _load_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load alert rules configuration"""
        return {
            'cpu_usage_high': {
                'metric_name': 'cpu_usage_percent',
                'threshold': 80.0,
                'level': AlertLevel.WARNING,
                'message': 'CPU usage is high'
            },
            'cpu_usage_critical': {
                'metric_name': 'cpu_usage_percent',
                'threshold': 95.0,
                'level': AlertLevel.CRITICAL,
                'message': 'CPU usage is critical'
            },
            'memory_usage_high': {
                'metric_name': 'memory_usage_percent',
                'threshold': 85.0,
                'level': AlertLevel.WARNING,
                'message': 'Memory usage is high'
            },
            'memory_usage_critical': {
                'metric_name': 'memory_usage_percent',
                'threshold': 95.0,
                'level': AlertLevel.CRITICAL,
                'message': 'Memory usage is critical'
            },
            'disk_usage_high': {
                'metric_name': 'disk_usage_percent',
                'threshold': 90.0,
                'level': AlertLevel.WARNING,
                'message': 'Disk usage is high'
            },
            'disk_usage_critical': {
                'metric_name': 'disk_usage_percent',
                'threshold': 98.0,
                'level': AlertLevel.CRITICAL,
                'message': 'Disk usage is critical'
            }
        }
    
    def register_handler(self, alert_level: AlertLevel, handler: Callable[[Alert], None]):
        """Register alert handler"""
        if alert_level not in self.alert_handlers:
            self.alert_handlers[alert_level] = []
        self.alert_handlers[alert_level].append(handler)
    
    def check_alerts(self, metrics: List[Metric]):
        """Check metrics against alert rules"""
        with self._lock:
            for rule_name, rule in self.alert_rules.items():
                metric_name = rule['threshold']
                threshold = rule['threshold']
                level = rule['level']
                message = rule['message']
                
                # Find latest metric value
                latest_metric = None
                for metric in reversed(metrics):
                    if metric.name == metric_name:
                        latest_metric = metric
                        break
                
                if latest_metric and latest_metric.value >= threshold:
                    # Check if alert already exists
                    alert_key = f"{rule_name}_{latest_metric.timestamp.date()}"
                    
                    if alert_key not in self.alerts or self.alerts[alert_key].status == AlertStatus.RESOLVED:
                        # Create new alert
                        alert = Alert(
                            alert_id=secrets.token_urlsafe(16),
                            name=rule_name,
                            level=level,
                            status=AlertStatus.ACTIVE,
                            message=message,
                            metric_name=metric_name,
                            threshold=threshold,
                            current_value=latest_metric.value,
                            timestamp=datetime.now(),
                            metadata={'rule': rule_name}
                        )
                        
                        self.alerts[alert_key] = alert
                        self.alert_history.append(alert)
                        
                        # Trigger handlers
                        self._trigger_handlers(alert)
                else:
                    # Resolve existing alert if metric is below threshold
                    alert_key = f"{rule_name}_{datetime.now().date()}"
                    if alert_key in self.alerts and self.alerts[alert_key].status == AlertStatus.ACTIVE:
                        self.alerts[alert_key].status = AlertStatus.RESOLVED
                        self.alerts[alert_key].resolved_at = datetime.now()
    
    def _trigger_handlers(self, alert: Alert):
        """Trigger alert handlers"""
        handlers = self.alert_handlers.get(alert.level, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        with self._lock:
            return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        with self._lock:
            active_alerts = self.get_active_alerts()
            
            summary = {
                'total_alerts': len(self.alert_history),
                'active_alerts': len(active_alerts),
                'alerts_by_level': defaultdict(int),
                'alerts_by_metric': defaultdict(int)
            }
            
            for alert in active_alerts:
                summary['alerts_by_level'][alert.level.value] += 1
                summary['alerts_by_metric'][alert.metric_name] += 1
            
            return dict(summary)

class AnalyticsEngine:
    """Advanced analytics engine"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize analytics database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    labels TEXT,
                    timestamp DATETIME NOT NULL,
                    unit TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    current_value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    resolved_at DATETIME,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                ON alerts(timestamp)
            ''')
    
    def store_metrics(self, metrics: List[Metric]):
        """Store metrics in database"""
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute('''
                    INSERT INTO metrics (name, value, metric_type, labels, timestamp, unit)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.name,
                    metric.value,
                    metric.metric_type.value,
                    json.dumps(metric.labels),
                    metric.timestamp,
                    metric.unit
                ))
    
    def store_alert(self, alert: Alert):
        """Store alert in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO alerts 
                (alert_id, name, level, status, message, metric_name, threshold, 
                 current_value, timestamp, resolved_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.name,
                alert.level.value,
                alert.status.value,
                alert.message,
                alert.metric_name,
                alert.threshold,
                alert.current_value,
                alert.timestamp,
                alert.resolved_at,
                json.dumps(alert.metadata)
            ))
    
    def get_metric_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get metric trends for analysis"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT value, timestamp FROM metrics 
                WHERE name = ? AND timestamp >= ?
                ORDER BY timestamp
            ''', (metric_name, start_time))
            
            rows = cursor.fetchall()
            
            if not rows:
                return {'error': 'No data available'}
            
            values = [row['value'] for row in rows]
            timestamps = [row['timestamp'] for row in rows]
            
            # Calculate trends
            if len(values) > 1:
                trend = self._calculate_trend(values)
                volatility = self._calculate_volatility(values)
                mean_value = np.mean(values)
                max_value = np.max(values)
                min_value = np.min(values)
            else:
                trend = 0
                volatility = 0
                mean_value = values[0] if values else 0
                max_value = values[0] if values else 0
                min_value = values[0] if values else 0
            
            return {
                'metric_name': metric_name,
                'time_range_hours': hours,
                'data_points': len(values),
                'trend': trend,
                'volatility': volatility,
                'mean': mean_value,
                'max': max_value,
                'min': min_value,
                'values': values,
                'timestamps': timestamps
            }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        n = len(values)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return slope
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation)"""
        if len(values) < 2:
            return 0.0
        
        return float(np.std(values))
    
    def get_system_health_score(self, hours: int = 1) -> SystemHealth:
        """Calculate system health score"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get latest metrics
            cursor = conn.execute('''
                SELECT name, value FROM metrics 
                WHERE timestamp >= ?
                GROUP BY name
                HAVING timestamp = MAX(timestamp)
            ''', (start_time,))
            
            latest_metrics = {row['name']: row['value'] for row in cursor.fetchall()}
        
        # Calculate health scores
        cpu_health = self._calculate_cpu_health(latest_metrics.get('cpu_usage_percent', 0))
        memory_health = self._calculate_memory_health(latest_metrics.get('memory_usage_percent', 0))
        disk_health = self._calculate_disk_health(latest_metrics.get('disk_usage_percent', 0))
        network_health = self._calculate_network_health(latest_metrics)
        gpu_health = self._calculate_gpu_health(latest_metrics)
        application_health = self._calculate_application_health(latest_metrics)
        
        # Calculate overall score
        overall_score = (cpu_health + memory_health + disk_health + 
                        network_health + gpu_health + application_health) / 6
        
        return SystemHealth(
            overall_score=overall_score,
            cpu_health=cpu_health,
            memory_health=memory_health,
            disk_health=disk_health,
            network_health=network_health,
            gpu_health=gpu_health,
            application_health=application_health,
            timestamp=datetime.now()
        )
    
    def _calculate_cpu_health(self, cpu_usage: float) -> float:
        """Calculate CPU health score"""
        if cpu_usage < 50:
            return 100.0
        elif cpu_usage < 80:
            return 100.0 - (cpu_usage - 50) * 1.67
        else:
            return max(0.0, 50.0 - (cpu_usage - 80) * 2.5)
    
    def _calculate_memory_health(self, memory_usage: float) -> float:
        """Calculate memory health score"""
        if memory_usage < 60:
            return 100.0
        elif memory_usage < 85:
            return 100.0 - (memory_usage - 60) * 2.0
        else:
            return max(0.0, 50.0 - (memory_usage - 85) * 3.33)
    
    def _calculate_disk_health(self, disk_usage: float) -> float:
        """Calculate disk health score"""
        if disk_usage < 70:
            return 100.0
        elif disk_usage < 90:
            return 100.0 - (disk_usage - 70) * 2.5
        else:
            return max(0.0, 50.0 - (disk_usage - 90) * 5.0)
    
    def _calculate_network_health(self, metrics: Dict[str, float]) -> float:
        """Calculate network health score"""
        # Simple network health based on availability
        return 100.0  # Placeholder
    
    def _calculate_gpu_health(self, metrics: Dict[str, float]) -> float:
        """Calculate GPU health score"""
        gpu_usage = metrics.get('gpu_usage_percent', 0)
        if gpu_usage == 0:
            return 100.0  # No GPU or not used
        
        if gpu_usage < 80:
            return 100.0
        elif gpu_usage < 95:
            return 100.0 - (gpu_usage - 80) * 1.33
        else:
            return max(0.0, 80.0 - (gpu_usage - 95) * 4.0)
    
    def _calculate_application_health(self, metrics: Dict[str, float]) -> float:
        """Calculate application health score"""
        # Based on process metrics
        process_cpu = metrics.get('process_cpu_percent', 0)
        process_memory = metrics.get('process_memory_mb', 0)
        
        cpu_score = 100.0 if process_cpu < 50 else max(0.0, 100.0 - (process_cpu - 50) * 2)
        memory_score = 100.0 if process_memory < 1000 else max(0.0, 100.0 - (process_memory - 1000) / 100)
        
        return (cpu_score + memory_score) / 2

class ReportingSystem:
    """Advanced reporting system"""
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.analytics_engine = analytics_engine
        self.report_templates = self._load_report_templates()
    
    def _load_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load report templates"""
        return {
            'system_overview': {
                'title': 'System Overview Report',
                'metrics': ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent'],
                'time_range': 24
            },
            'performance_analysis': {
                'title': 'Performance Analysis Report',
                'metrics': ['cpu_usage_percent', 'memory_usage_percent', 'process_cpu_percent'],
                'time_range': 168  # 1 week
            },
            'alert_summary': {
                'title': 'Alert Summary Report',
                'metrics': [],
                'time_range': 24
            }
        }
    
    def generate_report(self, report_type: str, output_format: str = 'json') -> Dict[str, Any]:
        """Generate report"""
        if report_type not in self.report_templates:
            return {'error': f'Unknown report type: {report_type}'}
        
        template = self.report_templates[report_type]
        time_range = template['time_range']
        
        report_data = {
            'report_type': report_type,
            'title': template['title'],
            'generated_at': datetime.now().isoformat(),
            'time_range_hours': time_range,
            'data': {}
        }
        
        # Generate metric trends
        for metric_name in template['metrics']:
            trends = self.analytics_engine.get_metric_trends(metric_name, time_range)
            report_data['data'][metric_name] = trends
        
        # Generate system health
        health = self.analytics_engine.get_system_health_score(time_range)
        report_data['system_health'] = {
            'overall_score': health.overall_score,
            'cpu_health': health.cpu_health,
            'memory_health': health.memory_health,
            'disk_health': health.disk_health,
            'network_health': health.network_health,
            'gpu_health': health.gpu_health,
            'application_health': health.application_health
        }
        
        return report_data
    
    def generate_visualization(self, metric_name: str, hours: int = 24, 
                             output_file: Optional[str] = None) -> str:
        """Generate metric visualization"""
        trends = self.analytics_engine.get_metric_trends(metric_name, hours)
        
        if 'error' in trends:
            return f"Error generating visualization: {trends['error']}"
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trends['timestamps'],
            y=trends['values'],
            mode='lines+markers',
            name=metric_name,
            line=dict(color='blue', width=2)
        ))
        
        # Add trend line
        if len(trends['values']) > 1:
            x = np.arange(len(trends['values']))
            y = np.array(trends['values'])
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            trend_line = p(x)
            
            fig.add_trace(go.Scatter(
                x=trends['timestamps'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=f'{metric_name} - Last {hours} Hours',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        if output_file:
            fig.write_html(output_file)
            return f"Visualization saved to {output_file}"
        else:
            return fig.to_html()

class ComprehensiveMonitoringSystem:
    """Main comprehensive monitoring system"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.metrics_collector = MetricsCollector()
        self.alerting_system = AlertingSystem()
        self.analytics_engine = AnalyticsEngine(db_path)
        self.reporting_system = ReportingSystem(self.analytics_engine)
        self.initialized = False
        self._storage_thread = None
        self._storage_active = False
    
    async def initialize(self):
        """Initialize monitoring system"""
        try:
            logger.info("Initializing Comprehensive Monitoring System...")
            
            # Start metrics collection
            self.metrics_collector.start_collection()
            
            # Start storage thread
            self._start_storage_thread()
            
            # Register alert handlers
            self._register_alert_handlers()
            
            self.initialized = True
            logger.info("Comprehensive Monitoring System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {e}")
            raise
    
    def _start_storage_thread(self):
        """Start storage thread"""
        self._storage_active = True
        self._storage_thread = threading.Thread(target=self._storage_loop, daemon=True)
        self._storage_thread.start()
    
    def _storage_loop(self):
        """Storage loop for persisting data"""
        while self._storage_active:
            try:
                # Get latest metrics
                metrics = self.metrics_collector.get_metrics()
                if metrics:
                    # Store metrics
                    self.analytics_engine.store_metrics(metrics)
                    
                    # Check alerts
                    self.alerting_system.check_alerts(metrics)
                
                # Store alerts
                active_alerts = self.alerting_system.get_active_alerts()
                for alert in active_alerts:
                    self.analytics_engine.store_alert(alert)
                
                time.sleep(60)  # Store every minute
                
            except Exception as e:
                logger.error(f"Error in storage loop: {e}")
                time.sleep(10)
    
    def _register_alert_handlers(self):
        """Register alert handlers"""
        def log_alert(alert: Alert):
            logger.warning(f"ALERT [{alert.level.value}]: {alert.message} - {alert.current_value}")
        
        def critical_alert_handler(alert: Alert):
            logger.critical(f"CRITICAL ALERT: {alert.message} - {alert.current_value}")
            # Here you could send notifications, emails, etc.
        
        # Register handlers for each level
        self.alerting_system.register_handler(AlertLevel.INFO, log_alert)
        self.alerting_system.register_handler(AlertLevel.WARNING, log_alert)
        self.alerting_system.register_handler(AlertLevel.ERROR, log_alert)
        self.alerting_system.register_handler(AlertLevel.CRITICAL, critical_alert_handler)
    
    def add_custom_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                         labels: Dict[str, str] = None, unit: str = ""):
        """Add custom metric"""
        self.metrics_collector.add_custom_metric(name, value, metric_type, labels, unit)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.initialized:
            return {'status': 'not_initialized'}
        
        # Get latest metrics
        latest_metrics = self.metrics_collector.get_latest_metrics()
        
        # Get system health
        health = self.analytics_engine.get_system_health_score()
        
        # Get alert summary
        alert_summary = self.alerting_system.get_alert_summary()
        
        return {
            'status': 'operational',
            'latest_metrics': latest_metrics,
            'system_health': health.__dict__,
            'alert_summary': alert_summary,
            'collection_active': self.metrics_collector.collection_active
        }
    
    def generate_report(self, report_type: str = 'system_overview') -> Dict[str, Any]:
        """Generate monitoring report"""
        return self.reporting_system.generate_report(report_type)
    
    def generate_visualization(self, metric_name: str, hours: int = 24) -> str:
        """Generate metric visualization"""
        return self.reporting_system.generate_visualization(metric_name, hours)
    
    async def shutdown(self):
        """Shutdown monitoring system"""
        logger.info("Shutting down Comprehensive Monitoring System...")
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Stop storage thread
        self._storage_active = False
        if self._storage_thread:
            self._storage_thread.join(timeout=5)
        
        self.initialized = False
        logger.info("Comprehensive Monitoring System shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the comprehensive monitoring system"""
    print("📊 HeyGen AI - Comprehensive Monitoring System Demo")
    print("=" * 60)
    
    # Initialize monitoring system
    monitoring_system = ComprehensiveMonitoringSystem()
    
    try:
        # Initialize the system
        await monitoring_system.initialize()
        
        # Add some custom metrics
        monitoring_system.add_custom_metric("api_requests_total", 150, MetricType.COUNTER, {"endpoint": "/api/v1/users"})
        monitoring_system.add_custom_metric("api_response_time_ms", 45.2, MetricType.HISTOGRAM, {"endpoint": "/api/v1/users"})
        monitoring_system.add_custom_metric("active_users", 1250, MetricType.GAUGE, {"type": "concurrent"})
        
        # Wait a bit for data collection
        print("⏳ Collecting metrics for 10 seconds...")
        await asyncio.sleep(10)
        
        # Get system status
        print("\n📊 System Status:")
        status = monitoring_system.get_system_status()
        print(f"Status: {status['status']}")
        print(f"Collection Active: {status['collection_active']}")
        
        # Display latest metrics
        print(f"\n📈 Latest Metrics:")
        for metric_name, value in status['latest_metrics'].items():
            print(f"  {metric_name}: {value:.2f}")
        
        # Display system health
        health = status['system_health']
        print(f"\n🏥 System Health:")
        print(f"  Overall Score: {health['overall_score']:.2f}")
        print(f"  CPU Health: {health['cpu_health']:.2f}")
        print(f"  Memory Health: {health['memory_health']:.2f}")
        print(f"  Disk Health: {health['disk_health']:.2f}")
        
        # Display alert summary
        alert_summary = status['alert_summary']
        print(f"\n🚨 Alert Summary:")
        print(f"  Total Alerts: {alert_summary['total_alerts']}")
        print(f"  Active Alerts: {alert_summary['active_alerts']}")
        
        # Generate report
        print(f"\n📋 Generating System Overview Report...")
        report = monitoring_system.generate_report('system_overview')
        print(f"Report generated: {report['title']}")
        print(f"Time range: {report['time_range_hours']} hours")
        
        # Generate visualization
        print(f"\n📊 Generating CPU Usage Visualization...")
        viz_result = monitoring_system.generate_visualization('cpu_usage_percent', 1)
        if not viz_result.startswith("Error"):
            print("Visualization generated successfully")
        else:
            print(f"Visualization error: {viz_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Shutdown
        await monitoring_system.shutdown()
        print("\n✅ Monitoring system shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())


