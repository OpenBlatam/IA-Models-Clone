"""
Advanced Performance Monitoring Framework for HeyGen AI Testing System.
Comprehensive performance monitoring including real-time metrics, alerting,
and performance optimization recommendations.
"""

import time
import psutil
import threading
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import queue
import signal
import sys

@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    category: str  # cpu, memory, disk, network, test
    severity: str = "info"  # info, warning, critical
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Represents a performance alert."""
    alert_id: str
    metric_name: str
    threshold: float
    current_value: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class PerformanceReport:
    """Represents a performance report."""
    report_id: str
    start_time: datetime
    end_time: datetime
    metrics_summary: Dict[str, Any]
    alerts: List[PerformanceAlert]
    recommendations: List[str]
    performance_score: float
    generated_at: datetime = field(default_factory=datetime.now)

class SystemMetricsCollector:
    """Collects system performance metrics."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.metrics_queue = queue.Queue()
        self.collecting = False
        self.collector_thread = None
        self.metrics_history = deque(maxlen=10000)
        self.start_time = None
        
    def start_collection(self):
        """Start metrics collection."""
        if self.collecting:
            return
        
        self.collecting = True
        self.start_time = time.time()
        self.collector_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self.collector_thread.start()
        logging.info("Performance metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.collecting = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
        logging.info("Performance metrics collection stopped")
    
    def _collect_metrics(self):
        """Collect system metrics in a loop."""
        while self.collecting:
            try:
                timestamp = datetime.now()
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                # Disk metrics
                disk_usage = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                
                # Network metrics
                network_io = psutil.net_io_counters()
                
                # Process metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()
                
                # Collect metrics
                metrics = [
                    PerformanceMetric("cpu_percent", cpu_percent, "%", timestamp, "cpu"),
                    PerformanceMetric("cpu_count", cpu_count, "cores", timestamp, "cpu"),
                    PerformanceMetric("memory_percent", memory.percent, "%", timestamp, "memory"),
                    PerformanceMetric("memory_used_mb", memory.used / 1024 / 1024, "MB", timestamp, "memory"),
                    PerformanceMetric("memory_available_mb", memory.available / 1024 / 1024, "MB", timestamp, "memory"),
                    PerformanceMetric("disk_usage_percent", disk_usage.percent, "%", timestamp, "disk"),
                    PerformanceMetric("disk_free_gb", disk_usage.free / 1024 / 1024 / 1024, "GB", timestamp, "disk"),
                    PerformanceMetric("network_bytes_sent", network_io.bytes_sent, "bytes", timestamp, "network"),
                    PerformanceMetric("network_bytes_recv", network_io.bytes_recv, "bytes", timestamp, "network"),
                    PerformanceMetric("process_memory_mb", process_memory.rss / 1024 / 1024, "MB", timestamp, "test"),
                    PerformanceMetric("process_cpu_percent", process_cpu, "%", timestamp, "test"),
                ]
                
                # Add CPU frequency if available
                if cpu_freq:
                    metrics.append(PerformanceMetric("cpu_freq_mhz", cpu_freq.current, "MHz", timestamp, "cpu"))
                
                # Add swap metrics
                metrics.append(PerformanceMetric("swap_percent", swap.percent, "%", timestamp, "memory"))
                metrics.append(PerformanceMetric("swap_used_mb", swap.used / 1024 / 1024, "MB", timestamp, "memory"))
                
                # Add disk I/O metrics
                if disk_io:
                    metrics.append(PerformanceMetric("disk_read_mb", disk_io.read_bytes / 1024 / 1024, "MB", timestamp, "disk"))
                    metrics.append(PerformanceMetric("disk_write_mb", disk_io.write_bytes / 1024 / 1024, "MB", timestamp, "disk"))
                
                # Store metrics
                for metric in metrics:
                    self.metrics_history.append(metric)
                    self.metrics_queue.put(metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logging.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def get_latest_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get the latest metrics for each type."""
        latest_metrics = {}
        
        for metric in reversed(self.metrics_history):
            if metric.metric_name not in latest_metrics:
                latest_metrics[metric.metric_name] = metric
        
        return latest_metrics
    
    def get_metrics_history(self, metric_name: str, duration_minutes: int = 60) -> List[PerformanceMetric]:
        """Get metrics history for a specific metric."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        return [
            metric for metric in self.metrics_history
            if metric.metric_name == metric_name and metric.timestamp >= cutoff_time
        ]

class PerformanceAnalyzer:
    """Analyzes performance metrics and generates insights."""
    
    def __init__(self):
        self.thresholds = {
            "cpu_percent": {"warning": 70, "critical": 90},
            "memory_percent": {"warning": 80, "critical": 95},
            "disk_usage_percent": {"warning": 85, "critical": 95},
            "process_memory_mb": {"warning": 500, "critical": 1000},
            "process_cpu_percent": {"warning": 50, "critical": 80}
        }
    
    def analyze_metrics(self, metrics: List[PerformanceMetric]) -> List[PerformanceAlert]:
        """Analyze metrics and generate alerts."""
        alerts = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.metric_name].append(metric)
        
        # Analyze each metric type
        for metric_name, metric_list in metrics_by_name.items():
            if metric_name not in self.thresholds:
                continue
            
            # Get latest value
            latest_metric = max(metric_list, key=lambda x: x.timestamp)
            current_value = latest_metric.value
            
            # Check thresholds
            thresholds = self.thresholds[metric_name]
            
            if current_value >= thresholds["critical"]:
                alert = PerformanceAlert(
                    alert_id=f"alert_{int(time.time())}_{metric_name}",
                    metric_name=metric_name,
                    threshold=thresholds["critical"],
                    current_value=current_value,
                    severity="critical",
                    message=f"{metric_name} is at critical level: {current_value:.2f} {latest_metric.unit}",
                    timestamp=datetime.now()
                )
                alerts.append(alert)
            
            elif current_value >= thresholds["warning"]:
                alert = PerformanceAlert(
                    alert_id=f"alert_{int(time.time())}_{metric_name}",
                    metric_name=metric_name,
                    threshold=thresholds["warning"],
                    current_value=current_value,
                    severity="warning",
                    message=f"{metric_name} is at warning level: {current_value:.2f} {latest_metric.unit}",
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        return alerts
    
    def calculate_performance_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall performance score (0-100)."""
        if not metrics:
            return 0.0
        
        # Get latest metrics
        latest_metrics = {}
        for metric in metrics:
            if metric.metric_name not in latest_metrics:
                latest_metrics[metric.metric_name] = metric
        
        score = 100.0
        
        # Penalize based on resource usage
        for metric_name, thresholds in self.thresholds.items():
            if metric_name in latest_metrics:
                value = latest_metrics[metric_name].value
                
                if value >= thresholds["critical"]:
                    score -= 30
                elif value >= thresholds["warning"]:
                    score -= 15
                else:
                    # Bonus for good performance
                    optimal_range = thresholds["warning"] * 0.5
                    if value < optimal_range:
                        score += 5
        
        return max(0.0, min(100.0, score))
    
    def generate_recommendations(self, metrics: List[PerformanceMetric], 
                               alerts: List[PerformanceAlert]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze CPU usage
        cpu_metrics = [m for m in metrics if m.metric_name == "cpu_percent"]
        if cpu_metrics:
            avg_cpu = np.mean([m.value for m in cpu_metrics])
            if avg_cpu > 80:
                recommendations.append("High CPU usage detected - consider optimizing algorithms or adding more CPU cores")
            elif avg_cpu < 20:
                recommendations.append("Low CPU usage - system may be underutilized")
        
        # Analyze memory usage
        memory_metrics = [m for m in metrics if m.metric_name == "memory_percent"]
        if memory_metrics:
            avg_memory = np.mean([m.value for m in memory_metrics])
            if avg_memory > 85:
                recommendations.append("High memory usage - consider memory optimization or increasing RAM")
            elif avg_memory < 30:
                recommendations.append("Low memory usage - system may be underutilized")
        
        # Analyze disk usage
        disk_metrics = [m for m in metrics if m.metric_name == "disk_usage_percent"]
        if disk_metrics:
            avg_disk = np.mean([m.value for m in disk_metrics])
            if avg_disk > 90:
                recommendations.append("High disk usage - consider cleaning up files or expanding storage")
        
        # Analyze process-specific metrics
        process_memory_metrics = [m for m in metrics if m.metric_name == "process_memory_mb"]
        if process_memory_metrics:
            max_memory = max([m.value for m in process_memory_metrics])
            if max_memory > 1000:
                recommendations.append("High process memory usage - consider memory profiling and optimization")
        
        # Analyze trends
        if len(metrics) > 10:
            # Check for memory leaks
            memory_values = [m.value for m in memory_metrics[-10:]]
            if len(memory_values) > 5:
                trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
                if trend > 1.0:
                    recommendations.append("Memory usage is increasing - potential memory leak detected")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("System performance is within normal parameters")
        
        return recommendations

class PerformanceDatabase:
    """Manages performance data storage and retrieval."""
    
    def __init__(self, db_path: str = "performance_monitoring.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                category TEXT NOT NULL,
                severity TEXT DEFAULT 'info',
                metadata TEXT
            )
        """)
        
        # Create alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                metric_name TEXT NOT NULL,
                threshold REAL NOT NULL,
                current_value REAL NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time DATETIME
            )
        """)
        
        # Create reports table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME NOT NULL,
                performance_score REAL NOT NULL,
                metrics_summary TEXT NOT NULL,
                recommendations TEXT NOT NULL,
                generated_at DATETIME NOT NULL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
        
        conn.commit()
        conn.close()
    
    def store_metric(self, metric: PerformanceMetric):
        """Store a performance metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics (metric_name, value, unit, timestamp, category, severity, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.metric_name,
            metric.value,
            metric.unit,
            metric.timestamp.isoformat(),
            metric.category,
            metric.severity,
            json.dumps(metric.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def store_alert(self, alert: PerformanceAlert):
        """Store a performance alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO alerts 
            (alert_id, metric_name, threshold, current_value, severity, message, timestamp, resolved, resolution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id,
            alert.metric_name,
            alert.threshold,
            alert.current_value,
            alert.severity,
            alert.message,
            alert.timestamp.isoformat(),
            alert.resolved,
            alert.resolution_time.isoformat() if alert.resolution_time else None
        ))
        
        conn.commit()
        conn.close()
    
    def store_report(self, report: PerformanceReport):
        """Store a performance report."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO reports 
            (report_id, start_time, end_time, performance_score, metrics_summary, recommendations, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            report.report_id,
            report.start_time.isoformat(),
            report.end_time.isoformat(),
            report.performance_score,
            json.dumps(report.metrics_summary),
            json.dumps(report.recommendations),
            report.generated_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics(self, start_time: datetime, end_time: datetime, 
                   metric_name: Optional[str] = None) -> List[PerformanceMetric]:
        """Retrieve metrics from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT metric_name, value, unit, timestamp, category, severity, metadata
            FROM metrics
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_time.isoformat(), end_time.isoformat()]
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        query += " ORDER BY timestamp"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        metrics = []
        for row in rows:
            metric = PerformanceMetric(
                metric_name=row[0],
                value=row[1],
                unit=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                category=row[4],
                severity=row[5],
                metadata=json.loads(row[6]) if row[6] else {}
            )
            metrics.append(metric)
        
        conn.close()
        return metrics
    
    def get_alerts(self, start_time: datetime, end_time: datetime, 
                  severity: Optional[str] = None) -> List[PerformanceAlert]:
        """Retrieve alerts from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT alert_id, metric_name, threshold, current_value, severity, message, 
                   timestamp, resolved, resolution_time
            FROM alerts
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_time.isoformat(), end_time.isoformat()]
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        query += " ORDER BY timestamp"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        alerts = []
        for row in rows:
            alert = PerformanceAlert(
                alert_id=row[0],
                metric_name=row[1],
                threshold=row[2],
                current_value=row[3],
                severity=row[4],
                message=row[5],
                timestamp=datetime.fromisoformat(row[6]),
                resolved=bool(row[7]),
                resolution_time=datetime.fromisoformat(row[8]) if row[8] else None
            )
            alerts.append(alert)
        
        conn.close()
        return alerts

class PerformanceReporter:
    """Generates performance reports and visualizations."""
    
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self, metrics: List[PerformanceMetric], 
                       alerts: List[PerformanceAlert],
                       recommendations: List[str],
                       performance_score: float,
                       start_time: datetime,
                       end_time: datetime) -> PerformanceReport:
        """Generate a comprehensive performance report."""
        
        # Calculate metrics summary
        metrics_summary = self._calculate_metrics_summary(metrics)
        
        # Create report
        report = PerformanceReport(
            report_id=f"report_{int(time.time())}",
            start_time=start_time,
            end_time=end_time,
            metrics_summary=metrics_summary,
            alerts=alerts,
            recommendations=recommendations,
            performance_score=performance_score
        )
        
        # Generate visualizations
        self._generate_visualizations(metrics, report)
        
        # Generate HTML report
        self._generate_html_report(report)
        
        return report
    
    def _calculate_metrics_summary(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calculate summary statistics for metrics."""
        if not metrics:
            return {}
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.metric_name].append(metric)
        
        summary = {}
        for metric_name, metric_list in metrics_by_name.items():
            values = [m.value for m in metric_list]
            summary[metric_name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": np.mean(values),
                "std": np.std(values),
                "unit": metric_list[0].unit
            }
        
        return summary
    
    def _generate_visualizations(self, metrics: List[PerformanceMetric], report: PerformanceReport):
        """Generate performance visualizations."""
        if not metrics:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Group metrics by category
        metrics_by_category = defaultdict(list)
        for metric in metrics:
            metrics_by_category[metric.category].append(metric)
        
        # Create subplots for each category
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Report - {report.report_id}', fontsize=16)
        
        # CPU metrics
        if 'cpu' in metrics_by_category:
            cpu_metrics = metrics_by_category['cpu']
            ax = axes[0, 0]
            for metric in cpu_metrics:
                if metric.metric_name == 'cpu_percent':
                    timestamps = [m.timestamp for m in cpu_metrics if m.metric_name == 'cpu_percent']
                    values = [m.value for m in cpu_metrics if m.metric_name == 'cpu_percent']
                    ax.plot(timestamps, values, label='CPU %', linewidth=2)
            ax.set_title('CPU Usage')
            ax.set_ylabel('Percentage')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Memory metrics
        if 'memory' in metrics_by_category:
            memory_metrics = metrics_by_category['memory']
            ax = axes[0, 1]
            for metric in memory_metrics:
                if metric.metric_name == 'memory_percent':
                    timestamps = [m.timestamp for m in memory_metrics if m.metric_name == 'memory_percent']
                    values = [m.value for m in memory_metrics if m.metric_name == 'memory_percent']
                    ax.plot(timestamps, values, label='Memory %', linewidth=2, color='orange')
            ax.set_title('Memory Usage')
            ax.set_ylabel('Percentage')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Disk metrics
        if 'disk' in metrics_by_category:
            disk_metrics = metrics_by_category['disk']
            ax = axes[1, 0]
            for metric in disk_metrics:
                if metric.metric_name == 'disk_usage_percent':
                    timestamps = [m.timestamp for m in disk_metrics if m.metric_name == 'disk_usage_percent']
                    values = [m.value for m in disk_metrics if m.metric_name == 'disk_usage_percent']
                    ax.plot(timestamps, values, label='Disk %', linewidth=2, color='green')
            ax.set_title('Disk Usage')
            ax.set_ylabel('Percentage')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Test metrics
        if 'test' in metrics_by_category:
            test_metrics = metrics_by_category['test']
            ax = axes[1, 1]
            for metric in test_metrics:
                timestamps = [m.timestamp for m in test_metrics if m.metric_name == metric.metric_name]
                values = [m.value for m in test_metrics if m.metric_name == metric.metric_name]
                ax.plot(timestamps, values, label=metric.metric_name, linewidth=2)
            ax.set_title('Test Performance')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.output_dir / f"{report.report_id}_performance.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, report: PerformanceReport):
        """Generate HTML performance report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - {report.report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007cba; }}
                .alert {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .alert.warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                .alert.critical {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                .recommendation {{ margin: 10px 0; padding: 10px; background-color: #d1ecf1; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; color: {'green' if report.performance_score >= 80 else 'orange' if report.performance_score >= 60 else 'red'}; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Report</h1>
                <p><strong>Report ID:</strong> {report.report_id}</p>
                <p><strong>Period:</strong> {report.start_time} to {report.end_time}</p>
                <p><strong>Performance Score:</strong> <span class="score">{report.performance_score:.1f}/100</span></p>
            </div>
            
            <h2>Metrics Summary</h2>
            {self._generate_metrics_html(report.metrics_summary)}
            
            <h2>Alerts</h2>
            {self._generate_alerts_html(report.alerts)}
            
            <h2>Recommendations</h2>
            {self._generate_recommendations_html(report.recommendations)}
        </body>
        </html>
        """
        
        html_path = self.output_dir / f"{report.report_id}_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
    
    def _generate_metrics_html(self, metrics_summary: Dict[str, Any]) -> str:
        """Generate HTML for metrics summary."""
        html = ""
        for metric_name, stats in metrics_summary.items():
            html += f"""
            <div class="metric">
                <h3>{metric_name}</h3>
                <p><strong>Average:</strong> {stats['avg']:.2f} {stats['unit']}</p>
                <p><strong>Min:</strong> {stats['min']:.2f} {stats['unit']}</p>
                <p><strong>Max:</strong> {stats['max']:.2f} {stats['unit']}</p>
                <p><strong>Std Dev:</strong> {stats['std']:.2f} {stats['unit']}</p>
            </div>
            """
        return html
    
    def _generate_alerts_html(self, alerts: List[PerformanceAlert]) -> str:
        """Generate HTML for alerts."""
        if not alerts:
            return "<p>No alerts during this period.</p>"
        
        html = ""
        for alert in alerts:
            html += f"""
            <div class="alert {alert.severity}">
                <h4>{alert.metric_name} - {alert.severity.upper()}</h4>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Time:</strong> {alert.timestamp}</p>
            </div>
            """
        return html
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations."""
        html = ""
        for i, rec in enumerate(recommendations, 1):
            html += f"""
            <div class="recommendation">
                <p><strong>{i}.</strong> {rec}</p>
            </div>
            """
        return html

class PerformanceMonitoringFramework:
    """Main performance monitoring framework."""
    
    def __init__(self, collection_interval: float = 1.0, db_path: str = "performance_monitoring.db"):
        self.collector = SystemMetricsCollector(collection_interval)
        self.analyzer = PerformanceAnalyzer()
        self.database = PerformanceDatabase(db_path)
        self.reporter = PerformanceReporter()
        self.monitoring = False
        self.start_time = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = datetime.now()
        self.collector.start_collection()
        
        # Start background processing
        self._start_background_processing()
        
        logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self.collector.stop_collection()
        
        logging.info("Performance monitoring stopped")
    
    def _start_background_processing(self):
        """Start background processing of metrics."""
        def process_metrics():
            while self.monitoring:
                try:
                    # Get metrics from queue
                    metrics = []
                    while not self.collector.metrics_queue.empty():
                        metric = self.collector.metrics_queue.get_nowait()
                        metrics.append(metric)
                        self.database.store_metric(metric)
                    
                    if metrics:
                        # Analyze metrics
                        alerts = self.analyzer.analyze_metrics(metrics)
                        
                        # Store alerts
                        for alert in alerts:
                            self.database.store_alert(alert)
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logging.error(f"Error in background processing: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=process_metrics, daemon=True)
        thread.start()
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status."""
        latest_metrics = self.collector.get_latest_metrics()
        
        # Calculate performance score
        metrics_list = list(latest_metrics.values())
        performance_score = self.analyzer.calculate_performance_score(metrics_list)
        
        # Get recent alerts
        recent_alerts = self.database.get_alerts(
            datetime.now() - timedelta(minutes=5),
            datetime.now()
        )
        
        return {
            "monitoring": self.monitoring,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "performance_score": performance_score,
            "latest_metrics": {name: metric.value for name, metric in latest_metrics.items()},
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.severity == "critical"])
        }
    
    def generate_performance_report(self, duration_minutes: int = 60) -> PerformanceReport:
        """Generate a performance report for the specified duration."""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration_minutes)
        
        # Get metrics from database
        metrics = self.database.get_metrics(start_time, end_time)
        
        # Get alerts
        alerts = self.database.get_alerts(start_time, end_time)
        
        # Analyze performance
        performance_score = self.analyzer.calculate_performance_score(metrics)
        recommendations = self.analyzer.generate_recommendations(metrics, alerts)
        
        # Generate report
        report = self.reporter.generate_report(
            metrics, alerts, recommendations, performance_score, start_time, end_time
        )
        
        # Store report
        self.database.store_report(report)
        
        return report
    
    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for the specified duration."""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration_minutes)
        
        metrics = self.database.get_metrics(start_time, end_time)
        alerts = self.database.get_alerts(start_time, end_time)
        
        return {
            "duration_minutes": duration_minutes,
            "total_metrics": len(metrics),
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.severity == "critical"]),
            "warning_alerts": len([a for a in alerts if a.severity == "warning"]),
            "performance_score": self.analyzer.calculate_performance_score(metrics),
            "recommendations": self.analyzer.generate_recommendations(metrics, alerts)
        }

# Example usage and demo
def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("📊 Performance Monitoring Framework Demo")
    print("=" * 50)
    
    # Create performance monitoring framework
    framework = PerformanceMonitoringFramework(collection_interval=0.5)
    
    try:
        # Start monitoring
        print("🚀 Starting performance monitoring...")
        framework.start_monitoring()
        
        # Monitor for 30 seconds
        print("⏱️  Monitoring for 30 seconds...")
        time.sleep(30)
        
        # Get current status
        print("\n📈 Current Status:")
        status = framework.get_current_status()
        print(f"   Performance Score: {status['performance_score']:.1f}/100")
        print(f"   Uptime: {status['uptime']:.1f}s")
        print(f"   Recent Alerts: {status['recent_alerts']}")
        print(f"   Critical Alerts: {status['critical_alerts']}")
        
        # Generate report
        print("\n📊 Generating performance report...")
        report = framework.generate_performance_report(duration_minutes=1)
        print(f"   Report ID: {report.report_id}")
        print(f"   Performance Score: {report.performance_score:.1f}/100")
        print(f"   Alerts: {len(report.alerts)}")
        print(f"   Recommendations: {len(report.recommendations)}")
        
        # Print recommendations
        if report.recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"   {i}. {rec}")
        
    finally:
        # Stop monitoring
        framework.stop_monitoring()
        print("\n✅ Performance monitoring stopped")

if __name__ == "__main__":
    # Run demo
    demo_performance_monitoring()
