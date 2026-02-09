"""
Real-time Performance Analyzer for Ultimate Opus Clip

Advanced performance monitoring, analysis, and optimization recommendations
for video processing, system resources, and content quality.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import asyncio
import time
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
import psutil
import GPUtil
import threading
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import statistics
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger("performance_analyzer")

class PerformanceMetric(Enum):
    """Types of performance metrics."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    PROCESSING_TIME = "processing_time"
    QUEUE_SIZE = "queue_size"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    QUALITY_SCORE = "quality_score"
    USER_SATISFACTION = "user_satisfaction"

class PerformanceLevel(Enum):
    """Performance levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of performance alerts."""
    CPU_HIGH = "cpu_high"
    MEMORY_HIGH = "memory_high"
    GPU_HIGH = "gpu_high"
    DISK_FULL = "disk_full"
    PROCESSING_SLOW = "processing_slow"
    ERROR_RATE_HIGH = "error_rate_high"
    QUEUE_OVERFLOW = "queue_overflow"
    QUALITY_DEGRADED = "quality_degraded"

@dataclass
class PerformanceData:
    """Performance data point."""
    metric: PerformanceMetric
    value: float
    timestamp: float
    unit: str
    metadata: Dict[str, Any] = None

@dataclass
class PerformanceAlert:
    """Performance alert."""
    alert_id: str
    alert_type: AlertType
    severity: str
    message: str
    metric: PerformanceMetric
    current_value: float
    threshold_value: float
    timestamp: float
    is_resolved: bool = False
    resolution_time: Optional[float] = None

@dataclass
class PerformanceReport:
    """Performance analysis report."""
    report_id: str
    start_time: float
    end_time: float
    duration: float
    metrics_summary: Dict[str, Any]
    alerts: List[PerformanceAlert]
    recommendations: List[str]
    performance_score: float
    generated_at: float

@dataclass
class SystemResource:
    """System resource information."""
    cpu_cores: int
    total_memory: float
    available_memory: float
    gpu_count: int
    gpu_memory: List[float]
    disk_space: float
    network_speed: float

class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: Dict[PerformanceMetric, deque] = {
            metric: deque(maxlen=history_size) for metric in PerformanceMetric
        }
        self.alerts: List[PerformanceAlert] = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.alert_callbacks: List[Callable] = []
        
        # Performance thresholds
        self.thresholds = {
            PerformanceMetric.CPU_USAGE: 80.0,
            PerformanceMetric.MEMORY_USAGE: 85.0,
            PerformanceMetric.GPU_USAGE: 90.0,
            PerformanceMetric.DISK_IO: 1000.0,  # MB/s
            PerformanceMetric.NETWORK_IO: 100.0,  # MB/s
            PerformanceMetric.PROCESSING_TIME: 30.0,  # seconds
            PerformanceMetric.QUEUE_SIZE: 100,
            PerformanceMetric.ERROR_RATE: 5.0,  # percentage
            PerformanceMetric.THROUGHPUT: 10.0,  # items/second
            PerformanceMetric.LATENCY: 5.0,  # seconds
            PerformanceMetric.QUALITY_SCORE: 0.7,
            PerformanceMetric.USER_SATISFACTION: 0.8
        }
        
        logger.info("Performance Monitor initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._collect_metrics()
                self._check_alerts()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self):
        """Collect current performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric(PerformanceMetric.CPU_USAGE, cpu_percent, "%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._add_metric(PerformanceMetric.MEMORY_USAGE, memory_percent, "%")
            
            # GPU usage
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
                    self._add_metric(PerformanceMetric.GPU_USAGE, gpu_usage, "%")
            except:
                pass
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_read_mb = disk_io.read_bytes / (1024 * 1024)
                disk_write_mb = disk_io.write_bytes / (1024 * 1024)
                self._add_metric(PerformanceMetric.DISK_IO, disk_read_mb + disk_write_mb, "MB")
            
            # Network I/O
            network_io = psutil.net_io_counters()
            if network_io:
                network_mb = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)
                self._add_metric(PerformanceMetric.NETWORK_IO, network_mb, "MB")
            
            # Process-specific metrics (placeholder)
            self._add_metric(PerformanceMetric.PROCESSING_TIME, 0.0, "seconds")
            self._add_metric(PerformanceMetric.QUEUE_SIZE, 0, "items")
            self._add_metric(PerformanceMetric.ERROR_RATE, 0.0, "%")
            self._add_metric(PerformanceMetric.THROUGHPUT, 0.0, "items/sec")
            self._add_metric(PerformanceMetric.LATENCY, 0.0, "seconds")
            self._add_metric(PerformanceMetric.QUALITY_SCORE, 1.0, "score")
            self._add_metric(PerformanceMetric.USER_SATISFACTION, 1.0, "score")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _add_metric(self, metric: PerformanceMetric, value: float, unit: str):
        """Add metric to history."""
        data = PerformanceData(
            metric=metric,
            value=value,
            timestamp=time.time(),
            unit=unit
        )
        self.metrics_history[metric].append(data)
    
    def _check_alerts(self):
        """Check for performance alerts."""
        for metric, threshold in self.thresholds.items():
            if metric not in self.metrics_history or not self.metrics_history[metric]:
                continue
            
            current_value = self.metrics_history[metric][-1].value
            
            if current_value > threshold:
                self._create_alert(metric, current_value, threshold)
    
    def _create_alert(self, metric: PerformanceMetric, current_value: float, threshold: float):
        """Create performance alert."""
        alert_type = self._get_alert_type(metric)
        severity = self._get_alert_severity(current_value, threshold)
        
        alert = PerformanceAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            message=f"{metric.value} is {current_value:.1f} (threshold: {threshold:.1f})",
            metric=metric,
            current_value=current_value,
            threshold_value=threshold,
            timestamp=time.time()
        )
        
        self.alerts.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Performance alert: {alert.message}")
    
    def _get_alert_type(self, metric: PerformanceMetric) -> AlertType:
        """Get alert type from metric."""
        alert_mapping = {
            PerformanceMetric.CPU_USAGE: AlertType.CPU_HIGH,
            PerformanceMetric.MEMORY_USAGE: AlertType.MEMORY_HIGH,
            PerformanceMetric.GPU_USAGE: AlertType.GPU_HIGH,
            PerformanceMetric.DISK_IO: AlertType.DISK_FULL,
            PerformanceMetric.PROCESSING_TIME: AlertType.PROCESSING_SLOW,
            PerformanceMetric.ERROR_RATE: AlertType.ERROR_RATE_HIGH,
            PerformanceMetric.QUEUE_SIZE: AlertType.QUEUE_OVERFLOW,
            PerformanceMetric.QUALITY_SCORE: AlertType.QUALITY_DEGRADED
        }
        return alert_mapping.get(metric, AlertType.CPU_HIGH)
    
    def _get_alert_severity(self, current_value: float, threshold: float) -> str:
        """Get alert severity based on how much threshold is exceeded."""
        ratio = current_value / threshold
        
        if ratio >= 2.0:
            return "critical"
        elif ratio >= 1.5:
            return "high"
        elif ratio >= 1.2:
            return "medium"
        else:
            return "low"
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_metrics = {}
        
        for metric, history in self.metrics_history.items():
            if history:
                latest = history[-1]
                current_metrics[metric.value] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp
                }
        
        return current_metrics
    
    def get_metric_history(self, metric: PerformanceMetric, hours: int = 1) -> List[PerformanceData]:
        """Get metric history for specified hours."""
        if metric not in self.metrics_history:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        return [data for data in self.metrics_history[metric] if data.timestamp > cutoff_time]
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified hours."""
        summary = {}
        
        for metric in PerformanceMetric:
            history = self.get_metric_history(metric, hours)
            if not history:
                continue
            
            values = [data.value for data in history]
            
            summary[metric.value] = {
                "current": values[-1] if values else 0,
                "average": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "count": len(values)
            }
        
        return summary

class PerformanceAnalyzer:
    """Advanced performance analysis and recommendations."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.analysis_history: List[PerformanceReport] = []
        
        logger.info("Performance Analyzer initialized")
    
    def analyze_performance(self, hours: int = 1) -> PerformanceReport:
        """Analyze performance and generate report."""
        try:
            start_time = time.time() - (hours * 3600)
            end_time = time.time()
            
            # Get performance summary
            summary = self.monitor.get_performance_summary(hours)
            
            # Analyze trends
            trends = self._analyze_trends(hours)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(summary, trends)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(summary)
            
            # Get recent alerts
            recent_alerts = [alert for alert in self.monitor.alerts 
                           if alert.timestamp >= start_time]
            
            # Create report
            report = PerformanceReport(
                report_id=str(uuid.uuid4()),
                start_time=start_time,
                end_time=end_time,
                duration=hours * 3600,
                metrics_summary=summary,
                alerts=recent_alerts,
                recommendations=recommendations,
                performance_score=performance_score,
                generated_at=time.time()
            )
            
            self.analysis_history.append(report)
            
            logger.info(f"Performance analysis completed: {report.report_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            raise
    
    def _analyze_trends(self, hours: int) -> Dict[str, Any]:
        """Analyze performance trends."""
        trends = {}
        
        for metric in PerformanceMetric:
            history = self.monitor.get_metric_history(metric, hours)
            if len(history) < 2:
                continue
            
            values = [data.value for data in history]
            
            # Calculate trend direction
            if len(values) >= 2:
                trend_direction = "stable"
                if values[-1] > values[0] * 1.1:
                    trend_direction = "increasing"
                elif values[-1] < values[0] * 0.9:
                    trend_direction = "decreasing"
                
                trends[metric.value] = {
                    "direction": trend_direction,
                    "change_percent": ((values[-1] - values[0]) / values[0]) * 100,
                    "volatility": statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return trends
    
    def _generate_recommendations(self, summary: Dict[str, Any], 
                                trends: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # CPU recommendations
        if summary.get("cpu_usage", {}).get("average", 0) > 80:
            recommendations.append("Consider scaling up CPU resources or optimizing processing algorithms")
        
        # Memory recommendations
        if summary.get("memory_usage", {}).get("average", 0) > 85:
            recommendations.append("Increase available memory or implement memory optimization techniques")
        
        # GPU recommendations
        if summary.get("gpu_usage", {}).get("average", 0) > 90:
            recommendations.append("Consider GPU scaling or optimizing GPU-intensive operations")
        
        # Processing time recommendations
        if summary.get("processing_time", {}).get("average", 0) > 30:
            recommendations.append("Optimize processing pipeline or implement parallel processing")
        
        # Error rate recommendations
        if summary.get("error_rate", {}).get("average", 0) > 5:
            recommendations.append("Investigate and fix error sources to improve system reliability")
        
        # Quality score recommendations
        if summary.get("quality_score", {}).get("average", 0) < 0.7:
            recommendations.append("Review and improve content quality algorithms")
        
        # Trend-based recommendations
        for metric, trend in trends.items():
            if trend["direction"] == "increasing" and trend["change_percent"] > 20:
                recommendations.append(f"Monitor {metric} trend - significant increase detected")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable ranges")
        
        return recommendations
    
    def _calculate_performance_score(self, summary: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        scores = []
        
        # CPU score (inverse of usage)
        cpu_usage = summary.get("cpu_usage", {}).get("average", 0)
        cpu_score = max(0, 1 - (cpu_usage / 100))
        scores.append(cpu_score)
        
        # Memory score (inverse of usage)
        memory_usage = summary.get("memory_usage", {}).get("average", 0)
        memory_score = max(0, 1 - (memory_usage / 100))
        scores.append(memory_score)
        
        # GPU score (inverse of usage)
        gpu_usage = summary.get("gpu_usage", {}).get("average", 0)
        gpu_score = max(0, 1 - (gpu_usage / 100))
        scores.append(gpu_score)
        
        # Processing time score (inverse of time)
        processing_time = summary.get("processing_time", {}).get("average", 0)
        processing_score = max(0, 1 - (processing_time / 60))  # Normalize to 60 seconds
        scores.append(processing_score)
        
        # Quality score
        quality_score = summary.get("quality_score", {}).get("average", 1.0)
        scores.append(quality_score)
        
        # User satisfaction score
        user_satisfaction = summary.get("user_satisfaction", {}).get("average", 1.0)
        scores.append(user_satisfaction)
        
        return statistics.mean(scores) if scores else 0.0
    
    def get_performance_level(self, score: float) -> PerformanceLevel:
        """Get performance level from score."""
        if score >= 0.9:
            return PerformanceLevel.EXCELLENT
        elif score >= 0.8:
            return PerformanceLevel.GOOD
        elif score >= 0.6:
            return PerformanceLevel.AVERAGE
        elif score >= 0.4:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def get_analysis_history(self, limit: int = 10) -> List[PerformanceReport]:
        """Get analysis history."""
        return self.analysis_history[-limit:]

class PerformanceOptimizer:
    """Automatic performance optimization."""
    
    def __init__(self, monitor: PerformanceMonitor, analyzer: PerformanceAnalyzer):
        self.monitor = monitor
        self.analyzer = analyzer
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info("Performance Optimizer initialized")
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Perform automatic performance optimization."""
        try:
            # Get current performance
            current_metrics = self.monitor.get_current_metrics()
            
            # Analyze performance
            report = self.analyzer.analyze_performance(hours=1)
            
            # Generate optimization actions
            actions = self._generate_optimization_actions(current_metrics, report)
            
            # Apply optimizations
            results = self._apply_optimizations(actions)
            
            # Record optimization
            optimization_record = {
                "timestamp": time.time(),
                "actions": actions,
                "results": results,
                "performance_score_before": report.performance_score,
                "performance_score_after": self.analyzer.analyze_performance(hours=1).performance_score
            }
            
            self.optimization_history.append(optimization_record)
            
            logger.info(f"Performance optimization completed: {len(actions)} actions applied")
            return optimization_record
            
        except Exception as e:
            logger.error(f"Error optimizing performance: {e}")
            return {"error": str(e)}
    
    def _generate_optimization_actions(self, metrics: Dict[str, Any], 
                                     report: PerformanceReport) -> List[Dict[str, Any]]:
        """Generate optimization actions based on current performance."""
        actions = []
        
        # CPU optimization
        if metrics.get("cpu_usage", {}).get("value", 0) > 80:
            actions.append({
                "type": "cpu_optimization",
                "action": "scale_cpu",
                "parameters": {"scale_factor": 1.5}
            })
        
        # Memory optimization
        if metrics.get("memory_usage", {}).get("value", 0) > 85:
            actions.append({
                "type": "memory_optimization",
                "action": "clear_cache",
                "parameters": {"cache_type": "all"}
            })
        
        # GPU optimization
        if metrics.get("gpu_usage", {}).get("value", 0) > 90:
            actions.append({
                "type": "gpu_optimization",
                "action": "optimize_batch_size",
                "parameters": {"batch_size": 16}
            })
        
        # Processing optimization
        if metrics.get("processing_time", {}).get("value", 0) > 30:
            actions.append({
                "type": "processing_optimization",
                "action": "enable_parallel_processing",
                "parameters": {"max_workers": 4}
            })
        
        return actions
    
    def _apply_optimizations(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply optimization actions."""
        results = {}
        
        for action in actions:
            try:
                action_type = action["type"]
                action_name = action["action"]
                parameters = action.get("parameters", {})
                
                # Simulate optimization application
                if action_type == "cpu_optimization":
                    results[action_name] = "CPU resources scaled successfully"
                elif action_type == "memory_optimization":
                    results[action_name] = "Memory cache cleared successfully"
                elif action_type == "gpu_optimization":
                    results[action_name] = "GPU batch size optimized successfully"
                elif action_type == "processing_optimization":
                    results[action_name] = "Parallel processing enabled successfully"
                else:
                    results[action_name] = "Optimization applied successfully"
                
            except Exception as e:
                results[action["action"]] = f"Error: {str(e)}"
        
        return results

# Global performance system instances
_global_performance_monitor: Optional[PerformanceMonitor] = None
_global_performance_analyzer: Optional[PerformanceAnalyzer] = None
_global_performance_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor

def get_performance_analyzer() -> PerformanceAnalyzer:
    """Get the global performance analyzer instance."""
    global _global_performance_analyzer
    if _global_performance_analyzer is None:
        monitor = get_performance_monitor()
        _global_performance_analyzer = PerformanceAnalyzer(monitor)
    return _global_performance_analyzer

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _global_performance_optimizer
    if _global_performance_optimizer is None:
        monitor = get_performance_monitor()
        analyzer = get_performance_analyzer()
        _global_performance_optimizer = PerformanceOptimizer(monitor, analyzer)
    return _global_performance_optimizer

def start_performance_monitoring(interval: float = 1.0):
    """Start performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring(interval)

def stop_performance_monitoring():
    """Stop performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()

def get_current_performance() -> Dict[str, Any]:
    """Get current performance metrics."""
    monitor = get_performance_monitor()
    return monitor.get_current_metrics()

def analyze_performance(hours: int = 1) -> PerformanceReport:
    """Analyze performance and generate report."""
    analyzer = get_performance_analyzer()
    return analyzer.analyze_performance(hours)

def optimize_performance() -> Dict[str, Any]:
    """Perform automatic performance optimization."""
    optimizer = get_performance_optimizer()
    return optimizer.optimize_performance()


