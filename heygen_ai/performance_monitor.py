#!/usr/bin/env python3
"""
HeyGen AI - Advanced Performance Monitor

This module provides comprehensive performance monitoring, profiling,
and optimization recommendations for the HeyGen AI system.
"""

import time
import psutil
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes: int = 0
    active_threads: int = 0
    active_processes: int = 0
    gpu_usage_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None

@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    timestamp: float = field(default_factory=time.time)
    alert_type: str = ""
    severity: str = "INFO"  # INFO, WARNING, CRITICAL
    message: str = ""
    metrics: Optional[PerformanceMetrics] = None
    recommendations: List[str] = field(default_factory=list)

class PerformanceMonitor:
    """Advanced performance monitoring system for HeyGen AI."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.metrics_history: List[PerformanceMetrics] = []
        self.alerts_history: List[PerformanceAlert] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        self.monitoring_interval = self.config.get('monitoring_interval', 5.0)
        
        # Performance tracking
        self.performance_baseline: Optional[PerformanceMetrics] = None
        self.performance_trends: Dict[str, List[float]] = {}
        
        # Initialize monitoring
        self._setup_monitoring()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'monitoring_interval': 5.0,  # seconds
            'metrics_history_size': 1000,
            'alert_thresholds': {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_usage_percent': 90.0,
                'gpu_usage_percent': 95.0,
                'gpu_memory_percent': 90.0
            },
            'performance_analysis': {
                'trend_analysis': True,
                'anomaly_detection': True,
                'optimization_suggestions': True
            },
            'export_formats': ['json', 'csv', 'html']
        }
    
    def _setup_monitoring(self):
        """Setup monitoring infrastructure."""
        try:
            # Check for GPU support
            self.gpu_available = self._check_gpu_availability()
            if self.gpu_available:
                logger.info("GPU monitoring enabled")
            else:
                logger.info("GPU monitoring disabled - no GPU detected")
            
            # Initialize performance tracking
            self._initialize_performance_tracking()
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking structures."""
        self.performance_trends = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_usage_percent': [],
            'gpu_usage_percent': [],
            'gpu_memory_percent': []
        }
        
        # Capture baseline metrics
        self.performance_baseline = self._capture_current_metrics()
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Capture current metrics
                metrics = self._capture_current_metrics()
                self.metrics_history.append(metrics)
                
                # Update performance trends
                self._update_performance_trends(metrics)
                
                # Check for alerts
                alerts = self._check_performance_alerts(metrics)
                if alerts:
                    self.alerts_history.extend(alerts)
                    for alert in alerts:
                        self._handle_alert(alert)
                
                # Maintain history size
                self._maintain_history_size()
                
                # Wait for next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _capture_current_metrics(self) -> PerformanceMetrics:
        """Capture current system performance metrics."""
        metrics = PerformanceMetrics()
        
        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_gb = memory.used / (1024**3)
            metrics.memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.network_io_bytes = network.bytes_sent + network.bytes_recv
            
            # Process metrics
            metrics.active_processes = len(psutil.pids())
            metrics.active_threads = threading.active_count()
            
            # GPU metrics (if available)
            if self.gpu_available:
                gpu_metrics = self._capture_gpu_metrics()
                if gpu_metrics:
                    metrics.gpu_usage_percent = gpu_metrics.get('usage_percent')
                    metrics.gpu_memory_percent = gpu_metrics.get('memory_percent')
            
        except Exception as e:
            logger.error(f"Error capturing metrics: {e}")
        
        return metrics
    
    def _capture_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """Capture GPU performance metrics."""
        try:
            import torch
            if torch.cuda.is_available():
                # Get GPU usage (this is a simplified approach)
                # In production, you might want to use nvidia-ml-py or similar
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                
                if reserved > 0:
                    memory_percent = (allocated / reserved) * 100
                else:
                    memory_percent = 0.0
                
                return {
                    'usage_percent': 0.0,  # Would need nvidia-ml-py for accurate usage
                    'memory_percent': memory_percent
                }
        except Exception as e:
            logger.error(f"Error capturing GPU metrics: {e}")
        
        return None
    
    def _update_performance_trends(self, metrics: PerformanceMetrics):
        """Update performance trend analysis."""
        for key in self.performance_trends:
            if hasattr(metrics, key):
                value = getattr(metrics, key)
                if value is not None:
                    self.performance_trends[key].append(value)
                    
                    # Keep only recent data for trend analysis
                    if len(self.performance_trends[key]) > 100:
                        self.performance_trends[key] = self.performance_trends[key][-100:]
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """Check if current metrics trigger any alerts."""
        alerts = []
        
        # CPU alerts
        if metrics.cpu_percent > self.alert_thresholds.get('cpu_percent', 80.0):
            alert = PerformanceAlert(
                alert_type="HIGH_CPU_USAGE",
                severity="WARNING" if metrics.cpu_percent < 95.0 else "CRITICAL",
                message=f"CPU usage is {metrics.cpu_percent:.1f}%",
                metrics=metrics,
                recommendations=[
                    "Check for CPU-intensive processes",
                    "Consider process optimization",
                    "Monitor system load"
                ]
            )
            alerts.append(alert)
        
        # Memory alerts
        if metrics.memory_percent > self.alert_thresholds.get('memory_percent', 85.0):
            alert = PerformanceAlert(
                alert_type="HIGH_MEMORY_USAGE",
                severity="WARNING" if metrics.memory_percent < 95.0 else "CRITICAL",
                message=f"Memory usage is {metrics.memory_percent:.1f}%",
                metrics=metrics,
                recommendations=[
                    "Check for memory leaks",
                    "Consider increasing swap space",
                    "Monitor application memory usage"
                ]
            )
            alerts.append(alert)
        
        # Disk alerts
        if metrics.disk_usage_percent > self.alert_thresholds.get('disk_usage_percent', 90.0):
            alert = PerformanceAlert(
                alert_type="HIGH_DISK_USAGE",
                severity="WARNING" if metrics.disk_usage_percent < 95.0 else "CRITICAL",
                message=f"Disk usage is {metrics.disk_usage_percent:.1f}%",
                metrics=metrics,
                recommendations=[
                    "Clean up unnecessary files",
                    "Check for large log files",
                    "Consider disk expansion"
                ]
            )
            alerts.append(alert)
        
        # GPU alerts (if available)
        if metrics.gpu_usage_percent and metrics.gpu_usage_percent > self.alert_thresholds.get('gpu_usage_percent', 95.0):
            alert = PerformanceAlert(
                alert_type="HIGH_GPU_USAGE",
                severity="WARNING",
                message=f"GPU usage is {metrics.gpu_usage_percent:.1f}%",
                metrics=metrics,
                recommendations=[
                    "Check GPU-intensive processes",
                    "Monitor GPU temperature",
                    "Consider GPU optimization"
                ]
            )
            alerts.append(alert)
        
        return alerts
    
    def _handle_alert(self, alert: PerformanceAlert):
        """Handle performance alerts."""
        log_level = logging.WARNING if alert.severity == "WARNING" else logging.ERROR
        logger.log(log_level, f"Performance Alert [{alert.alert_type}]: {alert.message}")
        
        # In production, you might want to:
        # - Send notifications (email, Slack, etc.)
        # - Trigger automated responses
        # - Log to external monitoring systems
    
    def _maintain_history_size(self):
        """Maintain metrics history size within limits."""
        max_history = self.config.get('metrics_history_size', 1000)
        
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]
        
        if len(self.alerts_history) > max_history:
            self.alerts_history = self.alerts_history[-max_history:]
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self._capture_current_metrics()
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[PerformanceMetrics]:
        """Get metrics history."""
        if limit is None:
            return self.metrics_history.copy()
        return self.metrics_history[-limit:]
    
    def get_alerts_history(self, limit: Optional[int] = None) -> List[PerformanceAlert]:
        """Get alerts history."""
        if limit is None:
            return self.alerts_history.copy()
        return self.alerts_history[-limit:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = self.metrics_history[-50:]  # Last 50 measurements
        
        summary = {
            "current_metrics": self._capture_current_metrics().__dict__,
            "baseline_metrics": self.performance_baseline.__dict__ if self.performance_baseline else None,
            "recent_averages": {},
            "trends": {},
            "alerts_summary": {
                "total_alerts": len(self.alerts_history),
                "recent_alerts": len([a for a in self.alerts_history if time.time() - a.timestamp < 3600]),
                "critical_alerts": len([a for a in self.alerts_history if a.severity == "CRITICAL"])
            }
        }
        
        # Calculate recent averages
        for key in self.performance_trends:
            if self.performance_trends[key]:
                summary["recent_averages"][key] = sum(self.performance_trends[key]) / len(self.performance_trends[key])
        
        # Calculate trends (simple linear trend)
        for key in self.performance_trends:
            if len(self.performance_trends[key]) > 10:
                values = self.performance_trends[key]
                trend = (values[-1] - values[0]) / len(values)
                summary["trends"][key] = {
                    "slope": trend,
                    "direction": "increasing" if trend > 0.01 else "decreasing" if trend < -0.01 else "stable"
                }
        
        return summary
    
    def export_metrics(self, format: str = "json", filepath: Optional[str] = None) -> str:
        """Export metrics to various formats."""
        if format not in self.config.get('export_formats', ['json']):
            raise ValueError(f"Unsupported format: {format}")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"heygen_ai_performance_{timestamp}.{format}"
        
        if format == "json":
            return self._export_json(filepath)
        elif format == "csv":
            return self._export_csv(filepath)
        elif format == "html":
            return self._export_html(filepath)
        
        raise ValueError(f"Export format {format} not implemented")
    
    def _export_json(self, filepath: str) -> str:
        """Export metrics to JSON format."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics_history": [m.__dict__ for m in self.metrics_history],
            "alerts_history": [a.__dict__ for a in self.alerts_history],
            "performance_summary": self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def _export_csv(self, filepath: str) -> str:
        """Export metrics to CSV format."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            if self.metrics_history:
                writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].__dict__.keys())
                writer.writeheader()
                for metrics in self.metrics_history:
                    writer.writerow(metrics.__dict__)
        
        return filepath
    
    def _export_html(self, filepath: str) -> str:
        """Export metrics to HTML format."""
        html_content = self._generate_html_report()
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return filepath
    
    def _generate_html_report(self) -> str:
        """Generate HTML performance report."""
        summary = self.get_performance_summary()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HeyGen AI Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ margin: 20px 0; }}
                .alert {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ff6b6b; background: #fff5f5; }}
                .warning {{ border-left-color: #ffd93d; background: #fffbf0; }}
                .info {{ border-left-color: #4ecdc4; background: #f0fffd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 HeyGen AI Performance Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="metrics">
                <h2>📊 Current Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        current = summary.get("current_metrics", {})
        for key, value in current.items():
            if value is not None:
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += """
                </table>
            </div>
            
            <div class="metrics">
                <h2>📈 Performance Trends</h2>
                <table>
                    <tr><th>Metric</th><th>Trend</th><th>Direction</th></tr>
        """
        
        trends = summary.get("trends", {})
        for key, trend_data in trends.items():
            html += f"<tr><td>{key}</td><td>{trend_data.get('slope', 0):.3f}</td><td>{trend_data.get('direction', 'unknown')}</td></tr>"
        
        html += """
                </table>
            </div>
            
            <div class="metrics">
                <h2>⚠️ Alerts Summary</h2>
                <p>Total Alerts: {total}</p>
                <p>Recent Alerts (1 hour): {recent}</p>
                <p>Critical Alerts: {critical}</p>
            </div>
        </body>
        </html>
        """.format(
            total=summary.get("alerts_summary", {}).get("total_alerts", 0),
            recent=summary.get("alerts_summary", {}).get("recent_alerts", 0),
            critical=summary.get("alerts_summary", {}).get("critical_alerts", 0)
        )
        
        return html
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        summary = self.get_performance_summary()
        
        # CPU recommendations
        current_cpu = summary.get("current_metrics", {}).get("cpu_percent", 0)
        if current_cpu > 70:
            recommendations.append("Consider CPU optimization or scaling")
        
        # Memory recommendations
        current_memory = summary.get("current_metrics", {}).get("memory_percent", 0)
        if current_memory > 80:
            recommendations.append("Monitor memory usage and consider optimization")
        
        # Disk recommendations
        current_disk = summary.get("current_metrics", {}).get("disk_usage_percent", 0)
        if current_disk > 85:
            recommendations.append("Clean up disk space and monitor storage")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("System performance is within normal ranges")
        
        return recommendations

def main():
    """Demo function for the performance monitor."""
    print("🚀 HeyGen AI Performance Monitor Demo")
    print("=" * 50)
    
    # Create monitor instance
    monitor = PerformanceMonitor()
    
    try:
        # Start monitoring
        print("Starting performance monitoring...")
        monitor.start_monitoring()
        
        # Let it run for a bit
        print("Monitoring for 30 seconds...")
        time.sleep(30)
        
        # Get performance summary
        print("\n📊 Performance Summary:")
        summary = monitor.get_performance_summary()
        print(json.dumps(summary, indent=2, default=str))
        
        # Get optimization recommendations
        print("\n💡 Optimization Recommendations:")
        recommendations = monitor.get_optimization_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Export metrics
        print("\n📁 Exporting metrics...")
        json_file = monitor.export_metrics("json")
        print(f"JSON export: {json_file}")
        
        html_file = monitor.export_metrics("html")
        print(f"HTML export: {html_file}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("\n✅ Performance monitoring demo completed!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
