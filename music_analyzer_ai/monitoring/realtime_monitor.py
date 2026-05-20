"""
Real-time Monitoring System
Monitor model performance, system health, and metrics in real-time
"""

from typing import Dict, Any, Optional, List, Callable
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Metric data point"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = None


class RealTimeMonitor:
    """
    Real-time monitoring system for models and system
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
        self.running = True
        self.lock = threading.Lock()
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.window_size)
            
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def get_metric_stats(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get statistics for a metric"""
        with self.lock:
            if name not in self.metrics:
                return {}
            
            metrics = list(self.metrics[name])
            
            # Filter by time window
            if window_seconds:
                cutoff = time.time() - window_seconds
                metrics = [m for m in metrics if m.timestamp >= cutoff]
            
            if not metrics:
                return {}
            
            values = [m.value for m in metrics]
            
            return {
                "name": name,
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std": (sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5,
                "latest": values[-1],
                "window_seconds": window_seconds
            }
    
    def register_alert(
        self,
        metric_name: str,
        condition: Callable[[float], bool],
        message: str,
        severity: str = "warning"  # "info", "warning", "error", "critical"
    ):
        """Register an alert condition"""
        alert = {
            "metric_name": metric_name,
            "condition": condition,
            "message": message,
            "severity": severity,
            "triggered": False
        }
        self.alerts.append(alert)
    
    def check_alerts(self):
        """Check all registered alerts"""
        with self.lock:
            for alert in self.alerts:
                if alert["metric_name"] in self.metrics:
                    metrics = list(self.metrics[alert["metric_name"]])
                    if metrics:
                        latest_value = metrics[-1].value
                        if alert["condition"](latest_value) and not alert["triggered"]:
                            alert["triggered"] = True
                            self._trigger_alert(alert, latest_value)
                        elif not alert["condition"](latest_value):
                            alert["triggered"] = False
    
    def _trigger_alert(self, alert: Dict[str, Any], value: float):
        """Trigger an alert"""
        alert_data = {
            "metric": alert["metric_name"],
            "value": value,
            "message": alert["message"],
            "severity": alert["severity"],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(f"Alert triggered: {alert_data}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {str(e)}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metric statistics"""
        with self.lock:
            return {
                name: self.get_metric_stats(name)
                for name in self.metrics.keys()
            }
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring"""
        def monitor_loop():
            while self.running:
                self.check_alerts()
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False


class ModelPerformanceMonitor:
    """
    Monitor model performance metrics
    """
    
    def __init__(self, monitor: RealTimeMonitor):
        self.monitor = monitor
    
    def record_inference(
        self,
        model_name: str,
        latency: float,
        success: bool,
        input_size: Optional[int] = None
    ):
        """Record inference metrics"""
        self.monitor.record_metric(
            f"{model_name}.latency",
            latency,
            tags={"model": model_name}
        )
        
        self.monitor.record_metric(
            f"{model_name}.success_rate",
            1.0 if success else 0.0,
            tags={"model": model_name}
        )
        
        if input_size:
            self.monitor.record_metric(
                f"{model_name}.throughput",
                input_size / latency if latency > 0 else 0,
                tags={"model": model_name}
            )
    
    def record_accuracy(
        self,
        model_name: str,
        accuracy: float,
        dataset: str = "validation"
    ):
        """Record accuracy metric"""
        self.monitor.record_metric(
            f"{model_name}.accuracy.{dataset}",
            accuracy,
            tags={"model": model_name, "dataset": dataset}
        )
    
    def setup_default_alerts(self, model_name: str):
        """Setup default alerts for a model"""
        # High latency alert
        self.monitor.register_alert(
            f"{model_name}.latency",
            lambda x: x > 1.0,  # > 1 second
            f"High latency detected for {model_name}",
            severity="warning"
        )
        
        # Low success rate alert
        self.monitor.register_alert(
            f"{model_name}.success_rate",
            lambda x: x < 0.95,  # < 95%
            f"Low success rate for {model_name}",
            severity="error"
        )
        
        # Low accuracy alert
        self.monitor.register_alert(
            f"{model_name}.accuracy.validation",
            lambda x: x < 0.7,  # < 70%
            f"Low accuracy for {model_name}",
            severity="warning"
        )

