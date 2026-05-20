"""
Advanced Monitoring Dashboard
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Advanced system monitoring"""
    
    def __init__(self, history_size: int = 10000):
        """
        Initialize system monitor
        
        Args:
            history_size: History buffer size
        """
        self.history_size = history_size
        self.metrics = defaultdict(lambda: deque(maxlen=history_size))
        self.alerts = []
        self.health_status = "healthy"
        
        logger.info("SystemMonitor initialized")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record metric"""
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        })
    
    def get_dashboard_data(
        self,
        time_window: float = 3600.0
    ) -> Dict[str, Any]:
        """
        Get dashboard data
        
        Args:
            time_window: Time window in seconds
        
        Returns:
            Dashboard data dictionary
        """
        now = time.time()
        cutoff = now - time_window
        
        dashboard = {
            "timestamp": now,
            "health_status": self.health_status,
            "metrics": {},
            "alerts": self.get_active_alerts(),
            "summary": {}
        }
        
        # Process metrics
        for metric_name, values in self.metrics.items():
            recent_values = [
                v for v in values
                if v["timestamp"] >= cutoff
            ]
            
            if recent_values:
                metric_values = [v["value"] for v in recent_values]
                
                dashboard["metrics"][metric_name] = {
                    "count": len(metric_values),
                    "current": metric_values[-1] if metric_values else 0,
                    "average": sum(metric_values) / len(metric_values),
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "trend": self._calculate_trend(metric_values)
                }
        
        # Summary
        dashboard["summary"] = {
            "total_metrics": len(self.metrics),
            "active_alerts": len(self.get_active_alerts()),
            "time_window_seconds": time_window
        }
        
        return dashboard
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend (increasing, decreasing, stable)"""
        if len(values) < 2:
            return "stable"
        
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        diff = (second_half - first_half) / first_half if first_half > 0 else 0
        
        if diff > 0.05:
            return "increasing"
        elif diff < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        active = []
        for alert in self.alerts:
            if alert.get("active", False):
                active.append(alert)
        return active
    
    def add_alert_rule(
        self,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str = "warning"
    ):
        """Add alert rule"""
        self.alerts.append({
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "active": False
        })
    
    def check_alerts(self):
        """Check all alert rules"""
        for alert in self.alerts:
            metric_name = alert["metric_name"]
            
            if metric_name not in self.metrics:
                continue
            
            recent_values = list(self.metrics[metric_name])[-10:]
            if not recent_values:
                continue
            
            current_value = recent_values[-1]["value"]
            threshold = alert["threshold"]
            condition = alert["condition"]
            
            # Simple condition checking
            if condition == ">" and current_value > threshold:
                alert["active"] = True
            elif condition == "<" and current_value < threshold:
                alert["active"] = True
            elif condition == ">=" and current_value >= threshold:
                alert["active"] = True
            elif condition == "<=" and current_value <= threshold:
                alert["active"] = True
            else:
                alert["active"] = False


class PerformanceMonitor:
    """Performance monitoring with detailed metrics"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.operation_times = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
    
    def record_operation(
        self,
        operation_name: str,
        duration_ms: float,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """Record operation"""
        self.operation_times[operation_name].append(duration_ms)
        self.operation_counts[operation_name] += 1
        
        if not success:
            self.error_counts[operation_name] += 1
            if error_type:
                self.error_counts[f"{operation_name}_{error_type}"] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        report = {
            "operations": {},
            "summary": {
                "total_operations": sum(self.operation_counts.values()),
                "total_errors": sum(self.error_counts.values()),
                "error_rate": 0.0
            }
        }
        
        total_ops = report["summary"]["total_operations"]
        total_errors = report["summary"]["total_errors"]
        
        if total_ops > 0:
            report["summary"]["error_rate"] = total_errors / total_ops
        
        for op_name in self.operation_counts.keys():
            times = self.operation_times[op_name]
            errors = self.error_counts[op_name]
            
            if times:
                report["operations"][op_name] = {
                    "count": self.operation_counts[op_name],
                    "errors": errors,
                    "error_rate": errors / self.operation_counts[op_name] if self.operation_counts[op_name] > 0 else 0,
                    "avg_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "p95_time_ms": sorted(times)[int(len(times) * 0.95)] if times else 0,
                    "p99_time_ms": sorted(times)[int(len(times) * 0.99)] if times else 0
                }
        
        return report

