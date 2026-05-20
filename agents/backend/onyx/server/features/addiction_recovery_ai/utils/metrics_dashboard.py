"""
Metrics Dashboard for Real-time Monitoring
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class MetricsDashboard:
    """Real-time metrics dashboard"""
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize metrics dashboard
        
        Args:
            history_size: Size of history buffer
        """
        self.history_size = history_size
        self.metrics = defaultdict(lambda: deque(maxlen=history_size))
        self.alerts = []
        
        logger.info("MetricsDashboard initialized")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record metric
        
        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Optional timestamp
            tags: Optional tags
        """
        if timestamp is None:
            timestamp = time.time()
        
        metric_entry = {
            "value": value,
            "timestamp": timestamp,
            "tags": tags or {}
        }
        
        self.metrics[metric_name].append(metric_entry)
    
    def get_metric_stats(
        self,
        metric_name: str,
        window: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get metric statistics
        
        Args:
            metric_name: Metric name
            window: Optional time window in seconds
        
        Returns:
            Statistics dictionary
        """
        if metric_name not in self.metrics:
            return {}
        
        metric_data = list(self.metrics[metric_name])
        
        # Filter by window if specified
        if window:
            cutoff = time.time() - window
            metric_data = [m for m in metric_data if m["timestamp"] >= cutoff]
        
        if not metric_data:
            return {}
        
        values = [m["value"] for m in metric_data]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1],
            "timestamp": metric_data[-1]["timestamp"]
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics statistics"""
        return {
            name: self.get_metric_stats(name)
            for name in self.metrics.keys()
        }
    
    def add_alert(
        self,
        condition: str,
        threshold: float,
        severity: str = "warning"
    ):
        """
        Add alert condition
        
        Args:
            condition: Condition expression (e.g., "latency > 100")
            threshold: Threshold value
            severity: Alert severity (info, warning, error, critical)
        """
        self.alerts.append({
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "active": False
        })
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alerts and return active ones"""
        active_alerts = []
        
        for alert in self.alerts:
            # Simple threshold checking
            # In practice, you'd parse the condition
            metric_name = alert["condition"].split()[0]
            threshold = alert["threshold"]
            
            stats = self.get_metric_stats(metric_name)
            if stats and stats.get("latest", 0) > threshold:
                alert["active"] = True
                active_alerts.append(alert)
            else:
                alert["active"] = False
        
        return active_alerts
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """
        Export metrics to file
        
        Args:
            filepath: Output file path
            format: Export format (json, csv)
        """
        if format == "json":
            data = {
                "metrics": {
                    name: list(entries)
                    for name, entries in self.metrics.items()
                },
                "alerts": self.alerts,
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "csv":
            import csv
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value", "timestamp", "tags"])
                
                for name, entries in self.metrics.items():
                    for entry in entries:
                        writer.writerow([
                            name,
                            entry["value"],
                            entry["timestamp"],
                            json.dumps(entry.get("tags", {}))
                        ])
        
        logger.info(f"Metrics exported to {filepath}")


class PerformanceTracker:
    """Track performance metrics over time"""
    
    def __init__(self):
        """Initialize performance tracker"""
        self.operations = defaultdict(list)
        self.start_times = {}
    
    def start_operation(self, operation_name: str):
        """Start tracking operation"""
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name: str) -> float:
        """
        End tracking operation
        
        Args:
            operation_name: Operation name
        
        Returns:
            Elapsed time in milliseconds
        """
        if operation_name not in self.start_times:
            return 0.0
        
        elapsed = (time.time() - self.start_times[operation_name]) * 1000
        self.operations[operation_name].append(elapsed)
        del self.start_times[operation_name]
        
        return elapsed
    
    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for operation"""
        if operation_name not in self.operations:
            return {}
        
        times = self.operations[operation_name]
        
        return {
            "count": len(times),
            "total_ms": sum(times),
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "p95_ms": sorted(times)[int(len(times) * 0.95)] if times else 0,
            "p99_ms": sorted(times)[int(len(times) * 0.99)] if times else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations"""
        return {
            name: self.get_stats(name)
            for name in self.operations.keys()
        }

