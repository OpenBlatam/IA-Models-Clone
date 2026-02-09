"""
Advanced Metrics System
Comprehensive metrics collection and analysis
"""

from typing import Dict, Any, Optional, List
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Advanced metrics collection system
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.aggregations: Dict[str, Dict[str, Any]] = {}
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Record a metric"""
        metric_data = {
            "value": value,
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "tags": tags or {}
        }
        
        self.metrics[metric_name].append(metric_data)
        
        # Update aggregations
        self._update_aggregations(metric_name, value)
    
    def _update_aggregations(self, metric_name: str, value: float):
        """Update metric aggregations"""
        if metric_name not in self.aggregations:
            self.aggregations[metric_name] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "values": []
            }
        
        agg = self.aggregations[metric_name]
        agg["count"] += 1
        agg["sum"] += value
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)
        agg["values"].append(value)
        
        # Keep only last 1000 values
        if len(agg["values"]) > 1000:
            agg["values"] = agg["values"][-1000:]
    
    def get_metric_stats(
        self,
        metric_name: str,
        window_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get metric statistics"""
        if metric_name not in self.metrics:
            return {}
        
        metrics = self.metrics[metric_name]
        
        # Filter by time window
        if window_seconds:
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            metrics = [
                m for m in metrics
                if datetime.fromisoformat(m["timestamp"]) >= cutoff
            ]
        
        if not metrics:
            return {}
        
        values = [m["value"] for m in metrics]
        
        return {
            "metric": metric_name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metric statistics"""
        return {
            name: self.get_metric_stats(name)
            for name in self.metrics.keys()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics"""
        if format == "json":
            import json
            return json.dumps(self.get_all_metrics(), indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")


class PerformanceMetrics:
    """
    Performance-specific metrics
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_inference_metrics(
        self,
        model_name: str,
        latency: float,
        throughput: float,
        memory_usage: Optional[float] = None
    ):
        """Record inference performance metrics"""
        self.collector.record_metric(
            f"{model_name}.latency",
            latency,
            tags={"model": model_name, "type": "inference"}
        )
        
        self.collector.record_metric(
            f"{model_name}.throughput",
            throughput,
            tags={"model": model_name, "type": "inference"}
        )
        
        if memory_usage:
            self.collector.record_metric(
                f"{model_name}.memory",
                memory_usage,
                tags={"model": model_name, "type": "inference"}
            )
    
    def record_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None
    ):
        """Record training metrics"""
        self.collector.record_metric(
            "training.train_loss",
            train_loss,
            tags={"epoch": str(epoch), "type": "training"}
        )
        
        self.collector.record_metric(
            "training.val_loss",
            val_loss,
            tags={"epoch": str(epoch), "type": "training"}
        )
        
        if train_acc is not None:
            self.collector.record_metric(
                "training.train_acc",
                train_acc,
                tags={"epoch": str(epoch), "type": "training"}
            )
        
        if val_acc is not None:
            self.collector.record_metric(
                "training.val_acc",
                val_acc,
                tags={"epoch": str(epoch), "type": "training"}
            )

