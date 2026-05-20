"""
Cache anomaly detection.

Provides anomaly detection for cache operations.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Anomaly types."""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    ACCESS_PATTERN = "access_pattern"
    ERROR_RATE = "error_rate"


@dataclass
class Anomaly:
    """Anomaly detection result."""
    type: AnomalyType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    timestamp: float
    metrics: Dict[str, Any]
    recommendation: str


class CacheAnomalyDetector:
    """
    Cache anomaly detector.
    
    Detects anomalies in cache behavior.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize detector.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.metrics_history: Dict[str, deque] = {}
        self.detected_anomalies: List[Anomaly] = []
        self.baseline: Dict[str, float] = {}
    
    def record_metric(self, name: str, value: float) -> None:
        """
        Record metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics_history:
            self.metrics_history[name] = deque(maxlen=1000)
        
        self.metrics_history[name].append({
            "value": value,
            "timestamp": time.time()
        })
    
    def calculate_baseline(self, metric_name: str) -> float:
        """
        Calculate baseline for metric.
        
        Args:
            metric_name: Metric name
            
        Returns:
            Baseline value
        """
        if metric_name not in self.metrics_history:
            return 0.0
        
        values = [m["value"] for m in self.metrics_history[metric_name]]
        
        if not values:
            return 0.0
        
        # Calculate mean as baseline
        baseline = sum(values) / len(values)
        self.baseline[metric_name] = baseline
        
        return baseline
    
    def detect_anomalies(self) -> List[Anomaly]:
        """
        Detect anomalies.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check performance anomalies
        perf_anomalies = self._detect_performance_anomalies()
        anomalies.extend(perf_anomalies)
        
        # Check memory anomalies
        mem_anomalies = self._detect_memory_anomalies()
        anomalies.extend(mem_anomalies)
        
        # Check access pattern anomalies
        access_anomalies = self._detect_access_pattern_anomalies()
        anomalies.extend(access_anomalies)
        
        # Check error rate anomalies
        error_anomalies = self._detect_error_rate_anomalies()
        anomalies.extend(error_anomalies)
        
        self.detected_anomalies.extend(anomalies)
        
        return anomalies
    
    def _detect_performance_anomalies(self) -> List[Anomaly]:
        """Detect performance anomalies."""
        anomalies = []
        
        stats = self.cache.get_stats()
        avg_latency = stats.get("avg_latency_ms", 0.0)
        
        # Record metric
        self.record_metric("avg_latency_ms", avg_latency)
        
        # Calculate baseline
        baseline = self.calculate_baseline("avg_latency_ms")
        
        # Detect if latency is significantly higher
        if baseline > 0 and avg_latency > baseline * 2:
            severity = "high" if avg_latency > baseline * 3 else "medium"
            
            anomalies.append(Anomaly(
                type=AnomalyType.PERFORMANCE,
                severity=severity,
                description=f"High latency detected: {avg_latency:.2f}ms (baseline: {baseline:.2f}ms)",
                timestamp=time.time(),
                metrics={"avg_latency_ms": avg_latency, "baseline": baseline},
                recommendation="Check cache configuration and system load"
            ))
        
        return anomalies
    
    def _detect_memory_anomalies(self) -> List[Anomaly]:
        """Detect memory anomalies."""
        anomalies = []
        
        stats = self.cache.get_stats()
        memory_mb = stats.get("memory_mb", 0.0)
        
        # Record metric
        self.record_metric("memory_mb", memory_mb)
        
        # Calculate baseline
        baseline = self.calculate_baseline("memory_mb")
        
        # Detect if memory usage is significantly higher
        if baseline > 0 and memory_mb > baseline * 1.5:
            severity = "critical" if memory_mb > baseline * 2 else "high"
            
            anomalies.append(Anomaly(
                type=AnomalyType.MEMORY,
                severity=severity,
                description=f"High memory usage: {memory_mb:.2f}MB (baseline: {baseline:.2f}MB)",
                timestamp=time.time(),
                metrics={"memory_mb": memory_mb, "baseline": baseline},
                recommendation="Enable compression or reduce cache size"
            ))
        
        return anomalies
    
    def _detect_access_pattern_anomalies(self) -> List[Anomaly]:
        """Detect access pattern anomalies."""
        anomalies = []
        
        stats = self.cache.get_stats()
        hit_rate = stats.get("hit_rate", 0.0)
        
        # Record metric
        self.record_metric("hit_rate", hit_rate)
        
        # Detect if hit rate is unusually low
        if hit_rate < 0.5:
            severity = "critical" if hit_rate < 0.3 else "high"
            
            anomalies.append(Anomaly(
                type=AnomalyType.ACCESS_PATTERN,
                severity=severity,
                description=f"Low hit rate: {hit_rate:.2%}",
                timestamp=time.time(),
                metrics={"hit_rate": hit_rate},
                recommendation="Increase cache size or adjust cache strategy"
            ))
        
        return anomalies
    
    def _detect_error_rate_anomalies(self) -> List[Anomaly]:
        """Detect error rate anomalies."""
        anomalies = []
        
        # In production: would track actual errors
        # For now: placeholder
        return anomalies
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get anomaly summary.
        
        Returns:
            Anomaly summary
        """
        summary = {
            "total_anomalies": len(self.detected_anomalies),
            "by_type": {},
            "by_severity": {},
            "recent": []
        }
        
        # Group by type
        for anomaly in self.detected_anomalies:
            anomaly_type = anomaly.type.value
            if anomaly_type not in summary["by_type"]:
                summary["by_type"][anomaly_type] = 0
            summary["by_type"][anomaly_type] += 1
        
        # Group by severity
        for anomaly in self.detected_anomalies:
            severity = anomaly.severity
            if severity not in summary["by_severity"]:
                summary["by_severity"][severity] = 0
            summary["by_severity"][severity] += 1
        
        # Recent anomalies (last 10)
        summary["recent"] = [
            {
                "type": a.type.value,
                "severity": a.severity,
                "description": a.description,
                "timestamp": a.timestamp
            }
            for a in self.detected_anomalies[-10:]
        ]
        
        return summary

