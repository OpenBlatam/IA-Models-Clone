"""
Anomaly Detector for Flux2 Clothing Changer
============================================

Detect anomalies in processing, quality, and performance.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Anomaly detection result."""
    type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    value: float
    threshold: float
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnomalyDetector:
    """Anomaly detection system for monitoring."""
    
    def __init__(
        self,
        window_size: int = 100,
        sensitivity: float = 2.0,  # Standard deviations for anomaly detection
    ):
        """
        Initialize anomaly detector.
        
        Args:
            window_size: Size of sliding window for statistics
            sensitivity: Sensitivity for anomaly detection (higher = less sensitive)
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        
        # History for different metrics
        self.processing_times: deque = deque(maxlen=window_size)
        self.quality_scores: deque = deque(maxlen=window_size)
        self.error_rates: deque = deque(maxlen=window_size)
        self.memory_usage: deque = deque(maxlen=window_size)
        
        # Detected anomalies
        self.anomalies: List[Anomaly] = []
        
        # Statistics
        self.stats = {
            "anomalies_detected": 0,
            "false_positives": 0,
            "true_positives": 0,
        }
    
    def record_metric(
        self,
        metric_type: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Anomaly]:
        """
        Record a metric and check for anomalies.
        
        Args:
            metric_type: Type of metric (processing_time, quality_score, error_rate, memory_usage)
            value: Metric value
            metadata: Optional metadata
            
        Returns:
            Anomaly if detected, None otherwise
        """
        # Add to history
        if metric_type == "processing_time":
            self.processing_times.append(value)
            history = self.processing_times
        elif metric_type == "quality_score":
            self.quality_scores.append(value)
            history = self.quality_scores
        elif metric_type == "error_rate":
            self.error_rates.append(value)
            history = self.error_rates
        elif metric_type == "memory_usage":
            self.memory_usage.append(value)
            history = self.memory_usage
        else:
            return None
        
        # Check for anomaly if we have enough data
        if len(history) < 10:
            return None
        
        anomaly = self._detect_anomaly(metric_type, value, history, metadata)
        
        if anomaly:
            self.anomalies.append(anomaly)
            self.stats["anomalies_detected"] += 1
            logger.warning(f"Anomaly detected: {anomaly.message}")
        
        return anomaly
    
    def _detect_anomaly(
        self,
        metric_type: str,
        value: float,
        history: deque,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[Anomaly]:
        """Detect anomaly in metric."""
        history_array = np.array(list(history))
        
        # Calculate statistics
        mean = np.mean(history_array)
        std = np.std(history_array)
        
        if std == 0:
            return None
        
        # Calculate z-score
        z_score = abs((value - mean) / std)
        
        # Determine severity
        if z_score > self.sensitivity * 3:
            severity = "critical"
        elif z_score > self.sensitivity * 2:
            severity = "high"
        elif z_score > self.sensitivity:
            severity = "medium"
        else:
            return None  # Not an anomaly
        
        # Generate message based on metric type
        messages = {
            "processing_time": {
                "high": f"Processing time is unusually high: {value:.2f}s (avg: {mean:.2f}s)",
                "critical": f"Processing time is critically high: {value:.2f}s (avg: {mean:.2f}s)",
            },
            "quality_score": {
                "high": f"Quality score is unusually low: {value:.2f} (avg: {mean:.2f})",
                "critical": f"Quality score is critically low: {value:.2f} (avg: {mean:.2f})",
            },
            "error_rate": {
                "high": f"Error rate is unusually high: {value:.2%} (avg: {mean:.2%})",
                "critical": f"Error rate is critically high: {value:.2%} (avg: {mean:.2%})",
            },
            "memory_usage": {
                "high": f"Memory usage is unusually high: {value:.2f}MB (avg: {mean:.2f}MB)",
                "critical": f"Memory usage is critically high: {value:.2f}MB (avg: {mean:.2f}MB)",
            },
        }
        
        message = messages.get(metric_type, {}).get(severity, f"Anomaly detected in {metric_type}")
        
        return Anomaly(
            type=metric_type,
            severity=severity,
            message=message,
            value=value,
            threshold=mean + (self.sensitivity * std),
            timestamp=time.time(),
            metadata=metadata or {},
        )
    
    def detect_pattern_anomalies(
        self,
        recent_events: List[Dict[str, Any]],
    ) -> List[Anomaly]:
        """
        Detect pattern-based anomalies.
        
        Args:
            recent_events: List of recent events
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(recent_events) < 10:
            return anomalies
        
        # Check for sudden increase in errors
        error_counts = [e.get("error", 0) for e in recent_events[-20:]]
        if len(error_counts) >= 10:
            recent_errors = sum(error_counts[-5:])
            previous_errors = sum(error_counts[-10:-5])
            
            if previous_errors == 0 and recent_errors > 3:
                anomalies.append(Anomaly(
                    type="error_pattern",
                    severity="high",
                    message=f"Sudden increase in errors: {recent_errors} errors in last 5 events",
                    value=recent_errors,
                    threshold=3,
                    timestamp=time.time(),
                ))
        
        # Check for degradation in quality
        quality_scores = [e.get("quality_score", 0.5) for e in recent_events[-20:] if "quality_score" in e]
        if len(quality_scores) >= 10:
            recent_avg = np.mean(quality_scores[-5:])
            previous_avg = np.mean(quality_scores[-10:-5])
            
            if previous_avg - recent_avg > 0.2:
                anomalies.append(Anomaly(
                    type="quality_degradation",
                    severity="medium",
                    message=f"Quality degradation detected: {recent_avg:.2f} vs {previous_avg:.2f}",
                    value=recent_avg,
                    threshold=previous_avg - 0.2,
                    timestamp=time.time(),
                ))
        
        return anomalies
    
    def get_anomaly_summary(
        self,
        time_range: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get summary of detected anomalies.
        
        Args:
            time_range: Time range in seconds (None for all)
            
        Returns:
            Anomaly summary
        """
        cutoff_time = time.time() - time_range if time_range else 0
        
        relevant_anomalies = [
            a for a in self.anomalies
            if a.timestamp >= cutoff_time
        ]
        
        if not relevant_anomalies:
            return {
                "total": 0,
                "by_severity": {},
                "by_type": {},
            }
        
        by_severity = {}
        by_type = {}
        
        for anomaly in relevant_anomalies:
            by_severity[anomaly.severity] = by_severity.get(anomaly.severity, 0) + 1
            by_type[anomaly.type] = by_type.get(anomaly.type, 0) + 1
        
        return {
            "total": len(relevant_anomalies),
            "by_severity": by_severity,
            "by_type": by_type,
            "recent": [
                {
                    "type": a.type,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp,
                }
                for a in relevant_anomalies[-10:]
            ],
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            **self.stats,
            "total_anomalies": len(self.anomalies),
            "window_size": self.window_size,
            "sensitivity": self.sensitivity,
            "current_metrics": {
                "processing_times": len(self.processing_times),
                "quality_scores": len(self.quality_scores),
                "error_rates": len(self.error_rates),
                "memory_usage": len(self.memory_usage),
            },
        }
    
    def reset(self) -> None:
        """Reset detector state."""
        self.processing_times.clear()
        self.quality_scores.clear()
        self.error_rates.clear()
        self.memory_usage.clear()
        self.anomalies.clear()
        self.stats = {
            "anomalies_detected": 0,
            "false_positives": 0,
            "true_positives": 0,
        }
        logger.info("Anomaly detector reset")


