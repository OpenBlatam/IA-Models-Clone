"""
Quality Monitor for Flux2 Clothing Changer
===========================================

Advanced quality monitoring and tracking system.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Quality metric."""
    metric_name: str
    value: float
    threshold: float
    timestamp: float
    passed: bool
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QualityReport:
    """Quality report."""
    report_id: str
    overall_score: float
    metrics: List[QualityMetric]
    passed: bool
    timestamp: float = time.time()
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class QualityMonitor:
    """Advanced quality monitoring system."""
    
    def __init__(
        self,
        history_size: int = 10000,
    ):
        """
        Initialize quality monitor.
        
        Args:
            history_size: Maximum history size
        """
        self.history_size = history_size
        self.metrics_history: Dict[str, deque] = {}
        self.thresholds: Dict[str, float] = {}
        self.reports: List[QualityReport] = []
    
    def set_threshold(
        self,
        metric_name: str,
        threshold: float,
    ) -> None:
        """
        Set quality threshold.
        
        Args:
            metric_name: Metric name
            threshold: Threshold value
        """
        self.thresholds[metric_name] = threshold
        logger.info(f"Set threshold for {metric_name}: {threshold}")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QualityMetric:
        """
        Record quality metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
            metadata: Optional metadata
            
        Returns:
            Quality metric
        """
        threshold = self.thresholds.get(metric_name, 0.0)
        passed = value >= threshold
        
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=self.history_size)
        
        metric = QualityMetric(
            metric_name=metric_name,
            value=value,
            threshold=threshold,
            timestamp=time.time(),
            passed=passed,
            metadata=metadata or {},
        )
        
        self.metrics_history[metric_name].append(metric)
        logger.debug(f"Recorded metric: {metric_name} = {value} (passed: {passed})")
        return metric
    
    def generate_report(
        self,
        metric_names: Optional[List[str]] = None,
    ) -> QualityReport:
        """
        Generate quality report.
        
        Args:
            metric_names: Optional list of metric names
            
        Returns:
            Quality report
        """
        if metric_names is None:
            metric_names = list(self.metrics_history.keys())
        
        metrics = []
        total_score = 0.0
        passed_count = 0
        
        for metric_name in metric_names:
            if metric_name in self.metrics_history and self.metrics_history[metric_name]:
                latest_metric = self.metrics_history[metric_name][-1]
                metrics.append(latest_metric)
                total_score += latest_metric.value
                if latest_metric.passed:
                    passed_count += 1
        
        overall_score = total_score / len(metrics) if metrics else 0.0
        passed = passed_count == len(metrics) if metrics else False
        
        # Generate recommendations
        recommendations = []
        for metric in metrics:
            if not metric.passed:
                recommendations.append(
                    f"Improve {metric.metric_name}: current {metric.value:.2f}, "
                    f"threshold {metric.threshold:.2f}"
                )
        
        report_id = f"report_{int(time.time() * 1000)}"
        
        report = QualityReport(
            report_id=report_id,
            overall_score=overall_score,
            metrics=metrics,
            passed=passed,
            recommendations=recommendations,
        )
        
        self.reports.append(report)
        logger.info(f"Generated quality report: {report_id}")
        return report
    
    def get_metric_trend(
        self,
        metric_name: str,
        time_range: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get metric trend.
        
        Args:
            metric_name: Metric name
            time_range: Optional time range in seconds
            
        Returns:
            Trend analysis
        """
        if metric_name not in self.metrics_history:
            return {}
        
        metrics = list(self.metrics_history[metric_name])
        
        if time_range:
            cutoff_time = time.time() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        passed_count = sum(1 for m in metrics if m.passed)
        
        return {
            "metric_name": metric_name,
            "current_value": values[-1],
            "average_value": sum(values) / len(values),
            "min_value": min(values),
            "max_value": max(values),
            "pass_rate": passed_count / len(metrics),
            "total_samples": len(metrics),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quality monitor statistics."""
        return {
            "total_metrics": len(self.metrics_history),
            "total_reports": len(self.reports),
            "thresholds_set": len(self.thresholds),
        }


