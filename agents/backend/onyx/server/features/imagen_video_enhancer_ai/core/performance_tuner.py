"""
Performance Tuner
=================

System for automatic performance tuning and optimization.
"""

import asyncio
import logging
import psutil
import gc
import statistics
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TuningAction(Enum):
    """Tuning action."""
    INCREASE_WORKERS = "increase_workers"
    DECREASE_WORKERS = "decrease_workers"
    CLEAR_CACHE = "clear_cache"
    OPTIMIZE_MEMORY = "optimize_memory"
    ADJUST_TIMEOUT = "adjust_timeout"
    NONE = "none"


@dataclass
class TuningRecommendation:
    """Performance tuning recommendation."""
    action: TuningAction
    reason: str
    current_value: Any
    recommended_value: Any
    priority: int = 0  # Higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceTuner:
    """Performance tuner."""
    
    def __init__(self):
        """Initialize performance tuner."""
        self.recommendations: List[TuningRecommendation] = []
        self.max_recommendations = 1000
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def analyze(
        self,
        current_workers: int,
        avg_duration: float,
        success_rate: float,
        memory_usage: float,
        cpu_usage: float
    ) -> List[TuningRecommendation]:
        """
        Analyze performance and generate recommendations.
        
        Args:
            current_workers: Current number of workers
            avg_duration: Average operation duration
            success_rate: Success rate
            memory_usage: Memory usage percentage
            cpu_usage: CPU usage percentage
            
        Returns:
            List of tuning recommendations
        """
        recommendations = []
        
        # Memory analysis
        if memory_usage > 85:
            recommendations.append(TuningRecommendation(
                action=TuningAction.OPTIMIZE_MEMORY,
                reason=f"High memory usage: {memory_usage:.1f}%",
                current_value=memory_usage,
                recommended_value=70.0,
                priority=3
            ))
        
        if memory_usage > 90:
            recommendations.append(TuningRecommendation(
                action=TuningAction.CLEAR_CACHE,
                reason=f"Very high memory usage: {memory_usage:.1f}%",
                current_value=memory_usage,
                recommended_value=70.0,
                priority=4
            ))
        
        # CPU analysis
        if cpu_usage < 30 and success_rate > 0.95:
            # Low CPU, high success - can increase workers
            new_workers = min(current_workers + 2, 20)
            recommendations.append(TuningRecommendation(
                action=TuningAction.INCREASE_WORKERS,
                reason=f"Low CPU usage ({cpu_usage:.1f}%) with high success rate ({success_rate:.2%})",
                current_value=current_workers,
                recommended_value=new_workers,
                priority=1
            ))
        
        if cpu_usage > 90:
            # High CPU - decrease workers
            new_workers = max(current_workers - 2, 1)
            recommendations.append(TuningRecommendation(
                action=TuningAction.DECREASE_WORKERS,
                reason=f"High CPU usage: {cpu_usage:.1f}%",
                current_value=current_workers,
                recommended_value=new_workers,
                priority=2
            ))
        
        # Duration analysis
        if avg_duration > 60 and success_rate < 0.8:
            # Long duration, low success - might need timeout adjustment
            recommendations.append(TuningRecommendation(
                action=TuningAction.ADJUST_TIMEOUT,
                reason=f"Long average duration ({avg_duration:.1f}s) with low success rate ({success_rate:.2%})",
                current_value=avg_duration,
                recommended_value=avg_duration * 1.5,
                priority=2
            ))
        
        # Save recommendations
        self.recommendations.extend(recommendations)
        if len(self.recommendations) > self.max_recommendations:
            self.recommendations = self.recommendations[-self.max_recommendations:]
        
        # Save metrics
        self.metrics_history.append({
            "timestamp": datetime.now(),
            "workers": current_workers,
            "avg_duration": avg_duration,
            "success_rate": success_rate,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage
        })
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        return recommendations
    
    def get_recommendations(
        self,
        priority: Optional[int] = None,
        limit: int = 10
    ) -> List[TuningRecommendation]:
        """
        Get recommendations.
        
        Args:
            priority: Optional minimum priority
            limit: Maximum number of recommendations
            
        Returns:
            List of recommendations
        """
        recommendations = self.recommendations
        
        if priority is not None:
            recommendations = [r for r in recommendations if r.priority >= priority]
        
        # Sort by priority and timestamp
        recommendations.sort(key=lambda r: (r.priority, r.timestamp), reverse=True)
        
        return recommendations[:limit]
    
    def get_metrics_trend(self, period: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get metrics trend.
        
        Args:
            period: Optional time period
            
        Returns:
            Metrics trend dictionary
        """
        metrics = self.metrics_history
        
        if period:
            cutoff = datetime.now() - period
            metrics = [m for m in metrics if m.get("timestamp", datetime.now()) >= cutoff]
        
        if not metrics:
            return {}
        
        return {
            "avg_workers": statistics.mean([m["workers"] for m in metrics]),
            "avg_duration": statistics.mean([m["avg_duration"] for m in metrics]),
            "avg_success_rate": statistics.mean([m["success_rate"] for m in metrics]),
            "avg_memory": statistics.mean([m["memory_usage"] for m in metrics]),
            "avg_cpu": statistics.mean([m["cpu_usage"] for m in metrics]),
            "sample_count": len(metrics)
        }
    
    def clear_recommendations(self):
        """Clear all recommendations."""
        self.recommendations.clear()

