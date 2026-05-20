"""
Machine Learning utilities for cache optimization.

Uses ML techniques to optimize cache behavior.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Callable
import time

from kv_cache.types import StatsDict

logger = logging.getLogger(__name__)


class CacheMLPredictor:
    """
    ML-based cache predictor.
    
    Uses simple heuristics and patterns to predict cache behavior.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize ML predictor.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.access_patterns: Dict[int, List[int]] = {}
        self.prediction_history: List[Dict[str, Any]] = []
    
    def learn_pattern(self, position: int, next_position: int) -> None:
        """
        Learn access pattern.
        
        Args:
            position: Current position
            next_position: Next position accessed
        """
        if position not in self.access_patterns:
            self.access_patterns[position] = []
        self.access_patterns[position].append(next_position)
        
        # Keep only recent patterns
        if len(self.access_patterns[position]) > 100:
            self.access_patterns[position] = self.access_patterns[position][-100:]
    
    def predict_next(self, position: int, top_k: int = 5) -> List[int]:
        """
        Predict next positions likely to be accessed.
        
        Args:
            position: Current position
            top_k: Number of predictions to return
            
        Returns:
            List of predicted positions
        """
        if position not in self.access_patterns:
            return []
        
        # Count frequency
        next_positions = self.access_patterns[position]
        frequency: Dict[int, int] = {}
        
        for pos in next_positions:
            frequency[pos] = frequency.get(pos, 0) + 1
        
        # Sort by frequency
        sorted_positions = sorted(
            frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [pos for pos, _ in sorted_positions]
    
    def predict_access_time(self, position: int) -> float:
        """
        Predict when position will be accessed next.
        
        Args:
            position: Position to predict
            
        Returns:
            Predicted time until next access (seconds)
        """
        # Simple heuristic: based on access frequency
        if position not in self.access_patterns:
            return float('inf')
        
        frequency = len(self.access_patterns[position])
        # Higher frequency = sooner access (inverse relationship)
        return 1.0 / max(frequency, 1)
    
    def get_prediction_accuracy(self) -> float:
        """
        Get prediction accuracy (simple metric).
        
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not self.prediction_history:
            return 0.0
        
        correct = sum(1 for p in self.prediction_history if p.get("correct", False))
        return correct / len(self.prediction_history)


class CacheMLOptimizer:
    """
    ML-based cache optimizer.
    
    Uses learning to optimize cache configuration.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize ML optimizer.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.performance_history: List[Dict[str, Any]] = []
        self.config_history: List[Dict[str, Any]] = []
    
    def record_performance(self, metrics: Dict[str, Any]) -> None:
        """
        Record performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
        """
        self.performance_history.append({
            "timestamp": time.time(),
            "metrics": metrics
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def suggest_configuration(self) -> Optional[Dict[str, Any]]:
        """
        Suggest optimal configuration based on learned patterns.
        
        Returns:
            Suggested configuration or None
        """
        if len(self.performance_history) < 10:
            return None
        
        # Analyze recent performance
        recent = self.performance_history[-10:]
        avg_hit_rate = sum(m["metrics"].get("hit_rate", 0) for m in recent) / len(recent)
        avg_memory = sum(m["metrics"].get("memory_mb", 0) for m in recent) / len(recent)
        
        suggestions = {}
        
        # Suggest based on hit rate
        if avg_hit_rate < 0.6:
            current_size = self.cache.config.max_tokens
            suggestions["max_tokens"] = int(current_size * 1.5)
        
        # Suggest based on memory
        if avg_memory > 1000:
            suggestions["use_compression"] = True
            suggestions["use_quantization"] = True
        
        return suggestions if suggestions else None
    
    def learn_from_performance(self) -> Dict[str, Any]:
        """
        Learn from performance history.
        
        Returns:
            Dictionary with learned insights
        """
        if len(self.performance_history) < 20:
            return {"message": "Insufficient data"}
        
        # Analyze trends
        recent = self.performance_history[-10:]
        older = self.performance_history[-20:-10]
        
        recent_avg_hit = sum(m["metrics"].get("hit_rate", 0) for m in recent) / len(recent)
        older_avg_hit = sum(m["metrics"].get("hit_rate", 0) for m in older) / len(older)
        
        hit_trend = "improving" if recent_avg_hit > older_avg_hit else "declining"
        
        return {
            "hit_rate_trend": hit_trend,
            "recent_avg_hit_rate": recent_avg_hit,
            "older_avg_hit_rate": older_avg_hit,
            "suggestions": self.suggest_configuration()
        }

