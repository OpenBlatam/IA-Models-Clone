"""
Advanced cache analytics.

Provides deep analytics and insights for cache performance.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import statistics

from kv_cache.types import StatsDict

logger = logging.getLogger(__name__)


class CacheAnalytics:
    """
    Advanced cache analytics engine.
    
    Provides deep insights and analytics for cache performance.
    """
    
    def __init__(
        self,
        cache: Any,
        window_size: int = 1000
    ):
        """
        Initialize cache analytics.
        
        Args:
            cache: Cache instance
            window_size: Size of sliding window for analytics
        """
        self.cache = cache
        self.window_size = window_size
        
        # Time series data
        self.hit_rate_series: deque = deque(maxlen=window_size)
        self.latency_series: deque = deque(maxlen=window_size)
        self.memory_series: deque = deque(maxlen=window_size)
        
        # Pattern analysis
        self.access_patterns: Dict[int, List[float]] = defaultdict(list)
        self.sequence_patterns: List[List[int]] = []
    
    def analyze_hit_rate_distribution(self) -> Dict[str, Any]:
        """
        Analyze hit rate distribution.
        
        Returns:
            Dictionary with hit rate analysis
        """
        if not self.hit_rate_series:
            return {"message": "Insufficient data"}
        
        hit_rates = list(self.hit_rate_series)
        
        return {
            "mean": statistics.mean(hit_rates),
            "median": statistics.median(hit_rates),
            "std": statistics.stdev(hit_rates) if len(hit_rates) > 1 else 0.0,
            "min": min(hit_rates),
            "max": max(hit_rates),
            "p25": statistics.quantiles(hit_rates, n=4)[0] if len(hit_rates) > 4 else hit_rates[0],
            "p75": statistics.quantiles(hit_rates, n=4)[2] if len(hit_rates) > 4 else hit_rates[-1],
            "p95": statistics.quantiles(hit_rates, n=20)[18] if len(hit_rates) > 20 else hit_rates[-1],
            "p99": statistics.quantiles(hit_rates, n=100)[98] if len(hit_rates) > 100 else hit_rates[-1]
        }
    
    def analyze_latency_patterns(self) -> Dict[str, Any]:
        """
        Analyze latency patterns.
        
        Returns:
            Dictionary with latency analysis
        """
        if not self.latency_series:
            return {"message": "Insufficient data"}
        
        latencies = list(self.latency_series)
        
        # Detect outliers
        mean = statistics.mean(latencies)
        std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        
        outliers = [
            lat for lat in latencies
            if abs(lat - mean) > 3 * std
        ] if std > 0 else []
        
        return {
            "mean_ms": statistics.mean(latencies) * 1000,
            "median_ms": statistics.median(latencies) * 1000,
            "std_ms": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0.0,
            "min_ms": min(latencies) * 1000,
            "max_ms": max(latencies) * 1000,
            "p95_ms": (
                statistics.quantiles(latencies, n=20)[18] * 1000
                if len(latencies) > 20 else max(latencies) * 1000
            ),
            "outliers": len(outliers),
            "outlier_rate": len(outliers) / len(latencies) if latencies else 0.0
        }
    
    def analyze_access_patterns(self) -> Dict[str, Any]:
        """
        Analyze access patterns.
        
        Returns:
            Dictionary with access pattern analysis
        """
        if not self.access_patterns:
            return {"message": "No access pattern data"}
        
        # Analyze frequency
        frequencies = [len(times) for times in self.access_patterns.values()]
        
        # Analyze temporal patterns
        temporal_spreads = []
        for times in self.access_patterns.values():
            if len(times) > 1:
                spread = max(times) - min(times)
                temporal_spreads.append(spread)
        
        return {
            "unique_positions": len(self.access_patterns),
            "total_accesses": sum(frequencies),
            "avg_accesses_per_position": statistics.mean(frequencies) if frequencies else 0.0,
            "max_accesses": max(frequencies) if frequencies else 0,
            "min_accesses": min(frequencies) if frequencies else 0,
            "temporal_spread_mean": statistics.mean(temporal_spreads) if temporal_spreads else 0.0,
            "most_accessed_positions": sorted(
                self.access_patterns.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10]
        }
    
    def analyze_sequence_patterns(self) -> Dict[str, Any]:
        """
        Analyze sequence patterns.
        
        Returns:
            Dictionary with sequence analysis
        """
        if not self.sequence_patterns:
            return {"message": "No sequence data"}
        
        # Find common sequences
        sequence_counts: Dict[tuple, int] = defaultdict(int)
        
        for seq in self.sequence_patterns:
            if len(seq) >= 2:
                # Extract pairs
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    sequence_counts[pair] += 1
        
        # Find most common transitions
        common_transitions = sorted(
            sequence_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_sequences": len(self.sequence_patterns),
            "unique_transitions": len(sequence_counts),
            "most_common_transitions": [
                {"from": pair[0], "to": pair[1], "count": count}
                for pair, count in common_transitions
            ]
        }
    
    def predict_future_access(
        self,
        current_position: int,
        lookahead: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predict future access patterns.
        
        Args:
            current_position: Current position
            lookahead: Number of positions to predict ahead
            
        Returns:
            List of predictions
        """
        predictions = []
        
        # Use sequence patterns to predict
        if self.sequence_patterns:
            # Find sequences starting with current position
            matching_sequences = [
                seq for seq in self.sequence_patterns
                if seq and seq[0] == current_position
            ]
            
            if matching_sequences:
                # Aggregate next positions
                next_positions: Dict[int, int] = defaultdict(int)
                
                for seq in matching_sequences:
                    if len(seq) > 1:
                        next_positions[seq[1]] += 1
                
                # Sort by frequency
                sorted_next = sorted(
                    next_positions.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:lookahead]
                
                predictions = [
                    {
                        "position": pos,
                        "confidence": count / len(matching_sequences),
                        "frequency": count
                    }
                    for pos, count in sorted_next
                ]
        
        return predictions
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights.
        
        Returns:
            Dictionary with comprehensive insights
        """
        insights = {
            "timestamp": time.time(),
            "hit_rate_analysis": self.analyze_hit_rate_distribution(),
            "latency_analysis": self.analyze_latency_patterns(),
            "access_pattern_analysis": self.analyze_access_patterns(),
            "sequence_analysis": self.analyze_sequence_patterns()
        }
        
        # Generate recommendations
        recommendations = []
        
        hit_rate_analysis = insights["hit_rate_analysis"]
        if "mean" in hit_rate_analysis:
            if hit_rate_analysis["mean"] < 0.6:
                recommendations.append({
                    "type": "low_hit_rate",
                    "severity": "high",
                    "message": "Hit rate is below optimal threshold",
                    "suggestion": "Consider increasing cache size or improving eviction strategy"
                })
        
        latency_analysis = insights["latency_analysis"]
        if "outlier_rate" in latency_analysis:
            if latency_analysis["outlier_rate"] > 0.1:
                recommendations.append({
                    "type": "high_latency_variance",
                    "severity": "medium",
                    "message": "High latency variance detected",
                    "suggestion": "Investigate cache operations for bottlenecks"
                })
        
        insights["recommendations"] = recommendations
        
        return insights

