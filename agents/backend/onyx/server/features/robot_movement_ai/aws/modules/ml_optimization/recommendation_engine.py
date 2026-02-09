"""
Recommendation Engine
=====================

ML-based recommendation engine for optimizations.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Optimization recommendation."""
    type: str  # cache, connection, query, etc.
    action: str
    priority: int  # 1-10
    expected_improvement: float  # percentage
    description: str
    parameters: Dict[str, Any]


class RecommendationEngine:
    """ML-based recommendation engine."""
    
    def __init__(self):
        self._recommendations: List[Recommendation] = []
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._applied_recommendations: List[str] = []
    
    def analyze_and_recommend(self, metrics: Dict[str, float]) -> List[Recommendation]:
        """Analyze metrics and generate recommendations."""
        recommendations = []
        
        # Cache recommendations
        cache_hit_rate = metrics.get("cache_hit_rate", 0)
        if cache_hit_rate < 0.7:
            recommendations.append(Recommendation(
                type="cache",
                action="increase_cache_ttl",
                priority=8,
                expected_improvement=20.0,
                description=f"Cache hit rate is low ({cache_hit_rate:.1%}), increase TTL",
                parameters={"ttl_multiplier": 1.5}
            ))
        
        # Connection pool recommendations
        connection_utilization = metrics.get("connection_utilization", 0)
        if connection_utilization > 0.9:
            recommendations.append(Recommendation(
                type="connection",
                action="increase_pool_size",
                priority=9,
                expected_improvement=15.0,
                description=f"Connection pool utilization is high ({connection_utilization:.1%})",
                parameters={"pool_size_multiplier": 1.5}
            ))
        
        # Query optimization recommendations
        avg_query_time = metrics.get("avg_query_time", 0)
        if avg_query_time > 0.5:  # 500ms
            recommendations.append(Recommendation(
                type="query",
                action="add_indexes",
                priority=7,
                expected_improvement=40.0,
                description=f"Average query time is high ({avg_query_time:.3f}s)",
                parameters={"analyze_queries": True}
            ))
        
        # Memory recommendations
        memory_usage = metrics.get("memory_usage_percent", 0)
        if memory_usage > 0.85:
            recommendations.append(Recommendation(
                type="memory",
                action="optimize_memory",
                priority=6,
                expected_improvement=25.0,
                description=f"Memory usage is high ({memory_usage:.1%})",
                parameters={"gc_threshold": "aggressive"}
            ))
        
        self._recommendations.extend(recommendations)
        return recommendations
    
    def get_recommendations(self, priority_min: int = 0, limit: int = 10) -> List[Recommendation]:
        """Get recommendations sorted by priority."""
        filtered = [
            r for r in self._recommendations
            if r.priority >= priority_min and r.action not in self._applied_recommendations
        ]
        
        sorted_recs = sorted(filtered, key=lambda x: x.priority, reverse=True)
        return sorted_recs[:limit]
    
    def apply_recommendation(self, recommendation_id: str) -> bool:
        """Mark recommendation as applied."""
        # In production, implement actual application logic
        self._applied_recommendations.append(recommendation_id)
        logger.info(f"Applied recommendation: {recommendation_id}")
        return True
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """Get recommendation statistics."""
        return {
            "total_recommendations": len(self._recommendations),
            "applied": len(self._applied_recommendations),
            "pending": len(self._recommendations) - len(self._applied_recommendations),
            "by_type": {
                rec.type: sum(1 for r in self._recommendations if r.type == rec.type)
                for rec in self._recommendations
            }
        }















