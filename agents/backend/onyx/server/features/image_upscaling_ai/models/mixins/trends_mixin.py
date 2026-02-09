"""
Trends Mixin

Contains trends analysis and insights functionality.
"""

import logging
from typing import Union, Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class TrendsMixin:
    """
    Mixin providing trends analysis and insights functionality.
    
    This mixin contains:
    - Usage trends
    - Method popularity
    - Performance trends
    - Quality trends
    - Recommendations based on trends
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize trends mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_trends_data'):
            self._trends_data = {
                "method_usage": defaultdict(list),
                "quality_scores": defaultdict(list),
                "performance_metrics": defaultdict(list),
            }
    
    def analyze_usage_trends(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze usage trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if not hasattr(self, 'get_history'):
            return {"error": "HistoryMixin required for trend analysis"}
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        history = self.get_history(start_date=cutoff_date)
        
        # Group by date
        daily_usage = defaultdict(int)
        method_usage = Counter()
        operation_types = Counter()
        
        for entry in history:
            # Daily usage
            date = entry["timestamp"][:10]  # YYYY-MM-DD
            daily_usage[date] += 1
            
            # Method usage
            if entry.get("method"):
                method_usage[entry["method"]] += 1
            
            # Operation types
            operation_types[entry["operation"]] += 1
        
        # Calculate trends
        dates = sorted(daily_usage.keys())
        usage_values = [daily_usage[date] for date in dates]
        
        trend_direction = "stable"
        if len(usage_values) >= 2:
            recent_avg = sum(usage_values[-7:]) / min(7, len(usage_values))
            older_avg = sum(usage_values[:-7]) / max(1, len(usage_values) - 7) if len(usage_values) > 7 else recent_avg
            if recent_avg > older_avg * 1.1:
                trend_direction = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend_direction = "decreasing"
        
        return {
            "period_days": days,
            "total_operations": len(history),
            "daily_usage": dict(daily_usage),
            "method_popularity": dict(method_usage.most_common()),
            "operation_distribution": dict(operation_types),
            "trend_direction": trend_direction,
            "average_daily_usage": sum(usage_values) / len(usage_values) if usage_values else 0,
        }
    
    def get_method_popularity(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get method popularity trends.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with method popularity
        """
        if not hasattr(self, 'get_history'):
            return {"error": "HistoryMixin required"}
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        history = self.get_history(start_date=cutoff_date)
        
        method_stats = defaultdict(lambda: {"count": 0, "success": 0, "total_time": 0.0})
        
        for entry in history:
            if entry.get("method"):
                method = entry["method"]
                method_stats[method]["count"] += 1
                if entry.get("success"):
                    method_stats[method]["success"] += 1
                # Note: time would need to be tracked in history
        
        # Calculate popularity scores
        popularity = {}
        for method, stats in method_stats.items():
            success_rate = stats["success"] / stats["count"] if stats["count"] > 0 else 0
            popularity[method] = {
                "usage_count": stats["count"],
                "success_rate": success_rate,
                "popularity_score": stats["count"] * success_rate,
            }
        
        # Sort by popularity
        sorted_popularity = sorted(
            popularity.items(),
            key=lambda x: x[1]["popularity_score"],
            reverse=True
        )
        
        return {
            "period_days": days,
            "method_popularity": dict(sorted_popularity),
            "most_popular": sorted_popularity[0][0] if sorted_popularity else None,
        }
    
    def get_performance_trends(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get performance trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with performance trends
        """
        if not hasattr(self, 'get_history'):
            return {"error": "HistoryMixin required"}
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        history = self.get_history(start_date=cutoff_date)
        
        # Group by week
        weekly_performance = defaultdict(lambda: {"count": 0, "success": 0})
        
        for entry in history:
            date = datetime.fromisoformat(entry["timestamp"])
            week_key = f"{date.year}-W{date.isocalendar()[1]}"
            weekly_performance[week_key]["count"] += 1
            if entry.get("success"):
                weekly_performance[week_key]["success"] += 1
        
        # Calculate success rates
        performance_trends = {}
        for week, stats in weekly_performance.items():
            performance_trends[week] = {
                "operations": stats["count"],
                "success_rate": stats["success"] / stats["count"] if stats["count"] > 0 else 0,
            }
        
        return {
            "period_days": days,
            "weekly_performance": dict(performance_trends),
            "overall_success_rate": sum(s["success"] for s in weekly_performance.values()) / sum(s["count"] for s in weekly_performance.values()) if weekly_performance else 0,
        }
    
    def get_recommendations_from_trends(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get recommendations based on usage trends.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with recommendations
        """
        usage_trends = self.analyze_usage_trends(days)
        method_popularity = self.get_method_popularity(days)
        
        recommendations = []
        
        # Recommend most popular method
        if method_popularity.get("most_popular"):
            recommendations.append({
                "type": "method",
                "recommendation": f"Consider using '{method_popularity['most_popular']}' - it's the most popular method",
                "confidence": "high",
            })
        
        # Recommend based on trend direction
        if usage_trends.get("trend_direction") == "increasing":
            recommendations.append({
                "type": "usage",
                "recommendation": "Usage is increasing - consider optimizing for scale",
                "confidence": "medium",
            })
        
        return {
            "period_days": days,
            "recommendations": recommendations,
            "trends": usage_trends,
            "method_popularity": method_popularity,
        }


