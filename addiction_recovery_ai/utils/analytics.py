"""
Advanced Analytics for Recovery AI
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RecoveryAnalytics:
    """Advanced analytics for recovery data"""
    
    def __init__(self):
        """Initialize analytics"""
        self.data = defaultdict(list)
        logger.info("RecoveryAnalytics initialized")
    
    def add_data_point(
        self,
        user_id: str,
        timestamp: datetime,
        features: Dict[str, float],
        prediction: Optional[float] = None
    ):
        """Add data point"""
        self.data[user_id].append({
            "timestamp": timestamp,
            "features": features,
            "prediction": prediction
        })
    
    def calculate_trends(
        self,
        user_id: str,
        metric: str = "progress",
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate trends for user
        
        Args:
            user_id: User identifier
            metric: Metric to analyze
            days: Number of days to analyze
        
        Returns:
            Trend analysis
        """
        if user_id not in self.data:
            return {}
        
        cutoff = datetime.now() - timedelta(days=days)
        user_data = [
            d for d in self.data[user_id]
            if d["timestamp"] >= cutoff
        ]
        
        if len(user_data) < 2:
            return {"trend": "insufficient_data"}
        
        # Extract metric values
        if metric == "progress":
            values = [d.get("prediction", 0) for d in user_data]
        else:
            values = [d["features"].get(metric, 0) for d in user_data]
        
        # Calculate trend
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        change = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
        
        if change > 5:
            trend = "improving"
        elif change < -5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change_percent": change,
            "current_value": values[-1],
            "average_value": np.mean(values),
            "data_points": len(values)
        }
    
    def detect_patterns(
        self,
        user_id: str,
        window_size: int = 7
    ) -> Dict[str, Any]:
        """
        Detect patterns in user data
        
        Args:
            user_id: User identifier
            window_size: Window size for pattern detection
        
        Returns:
            Detected patterns
        """
        if user_id not in self.data:
            return {}
        
        user_data = self.data[user_id][-window_size:]
        
        if len(user_data) < window_size:
            return {"patterns": []}
        
        patterns = []
        
        # Check for weekly patterns
        cravings = [d["features"].get("cravings_level", 5) for d in user_data]
        if len(set(cravings)) < len(cravings) * 0.5:
            patterns.append("repetitive_cravings")
        
        # Check for stress patterns
        stress = [d["features"].get("stress_level", 5) for d in user_data]
        if np.std(stress) > 3:
            patterns.append("high_stress_variability")
        
        # Check for improvement
        if user_data[-1].get("prediction", 0) > user_data[0].get("prediction", 0):
            patterns.append("improving_trend")
        
        return {
            "patterns": patterns,
            "window_size": window_size
        }
    
    def generate_insights(
        self,
        user_id: str
    ) -> List[str]:
        """
        Generate insights for user
        
        Args:
            user_id: User identifier
        
        Returns:
            List of insights
        """
        insights = []
        
        trends = self.calculate_trends(user_id)
        patterns = self.detect_patterns(user_id)
        
        if trends.get("trend") == "improving":
            insights.append("Your recovery progress is improving! Keep up the great work.")
        
        if trends.get("trend") == "declining":
            insights.append("We notice a decline in your progress. Consider reaching out for support.")
        
        if "repetitive_cravings" in patterns.get("patterns", []):
            insights.append("You're experiencing repetitive cravings. Try identifying triggers.")
        
        if "high_stress_variability" in patterns.get("patterns", []):
            insights.append("Your stress levels are highly variable. Consider stress management techniques.")
        
        return insights


class CohortAnalysis:
    """Cohort analysis for recovery groups"""
    
    def __init__(self):
        """Initialize cohort analysis"""
        self.cohorts = defaultdict(list)
    
    def add_to_cohort(
        self,
        cohort_id: str,
        user_id: str,
        data: Dict[str, Any]
    ):
        """Add user to cohort"""
        self.cohorts[cohort_id].append({
            "user_id": user_id,
            **data
        })
    
    def analyze_cohort(
        self,
        cohort_id: str,
        metric: str = "progress"
    ) -> Dict[str, Any]:
        """
        Analyze cohort
        
        Args:
            cohort_id: Cohort identifier
            metric: Metric to analyze
        
        Returns:
            Cohort analysis
        """
        if cohort_id not in self.cohorts:
            return {}
        
        cohort_data = self.cohorts[cohort_id]
        
        if not cohort_data:
            return {}
        
        values = [d.get(metric, 0) for d in cohort_data]
        
        return {
            "cohort_id": cohort_id,
            "size": len(cohort_data),
            "metric": metric,
            "average": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "percentiles": {
                "p25": np.percentile(values, 25),
                "p50": np.percentile(values, 50),
                "p75": np.percentile(values, 75),
                "p95": np.percentile(values, 95)
            }
        }
    
    def compare_cohorts(
        self,
        cohort1_id: str,
        cohort2_id: str,
        metric: str = "progress"
    ) -> Dict[str, Any]:
        """
        Compare two cohorts
        
        Args:
            cohort1_id: First cohort
            cohort2_id: Second cohort
            metric: Metric to compare
        
        Returns:
            Comparison results
        """
        analysis1 = self.analyze_cohort(cohort1_id, metric)
        analysis2 = self.analyze_cohort(cohort2_id, metric)
        
        if not analysis1 or not analysis2:
            return {}
        
        avg1 = analysis1["average"]
        avg2 = analysis2["average"]
        
        return {
            "cohort1": analysis1,
            "cohort2": analysis2,
            "difference": avg2 - avg1,
            "difference_percent": ((avg2 - avg1) / avg1 * 100) if avg1 > 0 else 0,
            "significant": abs(avg2 - avg1) > (analysis1["std"] + analysis2["std"]) / 2
        }

