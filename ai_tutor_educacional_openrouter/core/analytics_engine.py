"""
Advanced analytics engine for learning insights.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Advanced analytics engine for learning patterns and insights.
    """
    
    def __init__(self):
        self.analytics_data: Dict[str, Any] = defaultdict(list)
    
    def analyze_learning_patterns(
        self,
        student_id: str,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze learning patterns from student interactions.
        
        Args:
            student_id: Student identifier
            interactions: List of student interactions
        
        Returns:
            Analysis results with patterns and insights
        """
        if not interactions:
            return {"error": "No interactions to analyze"}
        
        # Analyze patterns
        patterns = {
            "total_interactions": len(interactions),
            "topics_covered": self._extract_topics(interactions),
            "difficulty_progression": self._analyze_difficulty_progression(interactions),
            "time_patterns": self._analyze_time_patterns(interactions),
            "engagement_trends": self._analyze_engagement_trends(interactions),
            "strengths": self._identify_strengths(interactions),
            "weaknesses": self._identify_weaknesses(interactions)
        }
        
        # Generate insights
        insights = self._generate_insights(patterns)
        
        return {
            "student_id": student_id,
            "patterns": patterns,
            "insights": insights,
            "analyzed_at": datetime.now().isoformat()
        }
    
    def _extract_topics(self, interactions: List[Dict[str, Any]]) -> List[str]:
        """Extract unique topics from interactions."""
        topics = set()
        for interaction in interactions:
            if "topic" in interaction:
                topics.add(interaction["topic"])
            if "subject" in interaction:
                topics.add(interaction["subject"])
        return sorted(list(topics))
    
    def _analyze_difficulty_progression(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how difficulty changes over time."""
        difficulties = []
        for interaction in interactions:
            if "difficulty" in interaction:
                difficulty_map = {"beginner": 1, "intermediate": 2, "advanced": 3}
                difficulties.append(difficulty_map.get(interaction["difficulty"], 2))
        
        if not difficulties:
            return {"trend": "stable", "average": 2}
        
        # Calculate trend
        if len(difficulties) > 1:
            trend = "increasing" if difficulties[-1] > difficulties[0] else "decreasing"
            if difficulties[-1] == difficulties[0]:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "average": sum(difficulties) / len(difficulties),
            "current": difficulties[-1] if difficulties else 2
        }
    
    def _analyze_time_patterns(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze time-based patterns."""
        if not interactions:
            return {}
        
        timestamps = []
        for interaction in interactions:
            if "timestamp" in interaction:
                try:
                    if isinstance(interaction["timestamp"], str):
                        timestamps.append(datetime.fromisoformat(interaction["timestamp"]))
                    else:
                        timestamps.append(interaction["timestamp"])
                except:
                    pass
        
        if not timestamps:
            return {}
        
        # Analyze patterns
        hours = [ts.hour for ts in timestamps]
        days = [ts.weekday() for ts in timestamps]
        
        return {
            "peak_hour": max(set(hours), key=hours.count) if hours else None,
            "peak_day": max(set(days), key=days.count) if days else None,
            "total_sessions": len(set(ts.date() for ts in timestamps)),
            "average_per_day": len(timestamps) / max(len(set(ts.date() for ts in timestamps)), 1)
        }
    
    def _analyze_engagement_trends(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze engagement trends over time."""
        if not interactions:
            return {"trend": "stable"}
        
        # Group by date
        daily_counts = defaultdict(int)
        for interaction in interactions:
            if "timestamp" in interaction:
                try:
                    if isinstance(interaction["timestamp"], str):
                        date = datetime.fromisoformat(interaction["timestamp"]).date()
                    else:
                        date = interaction["timestamp"].date()
                    daily_counts[date] += 1
                except:
                    pass
        
        if len(daily_counts) < 2:
            return {"trend": "stable", "average_daily": len(interactions)}
        
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]
        
        # Calculate trend
        if len(counts) > 1:
            recent_avg = sum(counts[-7:]) / min(7, len(counts))
            earlier_avg = sum(counts[:-7]) / max(1, len(counts) - 7) if len(counts) > 7 else counts[0]
            
            if recent_avg > earlier_avg * 1.1:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "average_daily": sum(counts) / len(counts),
            "total_days": len(daily_counts)
        }
    
    def _identify_strengths(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify student strengths."""
        strengths = []
        
        # Analyze correct answers
        correct_count = sum(1 for i in interactions if i.get("correct", False))
        total_questions = sum(1 for i in interactions if "correct" in i)
        
        if total_questions > 0 and correct_count / total_questions > 0.8:
            strengths.append("High accuracy in assessments")
        
        # Analyze engagement
        if len(interactions) > 20:
            strengths.append("Consistent engagement")
        
        return strengths
    
    def _identify_weaknesses(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify areas for improvement."""
        weaknesses = []
        
        # Analyze correct answers
        correct_count = sum(1 for i in interactions if i.get("correct", False))
        total_questions = sum(1 for i in interactions if "correct" in i)
        
        if total_questions > 0 and correct_count / total_questions < 0.6:
            weaknesses.append("Needs improvement in assessments")
        
        # Analyze engagement
        if len(interactions) < 10:
            weaknesses.append("Low engagement - needs motivation")
        
        return weaknesses
    
    def _generate_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from patterns."""
        insights = []
        
        # Difficulty insights
        if patterns.get("difficulty_progression", {}).get("trend") == "increasing":
            insights.append("Student is progressing to more advanced topics")
        elif patterns.get("difficulty_progression", {}).get("trend") == "decreasing":
            insights.append("Consider reviewing fundamentals")
        
        # Engagement insights
        engagement = patterns.get("engagement_trends", {})
        if engagement.get("trend") == "increasing":
            insights.append("Engagement is improving - maintain momentum")
        elif engagement.get("trend") == "decreasing":
            insights.append("Engagement declining - consider intervention")
        
        # Strengths and weaknesses
        if patterns.get("strengths"):
            insights.append(f"Strengths: {', '.join(patterns['strengths'])}")
        if patterns.get("weaknesses"):
            insights.append(f"Areas for improvement: {', '.join(patterns['weaknesses'])}")
        
        return insights
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of analytics data."""
        return {
            "total_analyses": len(self.analytics_data),
            "students_analyzed": len(set(
                data.get("student_id") for data in self.analytics_data.values()
                if isinstance(data, dict) and "student_id" in data
            ))
        }




