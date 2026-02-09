"""
Dashboard analytics for real-time insights and visualizations.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class DashboardAnalytics:
    """
    Provides analytics data for dashboard visualizations.
    """
    
    def __init__(self, metrics_collector, learning_analyzer, gamification_system):
        self.metrics_collector = metrics_collector
        self.learning_analyzer = learning_analyzer
        self.gamification_system = gamification_system
    
    def get_overview_stats(self) -> Dict[str, Any]:
        """Get overview statistics for dashboard."""
        metrics = self.metrics_collector.get_metrics()
        
        return {
            "total_students": len(self.learning_analyzer.student_profiles),
            "total_questions": metrics.get("total_questions", 0),
            "total_explanations": metrics.get("total_explanations", 0),
            "total_exercises": metrics.get("total_exercises", 0),
            "total_quizzes": metrics.get("total_quizzes", 0),
            "average_response_time": metrics.get("average_response_time", 0.0),
            "cache_hit_rate": self.metrics_collector.get_cache_hit_rate(),
            "total_tokens_used": metrics.get("total_tokens_used", 0),
            "total_cost": metrics.get("total_cost", 0.0)
        }
    
    def get_activity_timeline(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get activity timeline for the last N days."""
        timeline = []
        today = datetime.now().date()
        
        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.isoformat()
            
            # This would need actual daily tracking - simplified here
            timeline.append({
                "date": date_str,
                "questions": 0,
                "explanations": 0,
                "exercises": 0,
                "quizzes": 0
            })
        
        return list(reversed(timeline))
    
    def get_subject_distribution(self) -> Dict[str, int]:
        """Get distribution of questions by subject."""
        metrics = self.metrics_collector.get_metrics()
        return metrics.get("subjects_usage", {})
    
    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of questions by difficulty."""
        metrics = self.metrics_collector.get_metrics()
        return metrics.get("difficulty_usage", {})
    
    def get_top_students(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing students."""
        leaderboard = self.gamification_system.get_leaderboard(limit)
        
        result = []
        for entry in leaderboard:
            student_id = entry["student_id"]
            profile = self.learning_analyzer.student_profiles.get(student_id, {})
            
            result.append({
                "student_id": student_id,
                "points": entry["points"],
                "level": entry["level"],
                "rank": entry["rank"],
                "overall_performance": profile.get("overall_performance", 0.0),
                "subjects_studied": len(profile.get("subjects", {}))
            })
        
        return result
    
    def get_performance_trends(
        self,
        student_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get performance trends for a student."""
        profile = self.learning_analyzer.student_profiles.get(student_id, {})
        
        # Simplified - would need actual historical data
        trends = {
            "student_id": student_id,
            "period_days": days,
            "overall_trend": "improving",
            "subject_trends": {}
        }
        
        for subject, topics in profile.get("subjects", {}).items():
            avg_performance = sum(
                t.get("performance", 0.0) for t in topics.values()
            ) / len(topics) if topics else 0.0
            
            trends["subject_trends"][subject] = {
                "average_performance": avg_performance,
                "topics_count": len(topics),
                "trend": "stable"
            }
        
        return trends
    
    def get_engagement_metrics(self) -> Dict[str, Any]:
        """Get engagement metrics."""
        total_students = len(self.learning_analyzer.student_profiles)
        
        active_students = sum(
            1 for student_id in self.learning_analyzer.student_profiles.keys()
            if self.gamification_system.student_points.get(student_id, 0) > 0
        )
        
        avg_points = sum(
            self.gamification_system.student_points.values()
        ) / total_students if total_students > 0 else 0
        
        return {
            "total_students": total_students,
            "active_students": active_students,
            "engagement_rate": (active_students / total_students * 100) if total_students > 0 else 0,
            "average_points": avg_points,
            "total_badges_awarded": sum(
                len(badges) for badges in self.gamification_system.student_badges.values()
            )
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights and patterns."""
        all_strengths = defaultdict(int)
        all_weaknesses = defaultdict(int)
        
        for student_id in self.learning_analyzer.student_profiles.keys():
            sw = self.learning_analyzer.get_strengths_and_weaknesses(student_id)
            
            for subject, topics in sw.get("strengths", {}).items():
                all_strengths[subject] += len(topics)
            
            for subject, topics in sw.get("weaknesses", {}).items():
                all_weaknesses[subject] += len(topics)
        
        return {
            "common_strengths": dict(all_strengths),
            "common_weaknesses": dict(all_weaknesses),
            "most_studied_subjects": self.get_subject_distribution(),
            "recommendations": self._generate_insights_recommendations()
        }
    
    def _generate_insights_recommendations(self) -> List[str]:
        """Generate recommendations based on insights."""
        return [
            "Enfocar más recursos en las materias más populares",
            "Crear más contenido para áreas de debilidad común",
            "Implementar estrategias de gamificación adicionales"
        ]






