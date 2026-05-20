"""
Advanced analytics system for content analysis insights
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics

from metrics import get_system_metrics, get_endpoint_metrics, get_health_metrics
from cache import get_cache_stats

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsReport:
    """Analytics report structure"""
    id: str
    report_type: str
    data: Dict[str, Any]
    generated_at: float = field(default_factory=time.time)
    period_start: Optional[float] = None
    period_end: Optional[float] = None


class AnalyticsEngine:
    """Advanced analytics engine"""
    
    def __init__(self):
        self._reports: Dict[str, AnalyticsReport] = {}
        self._content_analysis_history: List[Dict[str, Any]] = []
        self._similarity_analysis_history: List[Dict[str, Any]] = []
        self._quality_analysis_history: List[Dict[str, Any]] = []
    
    def record_analysis(self, analysis_type: str, data: Dict[str, Any]) -> None:
        """Record analysis data for analytics"""
        timestamp = time.time()
        record = {
            "timestamp": timestamp,
            "type": analysis_type,
            "data": data
        }
        
        if analysis_type == "content":
            self._content_analysis_history.append(record)
        elif analysis_type == "similarity":
            self._similarity_analysis_history.append(record)
        elif analysis_type == "quality":
            self._quality_analysis_history.append(record)
        
        # Keep only last 1000 records to prevent memory issues
        if len(self._content_analysis_history) > 1000:
            self._content_analysis_history = self._content_analysis_history[-1000:]
        if len(self._similarity_analysis_history) > 1000:
            self._similarity_analysis_history = self._similarity_analysis_history[-1000:]
        if len(self._quality_analysis_history) > 1000:
            self._quality_analysis_history = self._quality_analysis_history[-1000:]
    
    def generate_performance_report(self) -> AnalyticsReport:
        """Generate performance analytics report"""
        system_metrics = get_system_metrics()
        endpoint_metrics = get_endpoint_metrics()
        health_metrics = get_health_metrics()
        cache_stats = get_cache_stats()
        
        # Calculate performance insights
        total_requests = system_metrics.get("total_requests", 0)
        avg_response_time = system_metrics.get("average_response_time", 0)
        error_rate = system_metrics.get("error_rate", 0)
        
        # Endpoint performance analysis
        endpoint_performance = {}
        for endpoint, metrics in endpoint_metrics.items():
            endpoint_performance[endpoint] = {
                "request_count": metrics.get("count", 0),
                "avg_response_time": metrics.get("average_response_time", 0),
                "error_rate": metrics.get("error_rate", 0),
                "performance_score": self._calculate_performance_score(metrics)
            }
        
        # Performance trends
        performance_trends = self._analyze_performance_trends()
        
        report_data = {
            "summary": {
                "total_requests": total_requests,
                "average_response_time": round(avg_response_time, 4),
                "error_rate": round(error_rate, 2),
                "uptime": health_metrics.get("uptime", 0),
                "cache_hit_rate": self._calculate_cache_hit_rate(cache_stats)
            },
            "endpoint_performance": endpoint_performance,
            "performance_trends": performance_trends,
            "recommendations": self._generate_performance_recommendations(system_metrics, endpoint_metrics)
        }
        
        report = AnalyticsReport(
            id=f"performance_{int(time.time())}",
            report_type="performance",
            data=report_data
        )
        
        self._reports[report.id] = report
        return report
    
    def generate_content_insights_report(self) -> AnalyticsReport:
        """Generate content analysis insights report"""
        if not self._content_analysis_history:
            return AnalyticsReport(
                id=f"content_insights_{int(time.time())}",
                report_type="content_insights",
                data={"message": "No content analysis data available"}
            )
        
        # Analyze content patterns
        redundancy_scores = []
        word_counts = []
        unique_word_ratios = []
        
        for record in self._content_analysis_history:
            data = record["data"]
            redundancy_scores.append(data.get("redundancy_score", 0))
            word_counts.append(data.get("word_count", 0))
            if data.get("word_count", 0) > 0:
                unique_ratio = data.get("unique_words", 0) / data.get("word_count", 1)
                unique_word_ratios.append(unique_ratio)
        
        # Calculate statistics
        content_stats = {
            "total_analyses": len(self._content_analysis_history),
            "redundancy_analysis": {
                "average": round(statistics.mean(redundancy_scores), 4),
                "median": round(statistics.median(redundancy_scores), 4),
                "std_dev": round(statistics.stdev(redundancy_scores) if len(redundancy_scores) > 1 else 0, 4),
                "min": round(min(redundancy_scores), 4),
                "max": round(max(redundancy_scores), 4)
            },
            "word_count_analysis": {
                "average": round(statistics.mean(word_counts), 2),
                "median": round(statistics.median(word_counts), 2),
                "std_dev": round(statistics.stdev(word_counts) if len(word_counts) > 1 else 0, 2),
                "min": min(word_counts),
                "max": max(word_counts)
            },
            "uniqueness_analysis": {
                "average_unique_ratio": round(statistics.mean(unique_word_ratios), 4),
                "median_unique_ratio": round(statistics.median(unique_word_ratios), 4)
            }
        }
        
        # Content quality distribution
        quality_distribution = self._analyze_content_quality_distribution()
        
        # Trends over time
        time_trends = self._analyze_content_trends()
        
        report_data = {
            "statistics": content_stats,
            "quality_distribution": quality_distribution,
            "time_trends": time_trends,
            "insights": self._generate_content_insights(content_stats)
        }
        
        report = AnalyticsReport(
            id=f"content_insights_{int(time.time())}",
            report_type="content_insights",
            data=report_data
        )
        
        self._reports[report.id] = report
        return report
    
    def generate_similarity_insights_report(self) -> AnalyticsReport:
        """Generate similarity analysis insights report"""
        if not self._similarity_analysis_history:
            return AnalyticsReport(
                id=f"similarity_insights_{int(time.time())}",
                report_type="similarity_insights",
                data={"message": "No similarity analysis data available"}
            )
        
        # Analyze similarity patterns
        similarity_scores = []
        similar_pairs = 0
        threshold_usage = Counter()
        
        for record in self._similarity_analysis_history:
            data = record["data"]
            similarity_scores.append(data.get("similarity_score", 0))
            if data.get("is_similar", False):
                similar_pairs += 1
            # Note: threshold would need to be passed in the data
        
        # Calculate statistics
        similarity_stats = {
            "total_comparisons": len(self._similarity_analysis_history),
            "similar_pairs": similar_pairs,
            "similarity_rate": round(similar_pairs / len(self._similarity_analysis_history) * 100, 2),
            "similarity_analysis": {
                "average": round(statistics.mean(similarity_scores), 4),
                "median": round(statistics.median(similarity_scores), 4),
                "std_dev": round(statistics.stdev(similarity_scores) if len(similarity_scores) > 1 else 0, 4),
                "min": round(min(similarity_scores), 4),
                "max": round(max(similarity_scores), 4)
            }
        }
        
        # Common words analysis
        common_words_analysis = self._analyze_common_words_patterns()
        
        report_data = {
            "statistics": similarity_stats,
            "common_words_analysis": common_words_analysis,
            "insights": self._generate_similarity_insights(similarity_stats)
        }
        
        report = AnalyticsReport(
            id=f"similarity_insights_{int(time.time())}",
            report_type="similarity_insights",
            data=report_data
        )
        
        self._reports[report.id] = report
        return report
    
    def generate_quality_insights_report(self) -> AnalyticsReport:
        """Generate quality analysis insights report"""
        if not self._quality_analysis_history:
            return AnalyticsReport(
                id=f"quality_insights_{int(time.time())}",
                report_type="quality_insights",
                data={"message": "No quality analysis data available"}
            )
        
        # Analyze quality patterns
        readability_scores = []
        complexity_scores = []
        quality_ratings = Counter()
        
        for record in self._quality_analysis_history:
            data = record["data"]
            readability_scores.append(data.get("readability_score", 0))
            complexity_scores.append(data.get("complexity_score", 0))
            quality_ratings[data.get("quality_rating", "Unknown")] += 1
        
        # Calculate statistics
        quality_stats = {
            "total_assessments": len(self._quality_analysis_history),
            "readability_analysis": {
                "average": round(statistics.mean(readability_scores), 2),
                "median": round(statistics.median(readability_scores), 2),
                "std_dev": round(statistics.stdev(readability_scores) if len(readability_scores) > 1 else 0, 2),
                "min": round(min(readability_scores), 2),
                "max": round(max(readability_scores), 2)
            },
            "complexity_analysis": {
                "average": round(statistics.mean(complexity_scores), 2),
                "median": round(statistics.median(complexity_scores), 2),
                "std_dev": round(statistics.stdev(complexity_scores) if len(complexity_scores) > 1 else 0, 2),
                "min": round(min(complexity_scores), 2),
                "max": round(max(complexity_scores), 2)
            },
            "quality_distribution": dict(quality_ratings)
        }
        
        # Quality trends
        quality_trends = self._analyze_quality_trends()
        
        report_data = {
            "statistics": quality_stats,
            "quality_trends": quality_trends,
            "insights": self._generate_quality_insights(quality_stats)
        }
        
        report = AnalyticsReport(
            id=f"quality_insights_{int(time.time())}",
            report_type="quality_insights",
            data=report_data
        )
        
        self._reports[report.id] = report
        return report
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score for an endpoint"""
        response_time = metrics.get("average_response_time", 0)
        error_rate = metrics.get("error_rate", 0)
        
        # Simple scoring: lower response time and error rate = higher score
        time_score = max(0, 100 - (response_time * 100))  # Penalize slow responses
        error_score = max(0, 100 - error_rate)  # Penalize errors
        
        return round((time_score + error_score) / 2, 2)
    
    def _calculate_cache_hit_rate(self, cache_stats: Dict[str, Any]) -> float:
        """Calculate cache hit rate"""
        # This would need to be implemented based on actual cache statistics
        return 0.0
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        # This would analyze historical performance data
        return {"trend": "stable", "change_percentage": 0.0}
    
    def _analyze_content_quality_distribution(self) -> Dict[str, Any]:
        """Analyze content quality distribution"""
        # This would analyze the distribution of content quality scores
        return {"high_quality": 0, "medium_quality": 0, "low_quality": 0}
    
    def _analyze_content_trends(self) -> Dict[str, Any]:
        """Analyze content analysis trends over time"""
        # This would analyze trends in content analysis over time
        return {"trend": "stable", "change_percentage": 0.0}
    
    def _analyze_common_words_patterns(self) -> Dict[str, Any]:
        """Analyze common words patterns in similarity analysis"""
        # This would analyze patterns in common words found
        return {"most_common_words": [], "average_common_words": 0}
    
    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        # This would analyze trends in quality scores over time
        return {"trend": "stable", "change_percentage": 0.0}
    
    def _generate_performance_recommendations(self, system_metrics: Dict[str, Any], 
                                            endpoint_metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if system_metrics.get("error_rate", 0) > 5:
            recommendations.append("Consider investigating high error rate")
        
        if system_metrics.get("average_response_time", 0) > 1.0:
            recommendations.append("Consider optimizing slow endpoints")
        
        return recommendations
    
    def _generate_content_insights(self, content_stats: Dict[str, Any]) -> List[str]:
        """Generate content analysis insights"""
        insights = []
        
        avg_redundancy = content_stats["redundancy_analysis"]["average"]
        if avg_redundancy > 0.5:
            insights.append("High average redundancy detected in analyzed content")
        elif avg_redundancy < 0.2:
            insights.append("Low redundancy detected - content appears diverse")
        
        return insights
    
    def _generate_similarity_insights(self, similarity_stats: Dict[str, Any]) -> List[str]:
        """Generate similarity analysis insights"""
        insights = []
        
        similarity_rate = similarity_stats["similarity_rate"]
        if similarity_rate > 50:
            insights.append("High similarity rate detected in compared texts")
        elif similarity_rate < 10:
            insights.append("Low similarity rate - texts appear mostly unique")
        
        return insights
    
    def _generate_quality_insights(self, quality_stats: Dict[str, Any]) -> List[str]:
        """Generate quality analysis insights"""
        insights = []
        
        avg_readability = quality_stats["readability_analysis"]["average"]
        if avg_readability > 70:
            insights.append("High readability scores detected in analyzed content")
        elif avg_readability < 30:
            insights.append("Low readability scores - content may need improvement")
        
        return insights
    
    def get_report(self, report_id: str) -> Optional[AnalyticsReport]:
        """Get analytics report by ID"""
        return self._reports.get(report_id)
    
    def get_all_reports(self) -> List[AnalyticsReport]:
        """Get all analytics reports"""
        return list(self._reports.values())
    
    def cleanup_old_reports(self, max_age_hours: int = 24) -> int:
        """Clean up old reports"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for report_id, report in self._reports.items():
            if current_time - report.generated_at > max_age_seconds:
                to_remove.append(report_id)
        
        for report_id in to_remove:
            del self._reports[report_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old reports")
        return len(to_remove)


# Global analytics engine
analytics_engine = AnalyticsEngine()


def record_analysis(analysis_type: str, data: Dict[str, Any]) -> None:
    """Record analysis data for analytics"""
    analytics_engine.record_analysis(analysis_type, data)


def generate_performance_report() -> AnalyticsReport:
    """Generate performance analytics report"""
    return analytics_engine.generate_performance_report()


def generate_content_insights_report() -> AnalyticsReport:
    """Generate content insights report"""
    return analytics_engine.generate_content_insights_report()


def generate_similarity_insights_report() -> AnalyticsReport:
    """Generate similarity insights report"""
    return analytics_engine.generate_similarity_insights_report()


def generate_quality_insights_report() -> AnalyticsReport:
    """Generate quality insights report"""
    return analytics_engine.generate_quality_insights_report()


def get_analytics_report(report_id: str) -> Optional[AnalyticsReport]:
    """Get analytics report"""
    return analytics_engine.get_report(report_id)


def get_all_analytics_reports() -> List[AnalyticsReport]:
    """Get all analytics reports"""
    return analytics_engine.get_all_reports()


def cleanup_old_analytics_reports(max_age_hours: int = 24) -> int:
    """Clean up old analytics reports"""
    return analytics_engine.cleanup_old_reports(max_age_hours)


