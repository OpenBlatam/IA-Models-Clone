"""
Analytics Service
Tracks metrics and statistics for video generation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Tracks analytics for video generation"""
    
    def __init__(self):
        self.metrics = {
            "total_videos": 0,
            "successful_videos": 0,
            "failed_videos": 0,
            "total_duration": 0.0,
            "total_size": 0,
            "average_generation_time": 0.0,
            "generation_times": [],
            "errors": defaultdict(int),
            "style_usage": defaultdict(int),
            "voice_usage": defaultdict(int),
            "resolution_usage": defaultdict(int),
        }
        self.job_metrics: Dict[str, Dict[str, Any]] = {}
    
    def record_generation_start(self, video_id: str, request_data: Dict[str, Any]):
        """Record start of video generation"""
        self.job_metrics[video_id] = {
            "start_time": datetime.utcnow(),
            "request_data": request_data,
            "status": "processing",
        }
        self.metrics["total_videos"] += 1
    
    def record_generation_complete(
        self,
        video_id: str,
        duration: float,
        file_size: int,
        generation_time: float
    ):
        """Record successful video generation"""
        if video_id in self.job_metrics:
            job = self.job_metrics[video_id]
            job["end_time"] = datetime.utcnow()
            job["status"] = "completed"
            job["duration"] = duration
            job["file_size"] = file_size
            job["generation_time"] = generation_time
            
            # Update request data for analytics
            request_data = job.get("request_data", {})
            video_config = request_data.get("video_config", {})
            audio_config = request_data.get("audio_config", {})
            
            self.metrics["style_usage"][video_config.get("style", "unknown")] += 1
            self.metrics["voice_usage"][audio_config.get("voice", "unknown")] += 1
            self.metrics["resolution_usage"][video_config.get("resolution", "unknown")] += 1
        
        self.metrics["successful_videos"] += 1
        self.metrics["total_duration"] += duration
        self.metrics["total_size"] += file_size
        self.metrics["generation_times"].append(generation_time)
        
        # Update average
        if self.metrics["generation_times"]:
            self.metrics["average_generation_time"] = sum(
                self.metrics["generation_times"]
            ) / len(self.metrics["generation_times"])
    
    def record_generation_failure(self, video_id: str, error: str):
        """Record failed video generation"""
        if video_id in self.job_metrics:
            job = self.job_metrics[video_id]
            job["end_time"] = datetime.utcnow()
            job["status"] = "failed"
            job["error"] = error
        
        self.metrics["failed_videos"] += 1
        self.metrics["errors"][error] += 1
    
    def get_metrics(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get analytics metrics"""
        metrics = self.metrics.copy()
        
        # Filter by time range if specified
        if time_range:
            cutoff_time = datetime.utcnow() - time_range
            filtered_jobs = {
                k: v for k, v in self.job_metrics.items()
                if v.get("start_time", datetime.min) >= cutoff_time
            }
            
            # Recalculate metrics for time range
            metrics["total_videos"] = len(filtered_jobs)
            metrics["successful_videos"] = sum(
                1 for v in filtered_jobs.values() if v.get("status") == "completed"
            )
            metrics["failed_videos"] = sum(
                1 for v in filtered_jobs.values() if v.get("status") == "failed"
            )
        
        # Calculate success rate
        if metrics["total_videos"] > 0:
            metrics["success_rate"] = (
                metrics["successful_videos"] / metrics["total_videos"]
            ) * 100
        else:
            metrics["success_rate"] = 0.0
        
        # Calculate average duration
        if metrics["successful_videos"] > 0:
            metrics["average_duration"] = (
                metrics["total_duration"] / metrics["successful_videos"]
            )
        else:
            metrics["average_duration"] = 0.0
        
        # Calculate average file size
        if metrics["successful_videos"] > 0:
            metrics["average_file_size"] = (
                metrics["total_size"] / metrics["successful_videos"]
            )
        else:
            metrics["average_file_size"] = 0
        
        return metrics
    
    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top errors by frequency"""
        errors = sorted(
            self.metrics["errors"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [{"error": error, "count": count} for error, count in errors]
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "styles": dict(self.metrics["style_usage"]),
            "voices": dict(self.metrics["voice_usage"]),
            "resolutions": dict(self.metrics["resolution_usage"]),
        }
    
    def reset_metrics(self):
        """Reset all metrics (use with caution)"""
        self.metrics = {
            "total_videos": 0,
            "successful_videos": 0,
            "failed_videos": 0,
            "total_duration": 0.0,
            "total_size": 0,
            "average_generation_time": 0.0,
            "generation_times": [],
            "errors": defaultdict(int),
            "style_usage": defaultdict(int),
            "voice_usage": defaultdict(int),
            "resolution_usage": defaultdict(int),
        }
        self.job_metrics.clear()


_analytics_service: Optional[AnalyticsService] = None


def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance (singleton)"""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsService()
    return _analytics_service

