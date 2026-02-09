"""
Analytics Service for Social Video Transcriber AI
Tracks usage metrics and generates insights
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from uuid import UUID

from ..config.settings import get_settings
from ..core.models import SupportedPlatform, TranscriptionStatus

logger = logging.getLogger(__name__)


@dataclass
class UsageMetrics:
    """Usage metrics for a time period"""
    total_transcriptions: int = 0
    successful_transcriptions: int = 0
    failed_transcriptions: int = 0
    total_video_duration: float = 0.0  # seconds
    total_processing_time: float = 0.0  # seconds
    total_words_transcribed: int = 0
    total_variants_generated: int = 0
    total_analyses: int = 0
    
    platform_breakdown: Dict[str, int] = field(default_factory=dict)
    language_breakdown: Dict[str, int] = field(default_factory=dict)
    framework_breakdown: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_transcriptions": self.total_transcriptions,
            "successful_transcriptions": self.successful_transcriptions,
            "failed_transcriptions": self.failed_transcriptions,
            "success_rate": round(
                self.successful_transcriptions / max(self.total_transcriptions, 1) * 100, 1
            ),
            "total_video_duration_seconds": self.total_video_duration,
            "total_video_duration_hours": round(self.total_video_duration / 3600, 2),
            "total_processing_time_seconds": self.total_processing_time,
            "avg_processing_time_seconds": round(
                self.total_processing_time / max(self.successful_transcriptions, 1), 2
            ),
            "total_words_transcribed": self.total_words_transcribed,
            "total_variants_generated": self.total_variants_generated,
            "total_analyses": self.total_analyses,
            "platform_breakdown": self.platform_breakdown,
            "language_breakdown": self.language_breakdown,
            "framework_breakdown": self.framework_breakdown,
        }


@dataclass
class DailyMetrics:
    """Daily metrics snapshot"""
    date: datetime
    metrics: UsageMetrics
    api_calls: int = 0
    unique_users: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    avg_download_time: float = 0.0
    avg_transcription_time: float = 0.0
    avg_analysis_time: float = 0.0
    avg_variant_generation_time: float = 0.0
    p95_processing_time: float = 0.0
    p99_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_download_time_seconds": round(self.avg_download_time, 2),
            "avg_transcription_time_seconds": round(self.avg_transcription_time, 2),
            "avg_analysis_time_seconds": round(self.avg_analysis_time, 2),
            "avg_variant_generation_time_seconds": round(self.avg_variant_generation_time, 2),
            "p95_processing_time_seconds": round(self.p95_processing_time, 2),
            "p99_processing_time_seconds": round(self.p99_processing_time, 2),
            "cache_hit_rate_percent": round(self.cache_hit_rate * 100, 1),
            "error_rate_percent": round(self.error_rate * 100, 2),
        }


class AnalyticsService:
    """Service for tracking and analyzing usage metrics"""
    
    def __init__(self):
        self.settings = get_settings()
        
        self._current_metrics = UsageMetrics()
        self._daily_metrics: Dict[str, DailyMetrics] = {}
        self._processing_times: List[float] = []
        self._errors: List[Dict[str, Any]] = []
        
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._api_key_usage: Dict[str, int] = defaultdict(int)
        self._hourly_requests: Dict[str, int] = defaultdict(int)
    
    def record_transcription_start(
        self,
        job_id: UUID,
        platform: Optional[SupportedPlatform] = None,
    ):
        """Record start of a transcription job"""
        self._current_metrics.total_transcriptions += 1
        
        if platform:
            platform_name = platform.value
            self._current_metrics.platform_breakdown[platform_name] = \
                self._current_metrics.platform_breakdown.get(platform_name, 0) + 1
    
    def record_transcription_complete(
        self,
        job_id: UUID,
        video_duration: float,
        processing_time: float,
        word_count: int,
        language: Optional[str] = None,
        framework: Optional[str] = None,
    ):
        """Record completion of a transcription job"""
        self._current_metrics.successful_transcriptions += 1
        self._current_metrics.total_video_duration += video_duration
        self._current_metrics.total_processing_time += processing_time
        self._current_metrics.total_words_transcribed += word_count
        
        self._processing_times.append(processing_time)
        if len(self._processing_times) > 1000:
            self._processing_times = self._processing_times[-1000:]
        
        if language:
            self._current_metrics.language_breakdown[language] = \
                self._current_metrics.language_breakdown.get(language, 0) + 1
        
        if framework:
            self._current_metrics.framework_breakdown[framework] = \
                self._current_metrics.framework_breakdown.get(framework, 0) + 1
    
    def record_transcription_failed(
        self,
        job_id: UUID,
        error: str,
        processing_time: float = 0,
    ):
        """Record failure of a transcription job"""
        self._current_metrics.failed_transcriptions += 1
        
        self._errors.append({
            "job_id": str(job_id),
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        if len(self._errors) > 100:
            self._errors = self._errors[-100:]
    
    def record_analysis(self, job_id: UUID):
        """Record an analysis operation"""
        self._current_metrics.total_analyses += 1
    
    def record_variant_generated(self, count: int = 1):
        """Record variant generation"""
        self._current_metrics.total_variants_generated += count
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self._cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self._cache_misses += 1
    
    def record_api_call(self, api_key_id: Optional[str] = None):
        """Record an API call"""
        hour_key = datetime.utcnow().strftime("%Y-%m-%d-%H")
        self._hourly_requests[hour_key] += 1
        
        if api_key_id:
            self._api_key_usage[api_key_id] += 1
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current usage metrics"""
        return self._current_metrics.to_dict()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        perf = PerformanceMetrics()
        
        if self._processing_times:
            sorted_times = sorted(self._processing_times)
            perf.p95_processing_time = sorted_times[int(len(sorted_times) * 0.95)]
            perf.p99_processing_time = sorted_times[int(len(sorted_times) * 0.99)]
        
        total_cache = self._cache_hits + self._cache_misses
        if total_cache > 0:
            perf.cache_hit_rate = self._cache_hits / total_cache
        
        total = self._current_metrics.total_transcriptions
        if total > 0:
            perf.error_rate = self._current_metrics.failed_transcriptions / total
        
        return perf.to_dict()
    
    def get_hourly_stats(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get hourly request statistics"""
        now = datetime.utcnow()
        stats = []
        
        for i in range(hours):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime("%Y-%m-%d-%H")
            
            stats.append({
                "hour": hour.strftime("%Y-%m-%d %H:00"),
                "requests": self._hourly_requests.get(hour_key, 0),
            })
        
        return list(reversed(stats))
    
    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent errors"""
        return self._errors[-limit:][::-1]
    
    def get_api_key_usage(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get API key usage statistics"""
        sorted_usage = sorted(
            self._api_key_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"api_key_id": key_id, "requests": count}
            for key_id, count in sorted_usage[:limit]
        ]
    
    def get_platform_insights(self) -> Dict[str, Any]:
        """Get insights by platform"""
        breakdown = self._current_metrics.platform_breakdown
        total = sum(breakdown.values())
        
        return {
            "breakdown": breakdown,
            "percentages": {
                platform: round(count / max(total, 1) * 100, 1)
                for platform, count in breakdown.items()
            },
            "most_used": max(breakdown.keys(), key=lambda k: breakdown[k]) if breakdown else None,
        }
    
    def get_language_insights(self) -> Dict[str, Any]:
        """Get insights by language"""
        breakdown = self._current_metrics.language_breakdown
        total = sum(breakdown.values())
        
        return {
            "breakdown": breakdown,
            "percentages": {
                lang: round(count / max(total, 1) * 100, 1)
                for lang, count in breakdown.items()
            },
            "most_common": max(breakdown.keys(), key=lambda k: breakdown[k]) if breakdown else None,
        }
    
    def get_framework_insights(self) -> Dict[str, Any]:
        """Get insights by content framework"""
        breakdown = self._current_metrics.framework_breakdown
        total = sum(breakdown.values())
        
        return {
            "breakdown": breakdown,
            "percentages": {
                fw: round(count / max(total, 1) * 100, 1)
                for fw, count in breakdown.items()
            },
            "most_common": max(breakdown.keys(), key=lambda k: breakdown[k]) if breakdown else None,
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data for dashboard"""
        return {
            "usage_metrics": self.get_current_metrics(),
            "performance_metrics": self.get_performance_metrics(),
            "hourly_stats": self.get_hourly_stats(24),
            "platform_insights": self.get_platform_insights(),
            "language_insights": self.get_language_insights(),
            "framework_insights": self.get_framework_insights(),
            "recent_errors": self.get_top_errors(5),
            "top_api_keys": self.get_api_key_usage(5),
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self._current_metrics = UsageMetrics()
        self._processing_times = []
        self._errors = []
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Analytics metrics reset")


_analytics_service: Optional[AnalyticsService] = None


def get_analytics_service() -> AnalyticsService:
    """Get analytics service singleton"""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsService()
    return _analytics_service












