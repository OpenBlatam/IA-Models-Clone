from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import threading
from typing import Dict, Any, List
from datetime import datetime, timezone
    from .config_v5 import config
    from config_v5 import config
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v5.0 - Metrics Module

Thread-safe performance monitoring and metrics collection.
"""

try:
except ImportError:


class UltraFastMetrics:
    """Thread-safe metrics collector for ultra-fast performance monitoring."""
    
    def __init__(self) -> Any:
        self._lock = threading.Lock()
        
        # Request metrics
        self.requests_total = 0
        self.requests_success = 0
        self.requests_error = 0
        self.batch_requests = 0
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Caption generation metrics
        self.total_captions_generated = 0
        self.quality_scores: List[float] = []
        
        # Performance metrics
        self.processing_times: List[float] = []
        self.start_time = time.time()
        
        # System metrics
        self.peak_concurrent_requests = 0
        self.current_active_requests = 0
    
    async def record_request_start(self) -> None:
        """Record the start of a request."""
        with self._lock:
            self.current_active_requests += 1
            self.peak_concurrent_requests = max(
                self.peak_concurrent_requests, 
                self.current_active_requests
            )
    
    async def record_request_end(self, success: bool, response_time: float, batch_size: int = 1) -> None:
        """Record the completion of a request."""
        with self._lock:
            self.current_active_requests = max(0, self.current_active_requests - 1)
            self.requests_total += 1
            
            if success:
                self.requests_success += 1
            else:
                self.requests_error += 1
            
            if batch_size > 1:
                self.batch_requests += 1
            
            # Store processing time (keep last 1000 for performance)
            self.processing_times.append(response_time)
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.cache_misses += 1
    
    def record_caption_generated(self, quality_score: float, count: int = 1) -> None:
        """Record caption generation with quality score."""
        with self._lock:
            self.total_captions_generated += count
            
            # Store quality scores (keep last 10k for performance)
            self.quality_scores.extend([quality_score] * count)
            if len(self.quality_scores) > 10000:
                self.quality_scores = self.quality_scores[-10000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance-related statistics."""
        with self._lock:
            uptime = time.time() - self.start_time
            
            # Calculate averages
            avg_response_time = (
                sum(self.processing_times) / len(self.processing_times) 
                if self.processing_times else 0
            )
            
            return {
                "requests_total": self.requests_total,
                "requests_success": self.requests_success,
                "requests_error": self.requests_error,
                "batch_requests": self.batch_requests,
                "success_rate": round(
                    (self.requests_success / max(1, self.requests_total)) * 100, 2
                ),
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "requests_per_second": round(self.requests_total / max(1, uptime), 2),
                "current_active_requests": self.current_active_requests,
                "peak_concurrent_requests": self.peak_concurrent_requests
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache-related statistics."""
        with self._lock:
            cache_total = self.cache_hits + self.cache_misses
            
            return {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "total_requests": cache_total,
                "hit_rate": round(
                    (self.cache_hits / max(1, cache_total)) * 100, 2
                )
            }
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get quality-related statistics."""
        with self._lock:
            uptime = time.time() - self.start_time
            
            # Calculate quality metrics
            avg_quality = (
                sum(self.quality_scores) / len(self.quality_scores)
                if self.quality_scores else 0
            )
            
            # Quality distribution
            high_quality_count = sum(1 for score in self.quality_scores if score >= 85)
            premium_rate = (
                (high_quality_count / len(self.quality_scores)) * 100
                if self.quality_scores else 0
            )
            
            return {
                "captions_generated": self.total_captions_generated,
                "avg_quality_score": round(avg_quality, 2),
                "premium_rate": round(premium_rate, 2),  # % with score >= 85
                "captions_per_second": round(
                    self.total_captions_generated / max(1, uptime), 2
                ),
                "quality_trend": self._calculate_quality_trend()
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-related statistics."""
        with self._lock:
            uptime = time.time() - self.start_time
            
            return {
                "uptime_seconds": round(uptime, 2),
                "uptime_hours": round(uptime / 3600, 2),
                "api_version": config.API_VERSION,
                "environment": config.ENVIRONMENT,
                "max_batch_size": config.MAX_BATCH_SIZE,
                "ai_workers": config.AI_PARALLEL_WORKERS
            }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get all statistics in a comprehensive format."""
        return {
            "performance": self.get_performance_stats(),
            "cache": self.get_cache_stats(),
            "quality": self.get_quality_stats(),
            "system": self.get_system_stats()
        }
    
    def _calculate_quality_trend(self) -> str:
        """Calculate quality trend based on recent scores."""
        if len(self.quality_scores) < 20:
            return "insufficient_data"
        
        recent_scores = self.quality_scores[-10:]
        older_scores = self.quality_scores[-20:-10]
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)
        
        if recent_avg > older_avg + 2:
            return "improving"
        elif recent_avg < older_avg - 2:
            return "declining"
        else:
            return "stable"
    
    def reset_metrics(self) -> None:
        """Reset all metrics (for testing or maintenance)."""
        with self._lock:
            self.requests_total = 0
            self.requests_success = 0
            self.requests_error = 0
            self.batch_requests = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_captions_generated = 0
            self.quality_scores.clear()
            self.processing_times.clear()
            self.start_time = time.time()
            self.peak_concurrent_requests = 0
            self.current_active_requests = 0


class PerformanceGrader:
    """Performance grading system for health checks."""
    
    @staticmethod
    def grade_performance(metrics: Dict[str, Any]) -> str:
        """Grade overall performance based on key metrics."""
        performance = metrics.get("performance", {})
        cache = metrics.get("cache", {})
        quality = metrics.get("quality", {})
        
        # Extract key metrics
        avg_response = performance.get("avg_response_time_ms", 1000)
        success_rate = performance.get("success_rate", 0)
        cache_hit_rate = cache.get("hit_rate", 0)
        avg_quality = quality.get("avg_quality_score", 0)
        
        # Scoring system
        score = 0
        
        # Response time scoring (40 points max)
        if avg_response < 50:
            score += 40
        elif avg_response < 100:
            score += 30
        elif avg_response < 200:
            score += 20
        elif avg_response < 500:
            score += 10
        
        # Success rate scoring (30 points max)
        if success_rate >= 99.5:
            score += 30
        elif success_rate >= 95:
            score += 20
        elif success_rate >= 90:
            score += 10
        elif success_rate >= 80:
            score += 5
        
        # Cache hit rate scoring (15 points max)
        if cache_hit_rate >= 90:
            score += 15
        elif cache_hit_rate >= 80:
            score += 12
        elif cache_hit_rate >= 70:
            score += 8
        elif cache_hit_rate >= 50:
            score += 5
        
        # Quality scoring (15 points max)
        if avg_quality >= 90:
            score += 15
        elif avg_quality >= 85:
            score += 12
        elif avg_quality >= 80:
            score += 8
        elif avg_quality >= 70:
            score += 5
        
        # Convert score to grade
        if score >= 90:
            return "A+ ULTRA-FAST"
        elif score >= 80:
            return "A FAST"
        elif score >= 70:
            return "B GOOD"
        elif score >= 60:
            return "C ACCEPTABLE"
        else:
            return "D NEEDS_OPTIMIZATION"


# Global metrics instance
metrics = UltraFastMetrics()
grader = PerformanceGrader()


# Export public interface
__all__ = [
    'UltraFastMetrics',
    'PerformanceGrader', 
    'metrics',
    'grader'
] 