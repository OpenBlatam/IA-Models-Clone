"""
Monitoring and Metrics for Music Generation
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GenerationMetrics:
    """Metrics for music generation"""
    
    def __init__(self):
        """Initialize metrics"""
        self.generation_times = deque(maxlen=1000)
        self.generation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_generations = 0
        logger.info("GenerationMetrics initialized")
    
    def record_generation(
        self,
        duration: float,
        text_length: int,
        cache_hit: bool = False
    ):
        """
        Record generation metrics
        
        Args:
            duration: Generation duration in seconds
            text_length: Length of input text
            cache_hit: Whether it was a cache hit
        """
        self.generation_times.append(duration)
        self.total_generations += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Categorize by text length
        if text_length < 20:
            category = "short"
        elif text_length < 50:
            category = "medium"
        else:
            category = "long"
        
        self.generation_counts[category] += 1
    
    def record_error(self, error_type: str):
        """Record error"""
        self.error_counts[error_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        if not self.generation_times:
            return {
                "total_generations": 0,
                "avg_generation_time": 0,
                "cache_hit_rate": 0
            }
        
        avg_time = sum(self.generation_times) / len(self.generation_times)
        min_time = min(self.generation_times)
        max_time = max(self.generation_times)
        
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_generations": self.total_generations,
            "avg_generation_time_seconds": avg_time,
            "min_generation_time_seconds": min_time,
            "max_generation_time_seconds": max_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": cache_hit_rate,
            "generation_counts_by_length": dict(self.generation_counts),
            "error_counts": dict(self.error_counts)
        }


class SystemMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        """Initialize system monitor"""
        self.metrics = GenerationMetrics()
        logger.info("SystemMonitor initialized")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            import psutil
            import torch
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            info = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3)
            }
            
            if torch.cuda.is_available():
                info["gpu_available"] = True
                info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            else:
                info["gpu_available"] = False
            
            return info
        except ImportError:
            return {"error": "psutil not available"}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return {
            "generation_metrics": self.metrics.get_stats(),
            "system_info": self.get_system_info(),
            "timestamp": datetime.now().isoformat()
        }


class PerformanceTracker:
    """Track performance over time"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance tracker
        
        Args:
            window_size: Size of tracking window
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.durations = deque(maxlen=window_size)
        self.cache_hits = deque(maxlen=window_size)
        logger.info("PerformanceTracker initialized")
    
    def track(
        self,
        duration: float,
        cache_hit: bool = False
    ):
        """
        Track performance
        
        Args:
            duration: Generation duration
            cache_hit: Whether it was a cache hit
        """
        self.timestamps.append(datetime.now())
        self.durations.append(duration)
        self.cache_hits.append(1 if cache_hit else 0)
    
    def get_performance_trend(self) -> Dict[str, Any]:
        """Get performance trend"""
        if not self.durations:
            return {}
        
        recent_durations = list(self.durations)[-10:]
        avg_recent = sum(recent_durations) / len(recent_durations)
        avg_overall = sum(self.durations) / len(self.durations)
        
        cache_hit_rate = sum(self.cache_hits) / len(self.cache_hits) * 100 if self.cache_hits else 0
        
        return {
            "avg_recent_duration": avg_recent,
            "avg_overall_duration": avg_overall,
            "trend": "improving" if avg_recent < avg_overall else "degrading",
            "cache_hit_rate_percent": cache_hit_rate,
            "samples": len(self.durations)
        }

