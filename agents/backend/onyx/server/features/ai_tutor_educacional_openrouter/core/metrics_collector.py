"""
Metrics collector for tracking tutor performance and usage.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class TutorMetrics:
    """Metrics for tutor performance."""
    total_questions: int = 0
    total_explanations: int = 0
    total_exercises: int = 0
    total_quizzes: int = 0
    average_response_time: float = 0.0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    subjects_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    difficulty_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_updated: Optional[str] = None


class MetricsCollector:
    """
    Collects and aggregates metrics for the AI Tutor system.
    """
    
    def __init__(self):
        self.metrics = TutorMetrics()
        self.response_times: List[float] = []
        self.error_log: List[Dict] = []
        self.daily_stats: Dict[str, Dict] = {}
    
    def record_question(self, subject: Optional[str] = None, difficulty: Optional[str] = None):
        """Record a question being asked."""
        self.metrics.total_questions += 1
        if subject:
            self.metrics.subjects_usage[subject] += 1
        if difficulty:
            self.metrics.difficulty_usage[difficulty] += 1
        self._update_timestamp()
    
    def record_explanation(self):
        """Record an explanation being generated."""
        self.metrics.total_explanations += 1
        self._update_timestamp()
    
    def record_exercise(self):
        """Record an exercise being generated."""
        self.metrics.total_exercises += 1
        self._update_timestamp()
    
    def record_quiz(self):
        """Record a quiz being generated."""
        self.metrics.total_quizzes += 1
        self._update_timestamp()
    
    def record_response_time(self, time: float):
        """Record response time for a request."""
        self.response_times.append(time)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        if self.response_times:
            self.metrics.average_response_time = sum(self.response_times) / len(self.response_times)
    
    def record_tokens(self, tokens: int, cost: float = 0.0):
        """Record token usage and cost."""
        self.metrics.total_tokens_used += tokens
        self.metrics.total_cost += cost
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics.cache_misses += 1
    
    def record_error(self, error_type: str, error_message: str):
        """Record an error."""
        self.metrics.errors += 1
        self.error_log.append({
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return asdict(self.metrics)
    
    def get_daily_stats(self, days: int = 7) -> Dict:
        """Get statistics for the last N days."""
        today = datetime.now().date()
        stats = {}
        
        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.isoformat()
            stats[date_str] = self.daily_stats.get(date_str, {})
        
        return stats
    
    def get_top_subjects(self, limit: int = 5) -> List[tuple]:
        """Get top subjects by usage."""
        return sorted(
            self.metrics.subjects_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.metrics.cache_hits + self.metrics.cache_misses
        if total == 0:
            return 0.0
        return self.metrics.cache_hits / total
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = TutorMetrics()
        self.response_times = []
        self.error_log = []
    
    def _update_timestamp(self):
        """Update last updated timestamp."""
        self.metrics.last_updated = datetime.now().isoformat()
        
        # Update daily stats
        today = datetime.now().date().isoformat()
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                "questions": 0,
                "explanations": 0,
                "exercises": 0
            }
        
        # This would need to be called with context, simplified here
        # self.daily_stats[today]["questions"] += 1






