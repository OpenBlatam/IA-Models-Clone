"""
Result Aggregator
=================

System for aggregating and analyzing results.
"""

import logging
import statistics
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Aggregation result."""
    name: str
    total: int
    successful: int
    failed: int
    success_rate: float
    avg_duration: float
    min_duration: float
    max_duration: float
    median_duration: float
    p95_duration: float
    p99_duration: float
    by_category: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ResultAggregator:
    """Result aggregator."""
    
    def __init__(self):
        """Initialize result aggregator."""
        self.results: List[Dict[str, Any]] = []
        self.max_results = 10000
    
    def add_result(
        self,
        success: bool,
        duration: float,
        category: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a result.
        
        Args:
            success: Whether operation was successful
            duration: Operation duration
            category: Optional category
            error: Optional error message
            metadata: Optional metadata
        """
        result = {
            "success": success,
            "duration": duration,
            "category": category,
            "error": error,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }
        
        self.results.append(result)
        
        # Limit results
        if len(self.results) > self.max_results:
            self.results = self.results[-self.max_results:]
    
    def aggregate(
        self,
        name: str,
        category: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> AggregationResult:
        """
        Aggregate results.
        
        Args:
            name: Aggregation name
            category: Optional category filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Aggregation result
        """
        # Filter results
        filtered = self.results
        
        if category:
            filtered = [r for r in filtered if r.get("category") == category]
        if start_time:
            filtered = [r for r in filtered if r.get("timestamp", datetime.now()) >= start_time]
        if end_time:
            filtered = [r for r in filtered if r.get("timestamp", datetime.now()) <= end_time]
        
        if not filtered:
            return AggregationResult(
                name=name,
                total=0,
                successful=0,
                failed=0,
                success_rate=0.0,
                avg_duration=0.0,
                min_duration=0.0,
                max_duration=0.0,
                median_duration=0.0,
                p95_duration=0.0,
                p99_duration=0.0
            )
        
        # Calculate statistics
        total = len(filtered)
        successful = len([r for r in filtered if r["success"]])
        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0
        
        durations = [r["duration"] for r in filtered]
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        median_duration = statistics.median(durations)
        
        # Percentiles
        sorted_durations = sorted(durations)
        p95_duration = sorted_durations[int(len(sorted_durations) * 0.95)] if len(sorted_durations) > 1 else sorted_durations[0]
        p99_duration = sorted_durations[int(len(sorted_durations) * 0.99)] if len(sorted_durations) > 1 else sorted_durations[0]
        
        # By category
        by_category = defaultdict(int)
        for r in filtered:
            cat = r.get("category", "unknown")
            by_category[cat] += 1
        
        # Errors
        errors = [r["error"] for r in filtered if r.get("error") and r["error"] not in [None, ""]]
        unique_errors = list(set(errors))[:10]  # Top 10 unique errors
        
        return AggregationResult(
            name=name,
            total=total,
            successful=successful,
            failed=failed,
            success_rate=success_rate,
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            median_duration=median_duration,
            p95_duration=p95_duration,
            p99_duration=p99_duration,
            by_category=dict(by_category),
            errors=unique_errors
        )
    
    def get_recent_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent results."""
        return self.results[-limit:]
    
    def clear(self, before: Optional[datetime] = None):
        """
        Clear results.
        
        Args:
            before: Optional timestamp (clears results before this time)
        """
        if before:
            self.results = [r for r in self.results if r.get("timestamp", datetime.now()) >= before]
        else:
            self.results.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "total_results": len(self.results),
            "max_results": self.max_results
        }

