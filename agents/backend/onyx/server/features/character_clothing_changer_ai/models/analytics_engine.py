"""
Analytics Engine for Flux2 Clothing Changer
===========================================

Advanced analytics and metrics collection for performance monitoring.
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessingEvent:
    """Processing event record."""
    timestamp: float
    event_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    image_hash: Optional[str] = None
    clothing_description: str = ""
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnalyticsEngine:
    """Advanced analytics engine for tracking and analysis."""
    
    def __init__(
        self,
        history_size: int = 10000,
        enable_persistence: bool = True,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize analytics engine.
        
        Args:
            history_size: Maximum number of events to keep in memory
            enable_persistence: Enable persistence to disk
            persistence_path: Path for persistence files
        """
        self.history_size = history_size
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or Path("analytics")
        
        self.events: deque = deque(maxlen=history_size)
        self.metrics: Dict[str, Any] = defaultdict(float)
        self.counters: Dict[str, int] = defaultdict(int)
        
        # Time-based aggregations
        self.hourly_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "successes": 0,
            "failures": 0,
        })
        
        # User analytics
        self.user_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "total_processing_time": 0.0,
            "last_request": None,
        })
        
        # Clothing description analytics
        self.clothing_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "avg_processing_time": 0.0,
            "success_rate": 0.0,
        })
        
        if enable_persistence:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
    
    def record_event(
        self,
        event_type: str,
        processing_time: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        image_hash: Optional[str] = None,
        clothing_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a processing event.
        
        Args:
            event_type: Type of event (e.g., "clothing_change", "validation", "enhancement")
            processing_time: Processing time in seconds
            success: Whether the operation was successful
            error: Error message if failed
            user_id: Optional user identifier
            session_id: Optional session identifier
            image_hash: Optional image hash
            clothing_description: Clothing description
            metadata: Optional additional metadata
        """
        event = ProcessingEvent(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            image_hash=image_hash,
            clothing_description=clothing_description,
            processing_time=processing_time,
            success=success,
            error=error,
            metadata=metadata or {},
        )
        
        self.events.append(event)
        
        # Update metrics
        self._update_metrics(event)
        
        # Persist if enabled
        if self.enable_persistence:
            self._persist_event(event)
    
    def _update_metrics(self, event: ProcessingEvent) -> None:
        """Update internal metrics from event."""
        # Global counters
        self.counters[f"{event.event_type}_total"] += 1
        if event.success:
            self.counters[f"{event.event_type}_success"] += 1
        else:
            self.counters[f"{event.event_type}_failure"] += 1
        
        # Time-based metrics
        self.metrics[f"{event.event_type}_total_time"] += event.processing_time
        self.metrics[f"{event.event_type}_avg_time"] = (
            self.metrics[f"{event.event_type}_total_time"] /
            self.counters[f"{event.event_type}_total"]
        )
        
        # Hourly stats
        hour_key = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d-%H")
        self.hourly_stats[hour_key]["count"] += 1
        self.hourly_stats[hour_key]["total_time"] += event.processing_time
        if event.success:
            self.hourly_stats[hour_key]["successes"] += 1
        else:
            self.hourly_stats[hour_key]["failures"] += 1
        
        # User stats
        if event.user_id:
            user_stats = self.user_stats[event.user_id]
            user_stats["total_requests"] += 1
            if event.success:
                user_stats["successful_requests"] += 1
            user_stats["total_processing_time"] += event.processing_time
            user_stats["last_request"] = event.timestamp
        
        # Clothing description stats
        if event.clothing_description:
            clothing_key = event.clothing_description.lower()[:50]  # Normalize and truncate
            clothing_stats = self.clothing_stats[clothing_key]
            clothing_stats["count"] += 1
            clothing_stats["avg_processing_time"] = (
                (clothing_stats["avg_processing_time"] * (clothing_stats["count"] - 1) +
                 event.processing_time) / clothing_stats["count"]
            )
            if clothing_stats["count"] > 0:
                success_count = sum(
                    1 for e in self.events
                    if e.clothing_description.lower()[:50] == clothing_key and e.success
                )
                clothing_stats["success_rate"] = success_count / clothing_stats["count"]
    
    def _persist_event(self, event: ProcessingEvent) -> None:
        """Persist event to disk."""
        try:
            date_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d")
            file_path = self.persistence_path / f"events_{date_str}.jsonl"
            
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist event: {e}")
    
    def get_statistics(
        self,
        event_type: Optional[str] = None,
        time_range: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics for events.
        
        Args:
            event_type: Filter by event type (None for all)
            time_range: Time range to analyze (None for all)
            
        Returns:
            Statistics dictionary
        """
        cutoff_time = time.time() - time_range.total_seconds() if time_range else 0
        
        relevant_events = [
            e for e in self.events
            if e.timestamp >= cutoff_time and
            (event_type is None or e.event_type == event_type)
        ]
        
        if not relevant_events:
            return {
                "total_events": 0,
                "success_rate": 0.0,
                "avg_processing_time": 0.0,
            }
        
        total = len(relevant_events)
        successes = sum(1 for e in relevant_events if e.success)
        total_time = sum(e.processing_time for e in relevant_events)
        
        return {
            "total_events": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_processing_time": total_time / total if total > 0 else 0.0,
            "min_processing_time": min(e.processing_time for e in relevant_events),
            "max_processing_time": max(e.processing_time for e in relevant_events),
            "p95_processing_time": self._percentile(
                [e.processing_time for e in relevant_events],
                0.95
            ),
            "p99_processing_time": self._percentile(
                [e.processing_time for e in relevant_events],
                0.99
            ),
        }
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user."""
        if user_id not in self.user_stats:
            return {}
        
        stats = self.user_stats[user_id]
        return {
            "total_requests": stats["total_requests"],
            "successful_requests": stats["successful_requests"],
            "success_rate": (
                stats["successful_requests"] / stats["total_requests"]
                if stats["total_requests"] > 0 else 0.0
            ),
            "avg_processing_time": (
                stats["total_processing_time"] / stats["total_requests"]
                if stats["total_requests"] > 0 else 0.0
            ),
            "last_request": stats["last_request"],
        }
    
    def get_trends(
        self,
        hours: int = 24,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get trends over time.
        
        Args:
            hours: Number of hours to analyze
            event_type: Filter by event type
            
        Returns:
            List of hourly statistics
        """
        cutoff_time = time.time() - (hours * 3600)
        
        trends = []
        for hour_key, stats in sorted(self.hourly_stats.items()):
            hour_timestamp = datetime.strptime(hour_key, "%Y-%m-%d-%H").timestamp()
            if hour_timestamp < cutoff_time:
                continue
            
            # Filter events if needed
            if event_type:
                hour_events = [
                    e for e in self.events
                    if datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d-%H") == hour_key
                    and e.event_type == event_type
                ]
                if not hour_events:
                    continue
                
                stats = {
                    "count": len(hour_events),
                    "total_time": sum(e.processing_time for e in hour_events),
                    "successes": sum(1 for e in hour_events if e.success),
                    "failures": sum(1 for e in hour_events if not e.success),
                }
            
            trends.append({
                "hour": hour_key,
                "timestamp": hour_timestamp,
                "count": stats["count"],
                "avg_time": (
                    stats["total_time"] / stats["count"]
                    if stats["count"] > 0 else 0.0
                ),
                "success_rate": (
                    stats["successes"] / stats["count"]
                    if stats["count"] > 0 else 0.0
                ),
            })
        
        return trends
    
    def get_top_clothing_descriptions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most requested clothing descriptions."""
        sorted_clothing = sorted(
            self.clothing_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:limit]
        
        return [
            {
                "description": desc,
                "count": stats["count"],
                "avg_processing_time": stats["avg_processing_time"],
                "success_rate": stats["success_rate"],
            }
            for desc, stats in sorted_clothing
        ]
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def export_data(self, file_path: Path) -> None:
        """Export all analytics data to JSON."""
        data = {
            "events": [asdict(e) for e in self.events],
            "metrics": dict(self.metrics),
            "counters": dict(self.counters),
            "hourly_stats": dict(self.hourly_stats),
            "user_stats": dict(self.user_stats),
            "clothing_stats": dict(self.clothing_stats),
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str, indent=2)
        
        logger.info(f"Analytics data exported to {file_path}")


