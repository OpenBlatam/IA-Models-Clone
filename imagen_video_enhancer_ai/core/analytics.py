"""
Analytics System
================

System for collecting and analyzing usage analytics.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Analytics event type."""
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    API_CALL = "api_call"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    ERROR = "error"
    USER_ACTION = "user_action"


@dataclass
class AnalyticsEvent:
    """Analytics event."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnalyticsCollector:
    """Analytics event collector."""
    
    def __init__(self):
        """Initialize analytics collector."""
        self.events: List[AnalyticsEvent] = []
        self.max_events = 10000
    
    def track(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track an analytics event.
        
        Args:
            event_type: Event type
            user_id: Optional user ID
            session_id: Optional session ID
            properties: Optional event properties
            metadata: Optional metadata
        """
        event = AnalyticsEvent(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            properties=properties or {},
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Limit events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        logger.debug(f"Tracked event: {event_type.value}")
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AnalyticsEvent]:
        """
        Get events with filters.
        
        Args:
            event_type: Optional event type filter
            user_id: Optional user ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events[-limit:]
    
    def get_stats(self, period: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get analytics statistics.
        
        Args:
            period: Optional time period
            
        Returns:
            Statistics dictionary
        """
        events = self.events
        
        if period:
            cutoff = datetime.now() - period
            events = [e for e in events if e.timestamp >= cutoff]
        
        # Count by event type
        by_type = defaultdict(int)
        for event in events:
            by_type[event.event_type.value] += 1
        
        # Count by user
        by_user = defaultdict(int)
        for event in events:
            if event.user_id:
                by_user[event.user_id] += 1
        
        return {
            "total_events": len(events),
            "by_type": dict(by_type),
            "unique_users": len(by_user),
            "by_user": dict(sorted(by_user.items(), key=lambda x: x[1], reverse=True)[:10])
        }


class AnalyticsReporter:
    """Analytics reporter for generating reports."""
    
    def __init__(self, collector: AnalyticsCollector):
        """
        Initialize analytics reporter.
        
        Args:
            collector: Analytics collector instance
        """
        self.collector = collector
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate daily analytics report.
        
        Args:
            date: Optional date (defaults to today)
            
        Returns:
            Daily report dictionary
        """
        if date is None:
            date = datetime.now()
        
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        events = self.collector.get_events(start_time=start, end_time=end)
        
        return {
            "date": date.date().isoformat(),
            "total_events": len(events),
            "stats": self.collector.get_stats(period=timedelta(days=1)),
            "top_events": self._get_top_events(events, limit=10)
        }
    
    def _get_top_events(self, events: List[AnalyticsEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top events by frequency."""
        by_type = defaultdict(int)
        for event in events:
            by_type[event.event_type.value] += 1
        
        return [
            {"event_type": event_type, "count": count}
            for event_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]




