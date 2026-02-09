"""
Analytics System for Piel Mejorador AI SAM3
===========================================

Advanced analytics and reporting.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsEvent:
    """Analytics event."""
    event_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AnalyticsEngine:
    """
    Analytics engine for tracking and reporting.
    
    Features:
    - Event tracking
    - User analytics
    - Session tracking
    - Custom reports
    - Aggregations
    """
    
    def __init__(self):
        """Initialize analytics engine."""
        self._events: List[AnalyticsEvent] = []
        self._max_events: int = 100000
        self._user_sessions: Dict[str, Dict[str, Any]] = {}
        
        self._stats = {
            "total_events": 0,
            "unique_users": 0,
            "unique_sessions": 0,
        }
    
    def track_event(self, event: AnalyticsEvent):
        """
        Track an analytics event.
        
        Args:
            event: Analytics event
        """
        self._events.append(event)
        self._stats["total_events"] += 1
        
        # Maintain max events
        if len(self._events) > self._max_events:
            self._events.pop(0)
        
        # Track user sessions
        if event.user_id:
            if event.user_id not in self._user_sessions:
                self._user_sessions[event.user_id] = {
                    "first_seen": event.timestamp,
                    "last_seen": event.timestamp,
                    "event_count": 0,
                    "sessions": set(),
                }
                self._stats["unique_users"] += 1
            
            user_data = self._user_sessions[event.user_id]
            user_data["last_seen"] = event.timestamp
            user_data["event_count"] += 1
            
            if event.session_id:
                user_data["sessions"].add(event.session_id)
                self._stats["unique_sessions"] = len(
                    set(sid for user_data in self._user_sessions.values() for sid in user_data["sessions"])
                )
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AnalyticsEvent]:
        """
        Get analytics events with filters.
        
        Args:
            event_type: Optional event type filter
            user_id: Optional user ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of events
            
        Returns:
            List of matching events
        """
        events = self._events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events[-limit:]
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User analytics data
        """
        if user_id not in self._user_sessions:
            return {}
        
        user_data = self._user_sessions[user_id]
        user_events = [e for e in self._events if e.user_id == user_id]
        
        # Event type distribution
        event_types = defaultdict(int)
        for event in user_events:
            event_types[event.event_type] += 1
        
        return {
            "user_id": user_id,
            "first_seen": user_data["first_seen"].isoformat(),
            "last_seen": user_data["last_seen"].isoformat(),
            "total_events": user_data["event_count"],
            "session_count": len(user_data["sessions"]),
            "event_types": dict(event_types),
            "active_days": (user_data["last_seen"] - user_data["first_seen"]).days + 1,
        }
    
    def get_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate analytics report.
        
        Args:
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            Analytics report
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.now()
        
        events = self.get_events(start_time=start_time, end_time=end_time, limit=100000)
        
        # Event type distribution
        event_types = defaultdict(int)
        for event in events:
            event_types[event.event_type] += 1
        
        # Daily distribution
        daily_events = defaultdict(int)
        for event in events:
            day = event.timestamp.date()
            daily_events[day.isoformat()] += 1
        
        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "summary": {
                "total_events": len(events),
                "unique_users": len(set(e.user_id for e in events if e.user_id)),
                "unique_sessions": len(set(e.session_id for e in events if e.session_id)),
            },
            "event_types": dict(event_types),
            "daily_distribution": dict(daily_events),
            "statistics": self._stats,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        return self._stats




