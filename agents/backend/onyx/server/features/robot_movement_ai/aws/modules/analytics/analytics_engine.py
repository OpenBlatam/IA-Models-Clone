"""
Analytics Engine
================

Analytics processing engine.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsEvent:
    """Analytics event."""
    event_type: str
    properties: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AnalyticsEngine:
    """Analytics engine."""
    
    def __init__(self):
        self._events: List[AnalyticsEvent] = []
        self._aggregations: Dict[str, Dict[str, Any]] = {}
    
    def track_event(
        self,
        event_type: str,
        properties: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Track analytics event."""
        event = AnalyticsEvent(
            event_type=event_type,
            properties=properties,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id
        )
        
        self._events.append(event)
        
        # Keep only recent events
        if len(self._events) > 100000:
            self._events = self._events[-50000:]
        
        logger.debug(f"Tracked event: {event_type}")
    
    def get_event_count(
        self,
        event_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """Get event count."""
        events = self._filter_events(event_type, start_date, end_date)
        return len(events)
    
    def get_unique_users(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """Get unique user count."""
        events = self._filter_events(event_type, start_date, end_date)
        unique_users = set(e.user_id for e in events if e.user_id)
        return len(unique_users)
    
    def get_aggregation(
        self,
        event_type: str,
        property_name: str,
        aggregation_type: str = "sum",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> float:
        """Get aggregation."""
        events = self._filter_events(event_type, start_date, end_date)
        
        values = [
            e.properties.get(property_name)
            for e in events
            if property_name in e.properties
        ]
        
        if not values:
            return 0.0
        
        if aggregation_type == "sum":
            return sum(v for v in values if isinstance(v, (int, float)))
        elif aggregation_type == "avg":
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
        elif aggregation_type == "max":
            return max(v for v in values if isinstance(v, (int, float)))
        elif aggregation_type == "min":
            return min(v for v in values if isinstance(v, (int, float)))
        else:
            return 0.0
    
    def get_trend(
        self,
        event_type: str,
        property_name: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get trend data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        daily_data = defaultdict(lambda: {"count": 0, "value": 0.0})
        
        events = self._filter_events(event_type, start_date, end_date)
        
        for event in events:
            date_key = event.timestamp.date().isoformat()
            daily_data[date_key]["count"] += 1
            
            if property_name and property_name in event.properties:
                value = event.properties[property_name]
                if isinstance(value, (int, float)):
                    daily_data[date_key]["value"] += value
        
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily_data": dict(daily_data)
        }
    
    def _filter_events(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AnalyticsEvent]:
        """Filter events."""
        events = self._events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        return events
    
    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        event_types = set(e.event_type for e in self._events)
        
        return {
            "total_events": len(self._events),
            "unique_event_types": len(event_types),
            "unique_users": len(set(e.user_id for e in self._events if e.user_id)),
            "by_event_type": {
                event_type: sum(1 for e in self._events if e.event_type == event_type)
                for event_type in event_types
            }
        }















