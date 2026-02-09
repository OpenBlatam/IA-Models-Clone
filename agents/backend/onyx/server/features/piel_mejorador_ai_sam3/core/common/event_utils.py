"""
Event Utilities for Piel Mejorador AI SAM3
==========================================

Unified event handling and observer pattern utilities.
"""

import asyncio
import logging
from typing import Callable, Any, Dict, Optional, List, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Event:
    """Event data structure."""
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


class EventUtils:
    """Unified event handling utilities."""
    
    @staticmethod
    def create_event(
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Event:
        """
        Create event object.
        
        Args:
            event_type: Event type
            data: Event data
            source: Event source
            correlation_id: Optional correlation ID
            
        Returns:
            Event object
        """
        return Event(
            event_type=event_type,
            data=data or {},
            source=source,
            correlation_id=correlation_id
        )
    
    @staticmethod
    def matches_pattern(event_type: str, pattern: str) -> bool:
        """
        Check if event type matches pattern (supports wildcards).
        
        Args:
            event_type: Event type
            pattern: Pattern (supports *, task.*, *.completed)
            
        Returns:
            True if matches
        """
        if pattern == "*" or pattern == event_type:
            return True
        
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix + ".")
        
        if pattern.startswith("*."):
            suffix = pattern[2:]
            return event_type.endswith("." + suffix)
        
        return False
    
    @staticmethod
    def filter_events(
        events: List[Event],
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[Event]:
        """
        Filter events by criteria.
        
        Args:
            events: List of events
            event_type: Optional exact event type
            source: Optional source filter
            pattern: Optional pattern filter
            
        Returns:
            Filtered events
        """
        filtered = events
        
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        if source:
            filtered = [e for e in filtered if e.source == source]
        
        if pattern:
            filtered = [e for e in filtered if EventUtils.matches_pattern(e.event_type, pattern)]
        
        return filtered
    
    @staticmethod
    def group_by_type(events: List[Event]) -> Dict[str, List[Event]]:
        """
        Group events by type.
        
        Args:
            events: List of events
            
        Returns:
            Dictionary mapping event types to lists of events
        """
        grouped = defaultdict(list)
        for event in events:
            grouped[event.event_type].append(event)
        return dict(grouped)
    
    @staticmethod
    def group_by_source(events: List[Event]) -> Dict[str, List[Event]]:
        """
        Group events by source.
        
        Args:
            events: List of events
            
        Returns:
            Dictionary mapping sources to lists of events
        """
        grouped = defaultdict(list)
        for event in events:
            source = event.source or "unknown"
            grouped[source].append(event)
        return dict(grouped)
    
    @staticmethod
    def get_latest(events: List[Event], event_type: Optional[str] = None) -> Optional[Event]:
        """
        Get latest event (optionally filtered by type).
        
        Args:
            events: List of events
            event_type: Optional event type filter
            
        Returns:
            Latest event or None
        """
        filtered = events
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        if not filtered:
            return None
        
        return max(filtered, key=lambda e: e.timestamp)
    
    @staticmethod
    def get_count_by_type(events: List[Event]) -> Dict[str, int]:
        """
        Get count of events by type.
        
        Args:
            events: List of events
            
        Returns:
            Dictionary mapping event types to counts
        """
        counts = defaultdict(int)
        for event in events:
            counts[event.event_type] += 1
        return dict(counts)


# Convenience functions
def create_event(event_type: str, **kwargs) -> Event:
    """Create event."""
    return EventUtils.create_event(event_type, **kwargs)


def matches_pattern(event_type: str, pattern: str) -> bool:
    """Check if event matches pattern."""
    return EventUtils.matches_pattern(event_type, pattern)


def filter_events(events: List[Event], **kwargs) -> List[Event]:
    """Filter events."""
    return EventUtils.filter_events(events, **kwargs)


def group_by_type(events: List[Event]) -> Dict[str, List[Event]]:
    """Group events by type."""
    return EventUtils.group_by_type(events)




