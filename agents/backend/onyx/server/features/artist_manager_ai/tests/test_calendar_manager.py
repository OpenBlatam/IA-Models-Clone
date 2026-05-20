"""
Tests for Calendar Manager
"""

import pytest
from datetime import datetime, timedelta
from artist_manager_ai.core.calendar_manager import CalendarManager, CalendarEvent, EventType


class TestCalendarManager:
    """Tests para CalendarManager."""
    
    def test_create_event(self):
        """Test crear evento."""
        manager = CalendarManager("artist_001")
        event = CalendarEvent(
            id="event_001",
            title="Test Event",
            description="Test Description",
            event_type=EventType.CONCERT,
            start_time=datetime.now() + timedelta(days=1),
            end_time=datetime.now() + timedelta(days=1, hours=2)
        )
        
        created = manager.add_event(event)
        assert created.id == "event_001"
        assert len(manager.events) == 1
    
    def test_get_events_by_date(self):
        """Test obtener eventos por fecha."""
        manager = CalendarManager("artist_001")
        today = datetime.now()
        
        event = CalendarEvent(
            id="event_001",
            title="Today Event",
            description="Test",
            event_type=EventType.MEETING,
            start_time=today,
            end_time=today + timedelta(hours=1)
        )
        
        manager.add_event(event)
        events = manager.get_events_by_date(today)
        assert len(events) == 1
        assert events[0].title == "Today Event"
    
    def test_check_conflicts(self):
        """Test verificar conflictos."""
        manager = CalendarManager("artist_001")
        base_time = datetime.now() + timedelta(days=1)
        
        event1 = CalendarEvent(
            id="event_001",
            title="Event 1",
            description="Test",
            event_type=EventType.MEETING,
            start_time=base_time,
            end_time=base_time + timedelta(hours=2)
        )
        
        event2 = CalendarEvent(
            id="event_002",
            title="Event 2",
            description="Test",
            event_type=EventType.MEETING,
            start_time=base_time + timedelta(hours=1),  # Se solapa
            end_time=base_time + timedelta(hours=3)
        )
        
        manager.add_event(event1)
        manager.add_event(event2)
        
        conflicts = manager.check_conflicts(
            base_time + timedelta(hours=1.5),
            base_time + timedelta(hours=2.5)
        )
        assert len(conflicts) > 0




