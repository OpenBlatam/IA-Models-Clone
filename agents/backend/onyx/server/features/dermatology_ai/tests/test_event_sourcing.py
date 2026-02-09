"""
Tests for Event Sourcing
Tests for event store, events, and aggregates
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from core.event_sourcing.event import DomainEvent
from core.event_sourcing.event_store import EventStore
from core.event_sourcing.aggregate import AggregateRoot
from core.domain.entities import Analysis, AnalysisStatus


class TestDomainEvent:
    """Tests for DomainEvent"""
    
    def test_create_event(self):
        """Test creating a domain event"""
        event = DomainEvent(
            event_type="analysis.created",
            aggregate_id="test-123",
            data={"user_id": "user-123"},
            timestamp=datetime.utcnow()
        )
        
        assert event.event_type == "analysis.created"
        assert event.aggregate_id == "test-123"
        assert event.data["user_id"] == "user-123"
        assert event.timestamp is not None
    
    def test_event_serialization(self):
        """Test event serialization"""
        event = DomainEvent(
            event_type="analysis.created",
            aggregate_id="test-123",
            data={"user_id": "user-123"},
            timestamp=datetime.utcnow()
        )
        
        serialized = event.to_dict()
        
        assert serialized["event_type"] == "analysis.created"
        assert serialized["aggregate_id"] == "test-123"
        assert serialized["data"]["user_id"] == "user-123"


class TestEventStore:
    """Tests for EventStore"""
    
    @pytest.mark.asyncio
    async def test_save_event(self):
        """Test saving an event"""
        event_store = EventStore()
        
        event = DomainEvent(
            event_type="analysis.created",
            aggregate_id="test-123",
            data={"user_id": "user-123"}
        )
        
        result = await event_store.save(event)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_events_by_aggregate(self):
        """Test getting events by aggregate ID"""
        event_store = EventStore()
        
        event1 = DomainEvent(
            event_type="analysis.created",
            aggregate_id="test-123",
            data={"user_id": "user-123"}
        )
        
        event2 = DomainEvent(
            event_type="analysis.completed",
            aggregate_id="test-123",
            data={"status": "completed"}
        )
        
        await event_store.save(event1)
        await event_store.save(event2)
        
        events = await event_store.get_events("test-123")
        
        assert len(events) >= 2
        assert events[0].event_type == "analysis.created"
        assert events[1].event_type == "analysis.completed"
    
    @pytest.mark.asyncio
    async def test_get_events_by_type(self):
        """Test getting events by type"""
        event_store = EventStore()
        
        event1 = DomainEvent(
            event_type="analysis.created",
            aggregate_id="test-123",
            data={}
        )
        
        event2 = DomainEvent(
            event_type="analysis.created",
            aggregate_id="test-456",
            data={}
        )
        
        await event_store.save(event1)
        await event_store.save(event2)
        
        events = await event_store.get_events_by_type("analysis.created")
        
        assert len(events) >= 2
        assert all(e.event_type == "analysis.created" for e in events)


class TestAggregateRoot:
    """Tests for AggregateRoot"""
    
    def test_aggregate_creation(self):
        """Test creating an aggregate"""
        class TestAggregate(AggregateRoot):
            def __init__(self, id: str):
                super().__init__(id)
                self.name = "Test"
        
        aggregate = TestAggregate("test-123")
        
        assert aggregate.id == "test-123"
        assert aggregate.name == "Test"
        assert len(aggregate.uncommitted_events) == 0
    
    def test_aggregate_apply_event(self):
        """Test applying an event to aggregate"""
        class TestAggregate(AggregateRoot):
            def __init__(self, id: str):
                super().__init__(id)
                self.status = "created"
            
            def handle_analysis_completed(self, event):
                self.status = "completed"
        
        aggregate = TestAggregate("test-123")
        
        event = DomainEvent(
            event_type="analysis.completed",
            aggregate_id="test-123",
            data={}
        )
        
        aggregate.apply(event)
        
        assert aggregate.status == "completed"
        assert len(aggregate.uncommitted_events) == 1
    
    def test_aggregate_uncommitted_events(self):
        """Test tracking uncommitted events"""
        class TestAggregate(AggregateRoot):
            def __init__(self, id: str):
                super().__init__(id)
            
            def complete(self):
                event = DomainEvent(
                    event_type="analysis.completed",
                    aggregate_id=self.id,
                    data={}
                )
                self.apply(event)
        
        aggregate = TestAggregate("test-123")
        
        assert len(aggregate.uncommitted_events) == 0
        
        aggregate.complete()
        
        assert len(aggregate.uncommitted_events) == 1
        assert aggregate.uncommitted_events[0].event_type == "analysis.completed"
    
    def test_aggregate_mark_events_as_committed(self):
        """Test marking events as committed"""
        class TestAggregate(AggregateRoot):
            def __init__(self, id: str):
                super().__init__(id)
            
            def complete(self):
                event = DomainEvent(
                    event_type="analysis.completed",
                    aggregate_id=self.id,
                    data={}
                )
                self.apply(event)
        
        aggregate = TestAggregate("test-123")
        aggregate.complete()
        
        assert len(aggregate.uncommitted_events) == 1
        
        aggregate.mark_events_as_committed()
        
        assert len(aggregate.uncommitted_events) == 0



