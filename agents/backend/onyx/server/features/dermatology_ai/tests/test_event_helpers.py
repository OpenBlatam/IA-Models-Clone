"""
Event Testing Helpers
Specialized helpers for event-driven testing
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
import asyncio
from datetime import datetime
from collections import deque


class EventTestHelpers:
    """Helpers for event testing"""
    
    @staticmethod
    def create_mock_event_publisher(
        events_published: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock event publisher with event tracking"""
        events = events_published or []
        publisher = Mock()
        
        async def publish_side_effect(event_type: str, event_data: Dict[str, Any]):
            events.append({
                "type": event_type,
                "data": event_data,
                "timestamp": datetime.utcnow().isoformat()
            })
            return True
        
        publisher.publish = AsyncMock(side_effect=publish_side_effect)
        publisher.publish_batch = AsyncMock(return_value=True)
        publisher.events = events
        return publisher
    
    @staticmethod
    def assert_event_published(
        publisher: Mock,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None
    ):
        """Assert event was published"""
        assert publisher.publish.called, f"Event {event_type} was not published"
        
        if hasattr(publisher, "events"):
            matching_events = [
                e for e in publisher.events
                if e["type"] == event_type
            ]
            assert len(matching_events) > 0, f"No events of type {event_type} found"
            
            if event_data:
                # Check if event data matches
                found = any(
                    all(event["data"].get(k) == v for k, v in event_data.items())
                    for event in matching_events
                )
                assert found, f"Event data does not match expected: {event_data}"
    
    @staticmethod
    def assert_event_not_published(publisher: Mock, event_type: str):
        """Assert event was not published"""
        if hasattr(publisher, "events"):
            matching_events = [
                e for e in publisher.events
                if e["type"] == event_type
            ]
            assert len(matching_events) == 0, f"Event {event_type} was published but should not be"


class EventStoreHelpers:
    """Helpers for event store testing"""
    
    @staticmethod
    def create_mock_event_store(
        events: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock event store"""
        event_list = events or []
        store = Mock()
        
        async def append_side_effect(stream_id: str, events: List[Dict[str, Any]]):
            for event in events:
                event_list.append({
                    "stream_id": stream_id,
                    "event": event,
                    "timestamp": datetime.utcnow().isoformat()
                })
            return len(event_list)
        
        async def get_side_effect(stream_id: str, from_version: int = 0):
            return [
                e for e in event_list
                if e["stream_id"] == stream_id and event_list.index(e) >= from_version
            ]
        
        store.append = AsyncMock(side_effect=append_side_effect)
        store.get = AsyncMock(side_effect=get_side_effect)
        store.events = event_list
        return store
    
    @staticmethod
    def assert_event_stored(
        store: Mock,
        stream_id: str,
        event_type: Optional[str] = None
    ):
        """Assert event was stored"""
        assert store.append.called, f"Event was not stored for stream {stream_id}"
        
        if hasattr(store, "events"):
            matching_events = [
                e for e in store.events
                if e["stream_id"] == stream_id
            ]
            assert len(matching_events) > 0, f"No events found for stream {stream_id}"
            
            if event_type:
                found = any(
                    e["event"].get("type") == event_type
                    for e in matching_events
                )
                assert found, f"Event type {event_type} not found in stream {stream_id}"


class EventHandlerHelpers:
    """Helpers for event handler testing"""
    
    @staticmethod
    def create_mock_event_handler(
        handler_name: str = "test_handler",
        processed_events: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock event handler"""
        events = processed_events or []
        handler = Mock()
        handler.name = handler_name
        
        async def handle_side_effect(event: Dict[str, Any]):
            events.append({
                "event": event,
                "handler": handler_name,
                "timestamp": datetime.utcnow().isoformat()
            })
            return True
        
        handler.handle = AsyncMock(side_effect=handle_side_effect)
        handler.events = events
        return handler
    
    @staticmethod
    def assert_event_handled(
        handler: Mock,
        event_type: Optional[str] = None
    ):
        """Assert event was handled"""
        assert handler.handle.called, "Event was not handled"
        
        if hasattr(handler, "events"):
            assert len(handler.events) > 0, "No events were processed"
            
            if event_type:
                found = any(
                    e["event"].get("type") == event_type
                    for e in handler.events
                )
                assert found, f"Event type {event_type} was not handled"


class EventSourcingHelpers:
    """Helpers for event sourcing testing"""
    
    @staticmethod
    def create_aggregate_from_events(
        events: List[Dict[str, Any]],
        aggregate_type: type
    ) -> Any:
        """Create aggregate from event list"""
        aggregate = aggregate_type()
        for event in events:
            # Apply event to aggregate
            if hasattr(aggregate, "apply_event"):
                aggregate.apply_event(event)
        return aggregate
    
    @staticmethod
    def assert_aggregate_state(
        aggregate: Any,
        expected_state: Dict[str, Any]
    ):
        """Assert aggregate has expected state"""
        for key, expected_value in expected_state.items():
            actual_value = getattr(aggregate, key, None)
            assert actual_value == expected_value, \
                f"Aggregate {key} is {actual_value}, expected {expected_value}"


# Convenience exports
create_mock_event_publisher = EventTestHelpers.create_mock_event_publisher
assert_event_published = EventTestHelpers.assert_event_published
assert_event_not_published = EventTestHelpers.assert_event_not_published

create_mock_event_store = EventStoreHelpers.create_mock_event_store
assert_event_stored = EventStoreHelpers.assert_event_stored

create_mock_event_handler = EventHandlerHelpers.create_mock_event_handler
assert_event_handled = EventHandlerHelpers.assert_event_handled

create_aggregate_from_events = EventSourcingHelpers.create_aggregate_from_events
assert_aggregate_state = EventSourcingHelpers.assert_aggregate_state



