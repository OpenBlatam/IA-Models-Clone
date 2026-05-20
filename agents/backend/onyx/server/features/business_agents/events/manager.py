"""
Event Manager
=============

Event management and processing system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
import uuid

from .types import (
    Event, EventType, EventHandler, EventSubscription, 
    EventStore, EventMetrics, EventFilter
)

logger = logging.getLogger(__name__)

class EventBus:
    """Event bus for publishing and subscribing to events."""
    
    def __init__(self):
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_store = EventStore()
        self.metrics = EventMetrics()
        self._lock = asyncio.Lock()
    
    async def publish(self, event: Event) -> bool:
        """Publish an event to the bus."""
        try:
            start_time = datetime.now()
            
            # Store the event
            await self.event_store.append(event)
            
            # Find matching subscriptions
            matching_subscriptions = []
            async with self._lock:
                for subscription in self.subscriptions.values():
                    if subscription.matches(event):
                        matching_subscriptions.append(subscription)
            
            # Process event with handlers
            success = await self._process_event(event, matching_subscriptions)
            
            # Record metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            if success:
                self.metrics.record_event_processed(event, processing_time)
            else:
                self.metrics.record_event_failed(event)
            
            logger.debug(f"Published event {event.id} of type {event.type.value}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.id}: {str(e)}")
            self.metrics.record_event_failed(event)
            return False
    
    async def _process_event(self, event: Event, subscriptions: List[EventSubscription]) -> bool:
        """Process an event with its subscriptions."""
        try:
            # Sort subscriptions by priority
            subscriptions.sort(key=lambda s: s.handler.get_priority())
            
            # Execute handlers
            for subscription in subscriptions:
                if not subscription.handler.is_enabled():
                    continue
                
                try:
                    handler_start_time = datetime.now()
                    success = await subscription.handler.handle(event)
                    handler_time = (datetime.now() - handler_start_time).total_seconds()
                    
                    if success:
                        self.metrics.record_handler_executed(subscription.handler.name, handler_time)
                    else:
                        self.metrics.record_handler_failed(subscription.handler.name)
                        logger.warning(f"Handler {subscription.handler.name} failed for event {event.id}")
                
                except Exception as e:
                    self.metrics.record_handler_failed(subscription.handler.name)
                    logger.error(f"Handler {subscription.handler.name} error for event {event.id}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process event {event.id}: {str(e)}")
            return False
    
    async def subscribe(
        self, 
        event_types: List[EventType], 
        handler: EventHandler,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> str:
        """Subscribe to events."""
        try:
            subscription_id = str(uuid.uuid4())
            
            subscription = EventSubscription(
                id=subscription_id,
                event_types=event_types,
                handler=handler,
                filter_conditions=filter_conditions
            )
            
            async with self._lock:
                self.subscriptions[subscription_id] = subscription
            
            logger.info(f"Created subscription {subscription_id} for event types: {[et.value for et in event_types]}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to create subscription: {str(e)}")
            raise
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        try:
            async with self._lock:
                if subscription_id in self.subscriptions:
                    del self.subscriptions[subscription_id]
                    logger.info(f"Removed subscription {subscription_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove subscription {subscription_id}: {str(e)}")
            return False
    
    async def get_events(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get events from the store."""
        return await self.event_store.get_events(event_types, start_time, end_time, limit)
    
    async def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        return await self.event_store.get_event_by_id(event_id)
    
    async def get_events_by_correlation_id(self, correlation_id: str) -> List[Event]:
        """Get events by correlation ID."""
        return await self.event_store.get_events_by_correlation_id(correlation_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event processing metrics."""
        return {
            "events_processed": self.metrics.events_processed,
            "events_failed": self.metrics.events_failed,
            "handlers_executed": self.metrics.handlers_executed,
            "handlers_failed": self.metrics.handlers_failed,
            "average_processing_time": self.metrics.get_average_processing_time(),
            "success_rate": self.metrics.get_success_rate(),
            "event_type_counts": {et.value: count for et, count in self.metrics.event_type_counts.items()},
            "handler_performance": {
                name: self.metrics.get_handler_average_time(name)
                for name in self.metrics.handler_performance.keys()
            }
        }

class EventManager:
    """Main event manager for the Business Agents System."""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.handlers: Dict[str, EventHandler] = {}
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the event manager."""
        try:
            # Register default handlers
            await self._register_default_handlers()
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._running = True
            logger.info("Event manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize event manager: {str(e)}")
            raise
    
    async def shutdown(self):
        """Shutdown the event manager."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event manager shutdown complete")
    
    async def publish_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str = "business_agents_system",
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Publish an event."""
        try:
            event = Event.create(
                event_type=event_type,
                data=data,
                source=source,
                correlation_id=correlation_id,
                causation_id=causation_id,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata
            )
            
            success = await self.event_bus.publish(event)
            if success:
                logger.info(f"Published event {event.id} of type {event_type.value}")
                return event.id
            else:
                raise Exception("Failed to publish event")
                
        except Exception as e:
            logger.error(f"Failed to publish event: {str(e)}")
            raise
    
    async def subscribe_to_events(
        self,
        event_types: List[EventType],
        handler: EventHandler,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> str:
        """Subscribe to events."""
        return await self.event_bus.subscribe(event_types, handler, filter_conditions)
    
    async def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        return await self.event_bus.unsubscribe(subscription_id)
    
    def register_handler(self, handler: EventHandler):
        """Register an event handler."""
        self.handlers[handler.name] = handler
        logger.info(f"Registered event handler: {handler.name}")
    
    def unregister_handler(self, handler_name: str):
        """Unregister an event handler."""
        if handler_name in self.handlers:
            del self.handlers[handler_name]
            logger.info(f"Unregistered event handler: {handler_name}")
    
    async def get_events(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get events."""
        return await self.event_bus.get_events(event_types, start_time, end_time, limit)
    
    async def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        return await self.event_bus.get_event_by_id(event_id)
    
    async def get_events_by_correlation_id(self, correlation_id: str) -> List[Event]:
        """Get events by correlation ID."""
        return await self.event_bus.get_events_by_correlation_id(correlation_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event processing metrics."""
        return self.event_bus.get_metrics()
    
    async def _register_default_handlers(self):
        """Register default event handlers."""
        try:
            # Import and register default handlers
            from .handlers import (
                AgentEventHandler, WorkflowEventHandler,
                DocumentEventHandler, SystemEventHandler
            )
            
            # Register handlers
            self.register_handler(AgentEventHandler())
            self.register_handler(WorkflowEventHandler())
            self.register_handler(DocumentEventHandler())
            self.register_handler(SystemEventHandler())
            
            # Subscribe handlers to events
            await self._subscribe_handlers()
            
        except Exception as e:
            logger.error(f"Failed to register default handlers: {str(e)}")
    
    async def _subscribe_handlers(self):
        """Subscribe handlers to their respective events."""
        try:
            # Agent events
            agent_handler = self.handlers.get("agent_event_handler")
            if agent_handler:
                await self.subscribe_to_events([
                    EventType.AGENT_CREATED,
                    EventType.AGENT_UPDATED,
                    EventType.AGENT_DELETED,
                    EventType.AGENT_EXECUTION_STARTED,
                    EventType.AGENT_EXECUTION_COMPLETED,
                    EventType.AGENT_EXECUTION_FAILED
                ], agent_handler)
            
            # Workflow events
            workflow_handler = self.handlers.get("workflow_event_handler")
            if workflow_handler:
                await self.subscribe_to_events([
                    EventType.WORKFLOW_CREATED,
                    EventType.WORKFLOW_UPDATED,
                    EventType.WORKFLOW_DELETED,
                    EventType.WORKFLOW_EXECUTION_STARTED,
                    EventType.WORKFLOW_EXECUTION_COMPLETED,
                    EventType.WORKFLOW_EXECUTION_FAILED,
                    EventType.WORKFLOW_STEP_COMPLETED
                ], workflow_handler)
            
            # Document events
            document_handler = self.handlers.get("document_event_handler")
            if document_handler:
                await self.subscribe_to_events([
                    EventType.DOCUMENT_GENERATION_STARTED,
                    EventType.DOCUMENT_GENERATION_COMPLETED,
                    EventType.DOCUMENT_GENERATION_FAILED,
                    EventType.DOCUMENT_DOWNLOADED
                ], document_handler)
            
            # System events
            system_handler = self.handlers.get("system_event_handler")
            if system_handler:
                await self.subscribe_to_events([
                    EventType.SYSTEM_STARTUP,
                    EventType.SYSTEM_SHUTDOWN,
                    EventType.SYSTEM_ERROR,
                    EventType.SYSTEM_ALERT,
                    EventType.METRICS_UPDATED
                ], system_handler)
            
        except Exception as e:
            logger.error(f"Failed to subscribe handlers: {str(e)}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                # Clean up old events (keep last 1000 events)
                events = await self.get_events(limit=1000)
                if len(events) > 1000:
                    # Remove oldest events
                    events_to_remove = events[1000:]
                    for event in events_to_remove:
                        # This would need to be implemented in EventStore
                        pass
                
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

# Global event manager instance
event_manager = EventManager()
