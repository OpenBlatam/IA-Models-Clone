#!/usr/bin/env python3
"""
Event System - Ultra-Modular Architecture v3.7
Pub/sub messaging and event handling system
"""
import time
import json
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import asyncio
import weakref

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 10

@dataclass
class Event:
    """Event information"""
    name: str
    data: Any
    source: str
    timestamp: datetime
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = None
    correlation_id: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventHandler:
    """Event handler with priority and filtering"""
    
    def __init__(self, callback: Callable, priority: EventPriority = EventPriority.NORMAL,
                 event_filter: Optional[Callable] = None, async_handler: bool = False):
        self.callback = callback
        self.priority = priority
        self.event_filter = event_filter
        self.async_handler = async_handler
        self.handler_id = id(callback)
        self.created_at = datetime.now()
        self.call_count = 0
        self.last_called = None
        self.error_count = 0
        self.last_error = None

class EventSystem:
    """
    Centralized event system with pub/sub messaging, filtering, and async support
    """
    
    def __init__(self, max_event_history: int = 10000, enable_async: bool = True):
        """Initialize event system"""
        self.max_event_history = max_event_history
        self.enable_async = enable_async
        
        # Event handlers
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._wildcard_handlers: List[EventHandler] = []
        
        # Event history
        self._event_history: deque = deque(maxlen=max_event_history)
        self._event_stats: Dict[str, int] = defaultdict(int)
        
        # Event processing
        self._event_queue: deque = deque()
        self._processing_thread = None
        self._shutdown_event = threading.Event()
        self._processing_enabled = True
        
        # Threading
        self._lock = threading.RLock()
        self._async_loop = None
        
        # Performance tracking
        self._performance_metrics = {
            'events_processed': 0,
            'events_dropped': 0,
            'handlers_executed': 0,
            'errors_encountered': 0,
            'start_time': datetime.now()
        }
        
        # Initialize
        self._start_processing_thread()
        if self.enable_async:
            self._start_async_loop()
    
    def _start_processing_thread(self):
        """Start event processing thread"""
        try:
            self._processing_thread = threading.Thread(target=self._process_events, daemon=True)
            self._processing_thread.start()
            logger.info("Event processing thread started")
            
        except Exception as e:
            logger.error(f"Error starting event processing thread: {e}")
    
    def _start_async_loop(self):
        """Start async event loop"""
        try:
            self._async_loop = asyncio.new_event_loop()
            self._async_loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self._async_loop_thread.start()
            logger.info("Async event loop started")
            
        except Exception as e:
            logger.error(f"Error starting async event loop: {e}")
    
    def _run_async_loop(self):
        """Run async event loop in separate thread"""
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()
    
    def subscribe(self, event_name: str, callback: Callable, priority: EventPriority = EventPriority.NORMAL,
                  event_filter: Optional[Callable] = None, async_handler: bool = False) -> str:
        """Subscribe to events"""
        try:
            handler = EventHandler(callback, priority, event_filter, async_handler)
            
            if event_name == "*":
                # Global handler for all events
                self._wildcard_handlers.append(handler)
                self._wildcard_handlers.sort(key=lambda x: x.priority.value, reverse=True)
            elif event_name == "**":
                # Global handler for all events (including system events)
                self._global_handlers.append(handler)
                self._global_handlers.sort(key=lambda x: x.priority.value, reverse=True)
            else:
                # Specific event handler
                self._handlers[event_name].append(handler)
                self._handlers[event_name].sort(key=lambda x: x.priority.value, reverse=True)
            
            logger.debug(f"Subscribed to event: {event_name} with priority {priority.value}")
            return str(handler.handler_id)
            
        except Exception as e:
            logger.error(f"Error subscribing to event {event_name}: {e}")
            return ""
    
    def unsubscribe(self, event_name: str, callback: Callable) -> bool:
        """Unsubscribe from events"""
        try:
            if event_name == "*":
                # Remove from wildcard handlers
                self._wildcard_handlers = [h for h in self._wildcard_handlers if h.callback != callback]
            elif event_name == "**":
                # Remove from global handlers
                self._global_handlers = [h for h in self._global_handlers if h.callback != callback]
            else:
                # Remove from specific event handlers
                if event_name in self._handlers:
                    self._handlers[event_name] = [h for h in self._handlers[event_name] if h.callback != callback]
            
            logger.debug(f"Unsubscribed from event: {event_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from event {event_name}: {e}")
            return False
    
    def publish(self, event_name: str, data: Any = None, source: str = "system",
                priority: EventPriority = EventPriority.NORMAL, correlation_id: str = None,
                metadata: Dict[str, Any] = None) -> str:
        """Publish an event"""
        try:
            # Create event
            event = Event(
                name=event_name,
                data=data,
                source=source,
                timestamp=datetime.now(),
                priority=priority,
                event_id=f"{int(time.time() * 1000000)}",
                correlation_id=correlation_id,
                metadata=metadata or {}
            )
            
            # Add to event queue
            self._event_queue.append(event)
            
            # Update statistics
            self._event_stats[event_name] += 1
            
            logger.debug(f"Event published: {event_name} from {source}")
            return event.event_id
            
        except Exception as e:
            logger.error(f"Error publishing event {event_name}: {e}")
            return ""
    
    def publish_async(self, event_name: str, data: Any = None, source: str = "system",
                      priority: EventPriority = EventPriority.NORMAL, correlation_id: str = None,
                      metadata: Dict[str, Any] = None) -> str:
        """Publish an event asynchronously"""
        try:
            if not self.enable_async or not self._async_loop:
                return self.publish(event_name, data, source, priority, correlation_id, metadata)
            
            # Create event
            event = Event(
                name=event_name,
                data=data,
                source=source,
                timestamp=datetime.now(),
                priority=priority,
                event_id=f"{int(time.time() * 1000000)}",
                correlation_id=correlation_id,
                metadata=metadata or {}
            )
            
            # Schedule async processing
            asyncio.run_coroutine_threadsafe(
                self._process_async_event(event), self._async_loop
            )
            
            # Update statistics
            self._event_stats[event_name] += 1
            
            logger.debug(f"Async event published: {event_name} from {source}")
            return event.event_id
            
        except Exception as e:
            logger.error(f"Error publishing async event {event_name}: {e}")
            return ""
    
    async def _process_async_event(self, event: Event):
        """Process event asynchronously"""
        try:
            # Find handlers
            handlers = self._find_handlers(event.name)
            
            # Execute handlers
            for handler in handlers:
                if handler.async_handler:
                    try:
                        if asyncio.iscoroutinefunction(handler.callback):
                            await handler.callback(event)
                        else:
                            # Run sync callback in thread pool
                            await asyncio.get_event_loop().run_in_executor(None, handler.callback, event)
                        
                        handler.call_count += 1
                        handler.last_called = datetime.now()
                        
                    except Exception as e:
                        handler.error_count += 1
                        handler.last_error = str(e)
                        logger.error(f"Error in async event handler: {e}")
            
        except Exception as e:
            logger.error(f"Error processing async event: {e}")
    
    def _process_events(self):
        """Process events in background thread"""
        while not self._shutdown_event.is_set():
            try:
                if self._event_queue and self._processing_enabled:
                    event = self._event_queue.popleft()
                    self._process_single_event(event)
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    
            except Exception as e:
                logger.error(f"Error in event processing thread: {e}")
                time.sleep(0.1)
    
    def _process_single_event(self, event: Event):
        """Process a single event"""
        try:
            # Find handlers
            handlers = self._find_handlers(event.name)
            
            # Execute handlers
            for handler in handlers:
                try:
                    # Check event filter
                    if handler.event_filter and not handler.event_filter(event):
                        continue
                    
                    # Execute handler
                    if handler.async_handler:
                        # Skip async handlers in sync processing
                        continue
                    
                    handler.callback(event)
                    handler.call_count += 1
                    handler.last_called = datetime.now()
                    
                except Exception as e:
                    handler.error_count += 1
                    handler.last_error = str(e)
                    logger.error(f"Error in event handler: {e}")
            
            # Store in history
            self._event_history.append(event)
            
            # Update performance metrics
            self._performance_metrics['events_processed'] += 1
            self._performance_metrics['handlers_executed'] += len(handlers)
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            self._performance_metrics['errors_encountered'] += 1
    
    def _find_handlers(self, event_name: str) -> List[EventHandler]:
        """Find handlers for an event"""
        handlers = []
        
        # Add specific event handlers
        if event_name in self._handlers:
            handlers.extend(self._handlers[event_name])
        
        # Add wildcard handlers
        handlers.extend(self._wildcard_handlers)
        
        # Add global handlers
        handlers.extend(self._global_handlers)
        
        # Sort by priority
        handlers.sort(key=lambda x: x.priority.value, reverse=True)
        
        return handlers
    
    def wait_for_event(self, event_name: str, timeout: float = None, 
                       condition: Optional[Callable] = None) -> Optional[Event]:
        """Wait for a specific event"""
        try:
            start_time = time.time()
            event_received = threading.Event()
            received_event = None
            
            def event_handler(event: Event):
                if condition is None or condition(event):
                    nonlocal received_event
                    received_event = event
                    event_received.set()
            
            # Subscribe to event
            handler_id = self.subscribe(event_name, event_handler)
            
            try:
                # Wait for event
                if event_received.wait(timeout=timeout):
                    return received_event
                else:
                    return None
                    
            finally:
                # Unsubscribe
                self.unsubscribe(event_name, event_handler)
                
        except Exception as e:
            logger.error(f"Error waiting for event {event_name}: {e}")
            return None
    
    def get_event_history(self, event_name: str = None, limit: int = 100) -> List[Event]:
        """Get event history"""
        try:
            if event_name is None:
                # Return all events
                return list(self._event_history)[-limit:]
            else:
                # Return filtered events
                filtered_events = [e for e in self._event_history if e.name == event_name]
                return filtered_events[-limit:]
                
        except Exception as e:
            logger.error(f"Error getting event history: {e}")
            return []
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event statistics"""
        try:
            return {
                'total_events': sum(self._event_stats.values()),
                'events_by_name': dict(self._event_stats),
                'handlers': {
                    'specific': sum(len(handlers) for handlers in self._handlers.values()),
                    'wildcard': len(self._wildcard_handlers),
                    'global': len(self._global_handlers)
                },
                'performance': self._performance_metrics.copy(),
                'queue_size': len(self._event_queue),
                'history_size': len(self._event_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting event statistics: {e}")
            return {}
    
    def clear_event_history(self):
        """Clear event history"""
        try:
            self._event_history.clear()
            logger.info("Event history cleared")
            
        except Exception as e:
            logger.error(f"Error clearing event history: {e}")
    
    def pause_processing(self):
        """Pause event processing"""
        try:
            self._processing_enabled = False
            logger.info("Event processing paused")
            
        except Exception as e:
            logger.error(f"Error pausing event processing: {e}")
    
    def resume_processing(self):
        """Resume event processing"""
        try:
            self._processing_enabled = True
            logger.info("Event processing resumed")
            
        except Exception as e:
            logger.error(f"Error resuming event processing: {e}")
    
    def get_handler_info(self) -> Dict[str, Any]:
        """Get information about event handlers"""
        try:
            handler_info = {}
            
            # Specific event handlers
            for event_name, handlers in self._handlers.items():
                handler_info[event_name] = [
                    {
                        'handler_id': h.handler_id,
                        'priority': h.priority.value,
                        'async': h.async_handler,
                        'call_count': h.call_count,
                        'error_count': h.error_count,
                        'last_called': h.last_called.isoformat() if h.last_called else None,
                        'last_error': h.last_error
                    }
                    for h in handlers
                ]
            
            # Wildcard handlers
            handler_info['*'] = [
                {
                    'handler_id': h.handler_id,
                    'priority': h.priority.value,
                    'async': h.async_handler,
                    'call_count': h.call_count,
                    'error_count': h.error_count,
                    'last_called': h.last_called.isoformat() if h.last_called else None,
                    'last_error': h.last_error
                }
                for h in self._wildcard_handlers
            ]
            
            # Global handlers
            handler_info['**'] = [
                {
                    'handler_id': h.handler_id,
                    'priority': h.priority.value,
                    'async': h.async_handler,
                    'call_count': h.call_count,
                    'error_count': h.error_count,
                    'last_called': h.last_called.isoformat() if h.last_called else None,
                    'last_error': h.last_error
                }
                for h in self._global_handlers
            ]
            
            return handler_info
            
        except Exception as e:
            logger.error(f"Error getting handler info: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown event system"""
        try:
            logger.info("Shutting down event system...")
            
            # Stop processing
            self._processing_enabled = False
            self._shutdown_event.set()
            
            # Wait for processing thread
            if self._processing_thread:
                self._processing_thread.join(timeout=5.0)
            
            # Stop async loop
            if self._async_loop:
                self._async_loop.call_soon_threadsafe(self._async_loop.stop)
                if self._async_loop_thread:
                    self._async_loop_thread.join(timeout=5.0)
            
            # Clear handlers
            self._handlers.clear()
            self._global_handlers.clear()
            self._wildcard_handlers.clear()
            
            # Clear event queue and history
            self._event_queue.clear()
            self._event_history.clear()
            
            logger.info("Event system shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()

# Utility functions
def create_event_filter(field_name: str, expected_value: Any) -> Callable:
    """Create an event filter function"""
    def filter_func(event: Event) -> bool:
        if hasattr(event, field_name):
            return getattr(event, field_name) == expected_value
        elif field_name in event.metadata:
            return event.metadata[field_name] == expected_value
        return False
    
    return filter_func

def create_priority_filter(min_priority: EventPriority) -> Callable:
    """Create a priority-based event filter"""
    def filter_func(event: Event) -> bool:
        return event.priority.value >= min_priority.value
    
    return filter_func

def create_source_filter(allowed_sources: List[str]) -> Callable:
    """Create a source-based event filter"""
    def filter_func(event: Event) -> bool:
        return event.source in allowed_sources
    
    return filter_func
