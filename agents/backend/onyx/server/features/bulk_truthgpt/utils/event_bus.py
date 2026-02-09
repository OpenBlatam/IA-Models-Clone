"""
Event Bus System
================

Advanced event bus for pub/sub pattern and event-driven architecture.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
from datetime import datetime
import weakref

logger = logging.getLogger(__name__)

class EventBus:
    """Advanced event bus with async support."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self.event_stats: Dict[str, int] = defaultdict(int)
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type."""
        if handler not in self.subscribers[event_type]:
            self.subscribers[event_type].append(handler)
            logger.debug(f"Subscribed handler to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from event type."""
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type}")
    
    async def publish(self, event_type: str, payload: Any = None):
        """Publish event."""
        event = {
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Update stats
        self.event_stats[event_type] += 1
        
        # Notify subscribers
        handlers = self.subscribers.get(event_type, [])
        if handlers:
            tasks = []
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(payload))
                    else:
                        handler(payload)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history."""
        if event_type:
            history = [e for e in self.event_history if e["type"] == event_type]
        else:
            history = self.event_history
        
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "total_events": len(self.event_history),
            "event_types": dict(self.event_stats),
            "subscribers": {
                event_type: len(handlers)
                for event_type, handlers in self.subscribers.items()
            }
        }

# Global instance
event_bus = EventBus()
































