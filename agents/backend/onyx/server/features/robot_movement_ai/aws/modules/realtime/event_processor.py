"""
Event Processor
===============

Real-time event processing.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class EventProcessor:
    """Real-time event processor."""
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "processed": 0,
            "failed": 0,
            "last_processed": None
        })
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler."""
        self._handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}")
    
    async def emit(self, event_type: str, data: Any):
        """Emit event."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now()
        }
        
        await self._event_queue.put(event)
        logger.debug(f"Emitted event: {event_type}")
    
    async def process_events(self):
        """Process events from queue."""
        self._processing = True
        
        while self._processing:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle single event."""
        event_type = event["type"]
        handlers = self._handlers.get(event_type, [])
        
        if not handlers:
            logger.warning(f"No handlers for event type: {event_type}")
            return
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event["data"])
                else:
                    handler(event["data"])
                
                self._stats[event_type]["processed"] += 1
                self._stats[event_type]["last_processed"] = datetime.now()
            
            except Exception as e:
                logger.error(f"Handler failed for {event_type}: {e}")
                self._stats[event_type]["failed"] += 1
    
    def start_processing(self):
        """Start event processing."""
        if not self._processing:
            asyncio.create_task(self.process_events())
            logger.info("Event processing started")
    
    def stop_processing(self):
        """Stop event processing."""
        self._processing = False
        logger.info("Event processing stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event processing statistics."""
        return {
            "queue_size": self._event_queue.qsize(),
            "processing": self._processing,
            "handlers": {
                event_type: len(handlers)
                for event_type, handlers in self._handlers.items()
            },
            "event_stats": dict(self._stats)
        }















