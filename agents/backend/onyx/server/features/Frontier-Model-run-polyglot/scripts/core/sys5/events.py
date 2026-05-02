"""
Frontier Model Polyglot — System 5.0+ Event Bus.
Asynchronous pub/sub for decoupled telemetry and monitoring.
"""
import asyncio
import logging
from typing import Dict, List, Any, Callable, Optional
from enum import Enum

class EventType(Enum):
    TELEMETRY = "telemetry"
    ERROR = "error"
    TRAINING_STEP = "training_step"
    MODEL_EVENT = "model_event"
    SYSTEM = "system"

class Event:
    def __init__(self, type: EventType, data: Any = None):
        self.type = type
        self.data = data or {}
        self.timestamp = None

class AsyncEventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {t: [] for t in EventType}
        self.log = logging.getLogger("AsyncEventBus")

    async def subscribe(self, type: EventType, callback: Callable):
        self._subscribers[type].append(callback)

    async def emit(self, type: EventType, data: Any = None):
        event = Event(type, data)
        for callback in self._subscribers[type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.log.error(f"Error in subscriber: {e}")

event_bus = AsyncEventBus()
