"""
Enhanced task tracking utilities for better observability.
"""
import logging
import json
from typing import Dict, Optional, Any, List
from datetime import datetime
from uuid import UUID
from enum import Enum

logger = logging.getLogger(__name__)


class TaskEventType(str, Enum):
    """Types of task events."""
    CREATED = "created"
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RESUMED = "resumed"


class TaskTracker:
    """Track task events and provide audit trail with Redis persistence."""
    
    def __init__(self, redis_client=None):
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self.redis_client = redis_client
        self.max_memory_events = 1000  # Keep last N events in memory
    
    async def record_event(
        self,
        task_id: str,
        event_type: TaskEventType,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a task event (async to support Redis)."""
        event = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        # Store in memory (for fast access)
        if task_id not in self.events:
            self.events[task_id] = []
        self.events[task_id].append(event)
        
        # Limit memory usage
        if len(self.events[task_id]) > self.max_memory_events:
            self.events[task_id] = self.events[task_id][-self.max_memory_events:]
        
        # Persist to Redis if available
        if self.redis_client:
            try:
                redis_key = f"task:events:{task_id}"
                await self.redis_client.lpush(redis_key, json.dumps(event))
                await self.redis_client.expire(redis_key, 86400 * 7)  # Keep 7 days
            except Exception as e:
                logger.warning(f"Failed to persist event to Redis: {str(e)}")
        
        logger.info(
            "task_event",
            extra={
                "task_id": task_id,
                "event_type": event_type.value,
                "metadata": metadata,
            }
        )
    
    async def get_event_history(self, task_id: str) -> List[Dict[str, Any]]:
        """Get event history for a task (from Redis if available, else memory)."""
        events = self.events.get(task_id, [])
        
        # Try to fetch from Redis if available
        if self.redis_client:
            try:
                redis_key = f"task:events:{task_id}"
                redis_events = await self.redis_client.lrange(redis_key, 0, -1)
                if redis_events:
                    # Parse and merge with memory events
                    parsed_events = [json.loads(e) for e in redis_events]
                    # Combine and deduplicate (Redis is source of truth for older events)
                    combined = {e["timestamp"]: e for e in parsed_events}
                    for e in events:
                        combined[e["timestamp"]] = e
                    events = sorted(combined.values(), key=lambda x: x["timestamp"])
            except Exception as e:
                logger.warning(f"Failed to fetch events from Redis: {str(e)}")
        
        return events
    
    async def get_task_summary(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of task events."""
        events = await self.get_event_history(task_id)
        if not events:
            return None
        
        return {
            "task_id": task_id,
            "total_events": len(events),
            "first_event": events[0] if events else None,
            "last_event": events[-1] if events else None,
            "events": events,
        }


# Global task tracker instance
task_tracker = TaskTracker()

