from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Callable, Coroutine
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from typing import Any, List, Dict, Optional
"""
Event Infrastructure
===================

Async event publisher with advanced event handling and routing.
"""



logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types enumeration."""
    COPYWRITING_REQUESTED = "copywriting.requested"
    COPYWRITING_COMPLETED = "copywriting.completed"
    COPYWRITING_FAILED = "copywriting.failed"
    COPYWRITING_IMPROVED = "copywriting.improved"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    PERFORMANCE_METRIC = "performance.metric"
    SYSTEM_HEALTH = "system.health"
    USER_ACTIVITY = "user.activity"
    ERROR_OCCURRED = "error.occurred"


@dataclass
class Event:
    """Event data structure."""
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> Any:
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {})
        )


class AsyncEventPublisher:
    """Async event publisher with Redis backend and local handlers."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 20,
        enable_redis: bool = True,
        enable_local_handlers: bool = True
    ):
        
    """__init__ function."""
self.redis_url = redis_url
        self.max_connections = max_connections
        self.enable_redis = enable_redis
        self.enable_local_handlers = enable_local_handlers
        
        # Redis connection
        self.pool = None
        self.client = None
        
        # Local event handlers
        self.handlers: Dict[str, List[Callable]] = {}
        self.subscriptions: Dict[str, str] = {}
        
        # Event statistics
        self.stats = {
            "events_published": 0,
            "events_handled": 0,
            "redis_events": 0,
            "local_events": 0,
            "errors": 0
        }
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
        
        logger.info("AsyncEventPublisher initialized")
    
    async def initialize(self) -> Any:
        """Initialize event publisher."""
        try:
            # Initialize Redis if enabled
            if self.enable_redis:
                await self._initialize_redis()
            
            # Start background tasks
            if self.enable_local_handlers:
                await self._start_background_tasks()
            
            self._running = True
            logger.info("Event publisher initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize event publisher: {e}")
            raise
    
    async def _initialize_redis(self) -> Any:
        """Initialize Redis connection."""
        try:
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=True
            )
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.enable_redis = False
    
    async def _start_background_tasks(self) -> Any:
        """Start background event processing tasks."""
        # Start event processor
        task = asyncio.create_task(self._process_events())
        self._background_tasks.append(task)
        
        # Start statistics collector
        task = asyncio.create_task(self._collect_statistics())
        self._background_tasks.append(task)
        
        logger.info("Background tasks started")
    
    async def publish(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Publish an event."""
        try:
            event = Event(
                id=str(uuid.uuid4()),
                type=event_type,
                data=data,
                timestamp=datetime.now(),
                source="copywriting_service"
            )
            
            # Publish to Redis if available
            if self.enable_redis and self.client:
                await self._publish_to_redis(event)
                self.stats["redis_events"] += 1
            
            # Publish locally
            if self.enable_local_handlers:
                await self._publish_locally(event)
                self.stats["local_events"] += 1
            
            self.stats["events_published"] += 1
            logger.debug(f"Published event: {event_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error publishing event {event_type}: {e}")
            self.stats["errors"] += 1
            return False
    
    async def publish_batch(self, events: List[Dict[str, Any]]) -> bool:
        """Publish multiple events."""
        try:
            success_count = 0
            
            for event_data in events:
                event_type = event_data.get("type")
                data = event_data.get("data", {})
                
                if event_type:
                    success = await self.publish(event_type, data)
                    if success:
                        success_count += 1
            
            logger.info(f"Published batch: {success_count}/{len(events)} events")
            return success_count == len(events)
            
        except Exception as e:
            logger.error(f"Error publishing batch: {e}")
            return False
    
    async def _publish_to_redis(self, event: Event):
        """Publish event to Redis."""
        try:
            # Publish to Redis pub/sub
            channel = f"events:{event.type}"
            message = json.dumps(event.to_dict())
            await self.client.publish(channel, message)
            
            # Store event in Redis for persistence
            event_key = f"event:{event.id}"
            await self.client.setex(
                event_key,
                3600,  # 1 hour TTL
                message
            )
            
        except Exception as e:
            logger.error(f"Error publishing to Redis: {e}")
            raise
    
    async def _publish_locally(self, event: Event):
        """Publish event locally."""
        try:
            # Get handlers for event type
            handlers = self.handlers.get(event.type, [])
            
            # Execute handlers asynchronously
            tasks = []
            for handler in handlers:
                if asyncio.iscoroutinefunction(handler):
                    task = asyncio.create_task(handler(event))
                else:
                    task = asyncio.create_task(self._run_sync_handler(handler, event))
                tasks.append(task)
            
            # Wait for all handlers to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.stats["events_handled"] += len(tasks)
            
        except Exception as e:
            logger.error(f"Error publishing locally: {e}")
            raise
    
    async def _run_sync_handler(self, handler: Callable, event: Event):
        """Run synchronous handler in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, handler, event)
    
    async def subscribe(self, event_type: str, handler: Callable) -> str:
        """Subscribe to an event type."""
        try:
            subscription_id = str(uuid.uuid4())
            
            # Add handler to local handlers
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            
            # Store subscription
            self.subscriptions[subscription_id] = event_type
            
            # Subscribe to Redis if available
            if self.enable_redis and self.client:
                await self._subscribe_to_redis(event_type, subscription_id)
            
            logger.info(f"Subscribed to {event_type} with ID: {subscription_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Error subscribing to {event_type}: {e}")
            raise
    
    async def _subscribe_to_redis(self, event_type: str, subscription_id: str):
        """Subscribe to Redis channel."""
        try:
            channel = f"events:{event_type}"
            pubsub = self.client.pubsub()
            await pubsub.subscribe(channel)
            
            # Start listening in background
            asyncio.create_task(self._listen_redis_channel(pubsub, subscription_id))
            
        except Exception as e:
            logger.error(f"Error subscribing to Redis: {e}")
    
    async def _listen_redis_channel(self, pubsub, subscription_id: str):
        """Listen to Redis channel for events."""
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        event = Event.from_dict(event_data)
                        
                        # Handle event locally
                        await self._publish_locally(event)
                        
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                        
        except Exception as e:
            logger.error(f"Error listening to Redis channel: {e}")
        finally:
            await pubsub.close()
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        try:
            event_type = self.subscriptions.get(subscription_id)
            if not event_type:
                return False
            
            # Remove from local handlers
            if event_type in self.handlers:
                # Note: This removes all handlers for the event type
                # In a production system, you'd want to track individual handlers
                del self.handlers[event_type]
            
            # Remove subscription
            del self.subscriptions[subscription_id]
            
            logger.info(f"Unsubscribed {subscription_id} from {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing {subscription_id}: {e}")
            return False
    
    async def get_event_stats(self) -> Dict[str, Any]:
        """Get event publishing statistics."""
        try:
            stats = self.stats.copy()
            
            # Add handler information
            stats["active_handlers"] = sum(len(handlers) for handlers in self.handlers.values())
            stats["active_subscriptions"] = len(self.subscriptions)
            stats["event_types"] = list(self.handlers.keys())
            
            # Add Redis information if available
            if self.enable_redis and self.client:
                try:
                    redis_info = await self.client.info()
                    stats["redis_info"] = {
                        "connected_clients": redis_info.get("connected_clients", 0),
                        "used_memory": redis_info.get("used_memory", 0),
                        "total_commands_processed": redis_info.get("total_commands_processed", 0)
                    }
                except Exception as e:
                    stats["redis_info"] = {"error": str(e)}
            
            stats["timestamp"] = datetime.now().isoformat()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting event stats: {e}")
            return {"error": str(e)}
    
    async def _process_events(self) -> Any:
        """Background task for processing events."""
        while self._running:
            try:
                # Process any pending events
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in event processor: {e}")
                await asyncio.sleep(1)
    
    async def _collect_statistics(self) -> Any:
        """Background task for collecting statistics."""
        while self._running:
            try:
                # Log statistics periodically
                if self.stats["events_published"] > 0:
                    logger.info(f"Event stats: {self.stats}")
                
                await asyncio.sleep(60)  # Log every minute
                
            except Exception as e:
                logger.error(f"Error collecting statistics: {e}")
                await asyncio.sleep(60)
    
    async def cleanup(self) -> Any:
        """Cleanup event publisher resources."""
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self.client:
                await self.client.close()
            
            if self.pool:
                await self.pool.disconnect()
            
            logger.info("Event publisher cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up event publisher: {e}")


# Predefined event handlers
async def log_event_handler(event: Event):
    """Default event logging handler."""
    logger.info(f"Event: {event.type} - {event.data}")

async def metrics_handler(event: Event):
    """Handler for collecting metrics from events."""
    if event.type == EventType.PERFORMANCE_METRIC.value:
        # Process performance metrics
        logger.debug(f"Processing performance metric: {event.data}")

async def error_handler(event: Event):
    """Handler for error events."""
    if event.type == EventType.ERROR_OCCURRED.value:
        logger.error(f"System error: {event.data}")

async def user_activity_handler(event: Event):
    """Handler for user activity events."""
    if event.type == EventType.USER_ACTIVITY.value:
        # Track user activity
        logger.info(f"User activity: {event.data}")

# Event factory functions
async def create_copywriting_requested_event(request_id: str, prompt: str, style: str, tone: str) -> Dict[str, Any]:
    """Create copywriting requested event."""
    return {
        "type": EventType.COPYWRITING_REQUESTED.value,
        "data": {
            "request_id": request_id,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "style": style,
            "tone": tone,
            "timestamp": datetime.now().isoformat()
        }
    }

def create_copywriting_completed_event(request_id: str, response_id: str, processing_time: float) -> Dict[str, Any]:
    """Create copywriting completed event."""
    return {
        "type": EventType.COPYWRITING_COMPLETED.value,
        "data": {
            "request_id": request_id,
            "response_id": response_id,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
    }

def create_copywriting_failed_event(request_id: str, error: str) -> Dict[str, Any]:
    """Create copywriting failed event."""
    return {
        "type": EventType.COPYWRITING_FAILED.value,
        "data": {
            "request_id": request_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    }

def create_performance_metric_event(metric_name: str, value: float, tags: Dict[str, str] = None) -> Dict[str, Any]:
    """Create performance metric event."""
    return {
        "type": EventType.PERFORMANCE_METRIC.value,
        "data": {
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat()
        }
    }

def create_error_event(error: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create error event."""
    return {
        "type": EventType.ERROR_OCCURRED.value,
        "data": {
            "error": error,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
    } 