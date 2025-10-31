"""
Gamma App - Event Streaming Service
Advanced event streaming and real-time data processing
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from datetime import datetime, timedelta
import redis
import aioredis
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import websockets
from websockets.server import WebSocketServerProtocol
import sseclient
import aiohttp
from prometheus_client import Counter, Histogram, Gauge
import threading
from queue import Queue, Empty
import pickle
import base64

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types"""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    CONTENT_UPDATE = "content_update"
    COLLABORATION = "collaboration"
    NOTIFICATION = "notification"
    ANALYTICS = "analytics"
    ERROR = "error"
    CUSTOM = "custom"

class StreamProtocol(Enum):
    """Streaming protocols"""
    WEBSOCKET = "websocket"
    SERVER_SENT_EVENTS = "sse"
    KAFKA = "kafka"
    REDIS_PUBSUB = "redis_pubsub"
    HTTP_STREAMING = "http_streaming"

class EventPriority(Enum):
    """Event priorities"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Event:
    """Event data structure"""
    id: str
    type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: EventPriority = EventPriority.NORMAL
    ttl: int = 3600  # Time to live in seconds
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamSubscription:
    """Stream subscription configuration"""
    id: str
    client_id: str
    event_types: List[EventType]
    filters: Dict[str, Any] = field(default_factory=dict)
    protocol: StreamProtocol = StreamProtocol.WEBSOCKET
    endpoint: Optional[str] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

@dataclass
class StreamProcessor:
    """Stream processor configuration"""
    name: str
    event_types: List[EventType]
    processor_function: Callable
    batch_size: int = 100
    batch_timeout: int = 5
    parallel_workers: int = 1
    is_active: bool = True

class EventStreamingService:
    """Advanced event streaming service"""
    
    def __init__(self):
        self.subscriptions = {}
        self.processors = {}
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.websocket_connections = {}
        self.event_queue = Queue()
        self.metrics = self._initialize_metrics()
        self._initialize_clients()
        self._start_background_tasks()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        return {
            'events_published': Counter('event_streaming_events_published_total', 'Total events published', ['type', 'source']),
            'events_consumed': Counter('event_streaming_events_consumed_total', 'Total events consumed', ['type', 'processor']),
            'active_subscriptions': Gauge('event_streaming_active_subscriptions', 'Active subscriptions', ['protocol']),
            'websocket_connections': Gauge('event_streaming_websocket_connections', 'WebSocket connections'),
            'event_processing_duration': Histogram('event_streaming_processing_duration_seconds', 'Event processing duration', ['processor']),
            'queue_size': Gauge('event_streaming_queue_size', 'Event queue size')
        }
    
    def _initialize_clients(self):
        """Initialize streaming clients"""
        try:
            # Initialize Redis
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for event streaming")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
        
        try:
            # Initialize Kafka
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            logger.info("Connected to Kafka for event streaming")
        except Exception as e:
            logger.warning(f"Could not connect to Kafka: {e}")
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        # Start event processing thread
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
        
        # Start metrics update thread
        self.metrics_thread = threading.Thread(target=self._update_metrics, daemon=True)
        self.metrics_thread.start()
        
        logger.info("Started background processing tasks")
    
    def _process_events(self):
        """Process events from queue"""
        while True:
            try:
                # Get event from queue
                event = self.event_queue.get(timeout=1)
                
                # Process event
                asyncio.create_task(self._handle_event(event))
                
                # Update queue size metric
                self.metrics['queue_size'].set(self.event_queue.qsize())
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _update_metrics(self):
        """Update metrics periodically"""
        while True:
            try:
                # Update active subscriptions
                for protocol in StreamProtocol:
                    count = len([s for s in self.subscriptions.values() if s.protocol == protocol and s.is_active])
                    self.metrics['active_subscriptions'].labels(protocol=protocol.value).set(count)
                
                # Update WebSocket connections
                self.metrics['websocket_connections'].set(len(self.websocket_connections))
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                time.sleep(10)
    
    async def publish_event(self, event: Event) -> bool:
        """Publish an event to the stream"""
        try:
            # Add to processing queue
            self.event_queue.put(event)
            
            # Update metrics
            self.metrics['events_published'].labels(
                type=event.type.value,
                source=event.source
            ).inc()
            
            logger.info(f"Published event: {event.id} ({event.type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False
    
    async def _handle_event(self, event: Event):
        """Handle a single event"""
        try:
            # Store event in Redis for persistence
            if self.redis_client:
                await self._store_event(event)
            
            # Process with registered processors
            await self._process_with_processors(event)
            
            # Distribute to subscribers
            await self._distribute_to_subscribers(event)
            
            # Publish to Kafka if configured
            if self.kafka_producer:
                await self._publish_to_kafka(event)
            
        except Exception as e:
            logger.error(f"Error handling event: {e}")
    
    async def _store_event(self, event: Event):
        """Store event in Redis"""
        try:
            event_data = {
                'id': event.id,
                'type': event.type.value,
                'source': event.source,
                'data': event.data,
                'timestamp': event.timestamp.isoformat(),
                'priority': event.priority.value,
                'ttl': event.ttl,
                'tags': event.tags,
                'metadata': event.metadata
            }
            
            # Store with TTL
            key = f"event:{event.id}"
            self.redis_client.setex(key, event.ttl, json.dumps(event_data))
            
            # Add to event index
            self.redis_client.zadd("events:index", {event.id: event.timestamp.timestamp()})
            
        except Exception as e:
            logger.error(f"Error storing event: {e}")
    
    async def _process_with_processors(self, event: Event):
        """Process event with registered processors"""
        try:
            for processor_name, processor in self.processors.items():
                if not processor.is_active:
                    continue
                
                if event.type in processor.event_types:
                    start_time = time.time()
                    
                    try:
                        if asyncio.iscoroutinefunction(processor.processor_function):
                            await processor.processor_function(event)
                        else:
                            processor.processor_function(event)
                        
                        # Update metrics
                        duration = time.time() - start_time
                        self.metrics['events_consumed'].labels(
                            type=event.type.value,
                            processor=processor_name
                        ).inc()
                        self.metrics['event_processing_duration'].labels(
                            processor=processor_name
                        ).observe(duration)
                        
                    except Exception as e:
                        logger.error(f"Error in processor {processor_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing with processors: {e}")
    
    async def _distribute_to_subscribers(self, event: Event):
        """Distribute event to subscribers"""
        try:
            for subscription_id, subscription in self.subscriptions.items():
                if not subscription.is_active:
                    continue
                
                if event.type in subscription.event_types:
                    # Check filters
                    if self._matches_filters(event, subscription.filters):
                        await self._send_to_subscriber(subscription, event)
                        
        except Exception as e:
            logger.error(f"Error distributing to subscribers: {e}")
    
    def _matches_filters(self, event: Event, filters: Dict[str, Any]) -> bool:
        """Check if event matches subscription filters"""
        try:
            for key, value in filters.items():
                if key == "source" and event.source != value:
                    return False
                elif key == "priority" and event.priority.value != value:
                    return False
                elif key == "tags" and not any(tag in event.tags for tag in value):
                    return False
                elif key.startswith("data."):
                    data_key = key[5:]  # Remove "data." prefix
                    if data_key not in event.data or event.data[data_key] != value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking filters: {e}")
            return True
    
    async def _send_to_subscriber(self, subscription: StreamSubscription, event: Event):
        """Send event to subscriber"""
        try:
            if subscription.protocol == StreamProtocol.WEBSOCKET:
                await self._send_websocket_event(subscription, event)
            elif subscription.protocol == StreamProtocol.SERVER_SENT_EVENTS:
                await self._send_sse_event(subscription, event)
            elif subscription.protocol == StreamProtocol.REDIS_PUBSUB:
                await self._send_redis_pubsub_event(subscription, event)
            
            # Update subscription activity
            subscription.last_activity = datetime.now()
            
        except Exception as e:
            logger.error(f"Error sending to subscriber: {e}")
    
    async def _send_websocket_event(self, subscription: StreamSubscription, event: Event):
        """Send event via WebSocket"""
        try:
            if subscription.client_id in self.websocket_connections:
                websocket = self.websocket_connections[subscription.client_id]
                
                event_data = {
                    'id': event.id,
                    'type': event.type.value,
                    'source': event.source,
                    'data': event.data,
                    'timestamp': event.timestamp.isoformat(),
                    'priority': event.priority.value
                }
                
                await websocket.send(json.dumps(event_data))
                
        except Exception as e:
            logger.error(f"Error sending WebSocket event: {e}")
    
    async def _send_sse_event(self, subscription: StreamSubscription, event: Event):
        """Send event via Server-Sent Events"""
        try:
            # This would be implemented with a proper SSE endpoint
            # For now, we'll just log the event
            logger.info(f"SSE event for {subscription.client_id}: {event.id}")
            
        except Exception as e:
            logger.error(f"Error sending SSE event: {e}")
    
    async def _send_redis_pubsub_event(self, subscription: StreamSubscription, event: Event):
        """Send event via Redis Pub/Sub"""
        try:
            if self.redis_client:
                channel = f"events:{subscription.client_id}"
                event_data = {
                    'id': event.id,
                    'type': event.type.value,
                    'source': event.source,
                    'data': event.data,
                    'timestamp': event.timestamp.isoformat(),
                    'priority': event.priority.value
                }
                
                self.redis_client.publish(channel, json.dumps(event_data))
                
        except Exception as e:
            logger.error(f"Error sending Redis Pub/Sub event: {e}")
    
    async def _publish_to_kafka(self, event: Event):
        """Publish event to Kafka"""
        try:
            topic = f"events-{event.type.value}"
            event_data = {
                'id': event.id,
                'type': event.type.value,
                'source': event.source,
                'data': event.data,
                'timestamp': event.timestamp.isoformat(),
                'priority': event.priority.value,
                'tags': event.tags,
                'metadata': event.metadata
            }
            
            self.kafka_producer.send(topic, value=event_data, key=event.id)
            
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {e}")
    
    def subscribe(self, subscription: StreamSubscription) -> str:
        """Subscribe to event stream"""
        try:
            subscription.id = str(uuid.uuid4())
            self.subscriptions[subscription.id] = subscription
            
            logger.info(f"Created subscription: {subscription.id} for {subscription.client_id}")
            return subscription.id
            
        except Exception as e:
            logger.error(f"Error creating subscription: {e}")
            raise
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from event stream"""
        try:
            if subscription_id in self.subscriptions:
                del self.subscriptions[subscription_id]
                logger.info(f"Removed subscription: {subscription_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing subscription: {e}")
            return False
    
    def register_processor(self, processor: StreamProcessor):
        """Register event processor"""
        try:
            self.processors[processor.name] = processor
            logger.info(f"Registered processor: {processor.name}")
        except Exception as e:
            logger.error(f"Error registering processor: {e}")
            raise
    
    def unregister_processor(self, processor_name: str) -> bool:
        """Unregister event processor"""
        try:
            if processor_name in self.processors:
                del self.processors[processor_name]
                logger.info(f"Unregistered processor: {processor_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error unregistering processor: {e}")
            return False
    
    async def add_websocket_connection(self, client_id: str, websocket: WebSocketServerProtocol):
        """Add WebSocket connection"""
        try:
            self.websocket_connections[client_id] = websocket
            logger.info(f"Added WebSocket connection: {client_id}")
        except Exception as e:
            logger.error(f"Error adding WebSocket connection: {e}")
    
    async def remove_websocket_connection(self, client_id: str):
        """Remove WebSocket connection"""
        try:
            if client_id in self.websocket_connections:
                del self.websocket_connections[client_id]
                logger.info(f"Removed WebSocket connection: {client_id}")
        except Exception as e:
            logger.error(f"Error removing WebSocket connection: {e}")
    
    async def get_events(
        self,
        event_types: Optional[List[EventType]] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get events from storage"""
        try:
            if not self.redis_client:
                return []
            
            events = []
            
            # Get event IDs from index
            if start_time and end_time:
                start_timestamp = start_time.timestamp()
                end_timestamp = end_time.timestamp()
                event_ids = self.redis_client.zrangebyscore(
                    "events:index", start_timestamp, end_timestamp, start=0, num=limit
                )
            else:
                event_ids = self.redis_client.zrevrange("events:index", 0, limit - 1)
            
            # Get event data
            for event_id in event_ids:
                event_data = self.redis_client.get(f"event:{event_id}")
                if event_data:
                    event_dict = json.loads(event_data)
                    
                    # Apply filters
                    if event_types and EventType(event_dict['type']) not in event_types:
                        continue
                    if source and event_dict['source'] != source:
                        continue
                    
                    event = Event(
                        id=event_dict['id'],
                        type=EventType(event_dict['type']),
                        source=event_dict['source'],
                        data=event_dict['data'],
                        timestamp=datetime.fromisoformat(event_dict['timestamp']),
                        priority=EventPriority(event_dict['priority']),
                        ttl=event_dict['ttl'],
                        tags=event_dict['tags'],
                        metadata=event_dict['metadata']
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    def get_streaming_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        try:
            stats = {
                'total_subscriptions': len(self.subscriptions),
                'active_subscriptions': len([s for s in self.subscriptions.values() if s.is_active]),
                'total_processors': len(self.processors),
                'active_processors': len([p for p in self.processors.values() if p.is_active]),
                'websocket_connections': len(self.websocket_connections),
                'queue_size': self.event_queue.qsize(),
                'subscriptions_by_protocol': {},
                'processors_by_type': {}
            }
            
            # Group subscriptions by protocol
            for subscription in self.subscriptions.values():
                protocol = subscription.protocol.value
                if protocol not in stats['subscriptions_by_protocol']:
                    stats['subscriptions_by_protocol'][protocol] = 0
                stats['subscriptions_by_protocol'][protocol] += 1
            
            # Group processors by event type
            for processor in self.processors.values():
                for event_type in processor.event_types:
                    event_type_str = event_type.value
                    if event_type_str not in stats['processors_by_type']:
                        stats['processors_by_type'][event_type_str] = 0
                    stats['processors_by_type'][event_type_str] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting streaming statistics: {e}")
            return {}
    
    def cleanup_old_events(self, days: int = 7):
        """Cleanup old events"""
        try:
            if not self.redis_client:
                return
            
            cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()
            
            # Remove old events from index
            removed_count = self.redis_client.zremrangebyscore("events:index", 0, cutoff_time)
            
            logger.info(f"Cleaned up {removed_count} old events")
            
        except Exception as e:
            logger.error(f"Error cleaning up old events: {e}")

# Global event streaming service instance
event_streaming_service = EventStreamingService()

async def publish_event(event: Event) -> bool:
    """Publish event using global service"""
    return await event_streaming_service.publish_event(event)

def subscribe_to_events(subscription: StreamSubscription) -> str:
    """Subscribe to events using global service"""
    return event_streaming_service.subscribe(subscription)

def unsubscribe_from_events(subscription_id: str) -> bool:
    """Unsubscribe from events using global service"""
    return event_streaming_service.unsubscribe(subscription_id)

def register_event_processor(processor: StreamProcessor):
    """Register event processor using global service"""
    event_streaming_service.register_processor(processor)

def unregister_event_processor(processor_name: str) -> bool:
    """Unregister event processor using global service"""
    return event_streaming_service.unregister_processor(processor_name)

async def add_websocket_connection(client_id: str, websocket: WebSocketServerProtocol):
    """Add WebSocket connection using global service"""
    await event_streaming_service.add_websocket_connection(client_id, websocket)

async def remove_websocket_connection(client_id: str):
    """Remove WebSocket connection using global service"""
    await event_streaming_service.remove_websocket_connection(client_id)

async def get_events(event_types: List[EventType] = None, source: str = None, start_time: datetime = None, end_time: datetime = None, limit: int = 100) -> List[Event]:
    """Get events using global service"""
    return await event_streaming_service.get_events(event_types, source, start_time, end_time, limit)

def get_streaming_statistics() -> Dict[str, Any]:
    """Get streaming statistics using global service"""
    return event_streaming_service.get_streaming_statistics()
























