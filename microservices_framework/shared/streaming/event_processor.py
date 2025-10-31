"""
Advanced Real-time Streaming and Event Processing
Features: Event streaming, real-time analytics, stream processing, event sourcing, CQRS
"""

import asyncio
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics
from abc import ABC, abstractmethod

# Streaming imports
try:
    import kafka
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import redis
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import asyncio_mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types"""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    BUSINESS_EVENT = "business_event"
    AUDIT_EVENT = "audit_event"
    METRIC_EVENT = "metric_event"
    ERROR_EVENT = "error_event"

class StreamType(Enum):
    """Stream types"""
    KAFKA = "kafka"
    REDIS_STREAMS = "redis_streams"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    HTTP_STREAM = "http_stream"

class ProcessingMode(Enum):
    """Processing modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"
    STREAMING = "streaming"

@dataclass
class Event:
    """Event data structure"""
    event_id: str
    event_type: EventType
    stream_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    version: int = 1
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

@dataclass
class StreamConfig:
    """Stream configuration"""
    stream_id: str
    stream_type: StreamType
    topic: str
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    group_id: str = "default_group"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    processing_mode: ProcessingMode
    batch_size: int = 100
    batch_timeout: float = 1.0  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0
    parallel_workers: int = 4
    enable_checkpointing: bool = True
    checkpoint_interval: float = 10.0

class EventStore:
    """
    Event store for event sourcing
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.events: Dict[str, List[Event]] = defaultdict(list)
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
    
    async def append_event(self, event: Event) -> bool:
        """Append event to store"""
        try:
            # Store in memory
            self.events[event.stream_id].append(event)
            
            # Store in Redis if available
            if self.redis:
                event_data = {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "stream_id": event.stream_id,
                    "data": event.data,
                    "metadata": event.metadata,
                    "timestamp": event.timestamp,
                    "version": event.version,
                    "correlation_id": event.correlation_id,
                    "causation_id": event.causation_id
                }
                
                await self.redis.lpush(
                    f"events:{event.stream_id}",
                    json.dumps(event_data)
                )
                
                # Set TTL for events (7 days)
                await self.redis.expire(f"events:{event.stream_id}", 604800)
            
            # Trigger event handlers
            await self._trigger_handlers(event)
            
            logger.debug(f"Event {event.event_id} appended to stream {event.stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append event: {e}")
            return False
    
    async def get_events(self, stream_id: str, from_version: int = 0) -> List[Event]:
        """Get events from stream"""
        try:
            if self.redis:
                # Get from Redis
                event_data_list = await self.redis.lrange(
                    f"events:{stream_id}",
                    from_version,
                    -1
                )
                
                events = []
                for event_data in event_data_list:
                    data = json.loads(event_data)
                    event = Event(
                        event_id=data["event_id"],
                        event_type=EventType(data["event_type"]),
                        stream_id=data["stream_id"],
                        data=data["data"],
                        metadata=data["metadata"],
                        timestamp=data["timestamp"],
                        version=data["version"],
                        correlation_id=data.get("correlation_id"),
                        causation_id=data.get("causation_id")
                    )
                    events.append(event)
                
                return events
            else:
                # Get from memory
                return self.events[stream_id][from_version:]
                
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []
    
    async def create_snapshot(self, stream_id: str, state: Dict[str, Any], version: int):
        """Create snapshot of stream state"""
        try:
            snapshot = {
                "stream_id": stream_id,
                "state": state,
                "version": version,
                "timestamp": time.time()
            }
            
            self.snapshots[stream_id] = snapshot
            
            if self.redis:
                await self.redis.set(
                    f"snapshot:{stream_id}",
                    json.dumps(snapshot)
                )
            
            logger.info(f"Snapshot created for stream {stream_id} at version {version}")
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
    
    async def get_snapshot(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get latest snapshot"""
        try:
            if self.redis:
                snapshot_data = await self.redis.get(f"snapshot:{stream_id}")
                if snapshot_data:
                    return json.loads(snapshot_data)
            
            return self.snapshots.get(stream_id)
            
        except Exception as e:
            logger.error(f"Failed to get snapshot: {e}")
            return None
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type {event_type.value}")
    
    async def _trigger_handlers(self, event: Event):
        """Trigger event handlers"""
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Event handler failed: {e}")

class StreamProcessor(ABC):
    """Abstract stream processor"""
    
    @abstractmethod
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process a single event"""
        pass
    
    @abstractmethod
    async def process_batch(self, events: List[Event]) -> List[Event]:
        """Process a batch of events"""
        pass

class RealTimeProcessor(StreamProcessor):
    """
    Real-time event processor
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.processed_count = 0
        self.error_count = 0
        self.processing_active = False
        self.workers: List[asyncio.Task] = []
    
    async def start_processing(self):
        """Start real-time processing"""
        if self.processing_active:
            return
        
        self.processing_active = True
        
        # Start worker tasks
        for i in range(self.config.parallel_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started real-time processing with {self.config.parallel_workers} workers")
    
    async def stop_processing(self):
        """Stop real-time processing"""
        self.processing_active = False
        
        # Cancel worker tasks
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Stopped real-time processing")
    
    async def submit_event(self, event: Event):
        """Submit event for processing"""
        await self.processing_queue.put(event)
    
    async def _worker(self, worker_id: str):
        """Worker task for processing events"""
        while self.processing_active:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                
                # Process event
                result = await self.process_event(event)
                self.processed_count += 1
                
                if result:
                    logger.debug(f"Worker {worker_id} processed event {event.event_id}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.error_count += 1
                logger.error(f"Worker {worker_id} processing error: {e}")
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process a single event"""
        try:
            # Simulate processing
            await asyncio.sleep(0.001)  # 1ms processing time
            
            # Transform event data
            processed_data = {
                **event.data,
                "processed_at": time.time(),
                "processing_duration": 0.001
            }
            
            # Create processed event
            processed_event = Event(
                event_id=str(uuid.uuid4()),
                event_type=event.event_type,
                stream_id=f"{event.stream_id}_processed",
                data=processed_data,
                metadata={
                    **event.metadata,
                    "original_event_id": event.event_id,
                    "processor": "real_time"
                },
                correlation_id=event.correlation_id,
                causation_id=event.event_id
            )
            
            return processed_event
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_id}: {e}")
            return None
    
    async def process_batch(self, events: List[Event]) -> List[Event]:
        """Process a batch of events"""
        processed_events = []
        
        for event in events:
            processed_event = await self.process_event(event)
            if processed_event:
                processed_events.append(processed_event)
        
        return processed_events

class BatchProcessor(StreamProcessor):
    """
    Batch event processor
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.batch_buffer: List[Event] = []
        self.last_batch_time = time.time()
        self.processing_active = False
        self.batch_task: Optional[asyncio.Task] = None
    
    async def start_processing(self):
        """Start batch processing"""
        if self.processing_active:
            return
        
        self.processing_active = True
        self.batch_task = asyncio.create_task(self._batch_loop())
        logger.info("Started batch processing")
    
    async def stop_processing(self):
        """Stop batch processing"""
        self.processing_active = False
        
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining events
        if self.batch_buffer:
            await self.process_batch(self.batch_buffer)
            self.batch_buffer.clear()
        
        logger.info("Stopped batch processing")
    
    async def submit_event(self, event: Event):
        """Submit event for batch processing"""
        self.batch_buffer.append(event)
        
        # Process batch if size limit reached
        if len(self.batch_buffer) >= self.config.batch_size:
            await self._process_current_batch()
    
    async def _batch_loop(self):
        """Batch processing loop"""
        while self.processing_active:
            try:
                await asyncio.sleep(self.config.batch_timeout)
                
                # Check if batch should be processed
                if (self.batch_buffer and 
                    (len(self.batch_buffer) >= self.config.batch_size or
                     time.time() - self.last_batch_time >= self.config.batch_timeout)):
                    await self._process_current_batch()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    async def _process_current_batch(self):
        """Process current batch"""
        if not self.batch_buffer:
            return
        
        batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.last_batch_time = time.time()
        
        await self.process_batch(batch)
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process a single event (not used in batch mode)"""
        return None
    
    async def process_batch(self, events: List[Event]) -> List[Event]:
        """Process a batch of events"""
        try:
            logger.info(f"Processing batch of {len(events)} events")
            
            # Simulate batch processing
            await asyncio.sleep(0.1)  # 100ms batch processing time
            
            processed_events = []
            for event in events:
                processed_data = {
                    **event.data,
                    "batch_processed_at": time.time(),
                    "batch_size": len(events)
                }
                
                processed_event = Event(
                    event_id=str(uuid.uuid4()),
                    event_type=event.event_type,
                    stream_id=f"{event.stream_id}_batch_processed",
                    data=processed_data,
                    metadata={
                        **event.metadata,
                        "original_event_id": event.event_id,
                        "processor": "batch"
                    },
                    correlation_id=event.correlation_id,
                    causation_id=event.event_id
                )
                
                processed_events.append(processed_event)
            
            logger.info(f"Processed batch: {len(processed_events)} events")
            return processed_events
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []

class KafkaStreamManager:
    """
    Kafka stream manager
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        self.connected = False
    
    async def connect(self):
        """Connect to Kafka"""
        try:
            if not KAFKA_AVAILABLE:
                raise ImportError("Kafka not available")
            
            # Create producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            # Create consumer
            self.consumer = KafkaConsumer(
                self.config.topic,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                max_poll_records=self.config.max_poll_records,
                session_timeout_ms=self.config.session_timeout_ms,
                heartbeat_interval_ms=self.config.heartbeat_interval_ms,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            self.connected = True
            logger.info(f"Connected to Kafka: {self.config.topic}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Kafka"""
        try:
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
            
            self.connected = False
            logger.info("Disconnected from Kafka")
            
        except Exception as e:
            logger.error(f"Failed to disconnect from Kafka: {e}")
    
    async def publish_event(self, event: Event) -> bool:
        """Publish event to Kafka"""
        try:
            if not self.connected or not self.producer:
                raise RuntimeError("Not connected to Kafka")
            
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "stream_id": event.stream_id,
                "data": event.data,
                "metadata": event.metadata,
                "timestamp": event.timestamp,
                "version": event.version,
                "correlation_id": event.correlation_id,
                "causation_id": event.causation_id
            }
            
            # Send to Kafka
            future = self.producer.send(
                self.config.topic,
                value=event_data,
                key=event.stream_id
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Published event {event.event_id} to Kafka")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event to Kafka: {e}")
            return False
    
    async def consume_events(self) -> AsyncGenerator[Event, None]:
        """Consume events from Kafka"""
        try:
            if not self.connected or not self.consumer:
                raise RuntimeError("Not connected to Kafka")
            
            for message in self.consumer:
                try:
                    event_data = message.value
                    
                    event = Event(
                        event_id=event_data["event_id"],
                        event_type=EventType(event_data["event_type"]),
                        stream_id=event_data["stream_id"],
                        data=event_data["data"],
                        metadata=event_data["metadata"],
                        timestamp=event_data["timestamp"],
                        version=event_data["version"],
                        correlation_id=event_data.get("correlation_id"),
                        causation_id=event_data.get("causation_id")
                    )
                    
                    yield event
                    
                except Exception as e:
                    logger.error(f"Failed to parse Kafka message: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Kafka consumption error: {e}")

class WebSocketStreamManager:
    """
    WebSocket stream manager for real-time communication
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.server: Optional[websockets.WebSocketServer] = None
        self.connected = False
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server"""
        try:
            if not WEBSOCKETS_AVAILABLE:
                raise ImportError("WebSockets not available")
            
            self.server = await websockets.serve(
                self._handle_connection,
                host,
                port
            )
            
            self.connected = True
            logger.info(f"WebSocket server started on {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop_server(self):
        """Stop WebSocket server"""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            self.connected = False
            logger.info("WebSocket server stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop WebSocket server: {e}")
    
    async def _handle_connection(self, websocket, path):
        """Handle WebSocket connection"""
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = websocket
        
        try:
            logger.info(f"WebSocket connection established: {connection_id}")
            
            async for message in websocket:
                try:
                    # Parse incoming event
                    event_data = json.loads(message)
                    event = Event(
                        event_id=event_data.get("event_id", str(uuid.uuid4())),
                        event_type=EventType(event_data["event_type"]),
                        stream_id=event_data["stream_id"],
                        data=event_data["data"],
                        metadata={**event_data.get("metadata", {}), "connection_id": connection_id}
                    )
                    
                    # Echo back the event (in real implementation, would process it)
                    response = {
                        "status": "received",
                        "event_id": event.event_id,
                        "timestamp": time.time()
                    }
                    
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    error_response = {"error": "Invalid JSON"}
                    await websocket.send(json.dumps(error_response))
                except Exception as e:
                    logger.error(f"WebSocket message processing error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.connections.pop(connection_id, None)
    
    async def broadcast_event(self, event: Event):
        """Broadcast event to all connected clients"""
        if not self.connections:
            return
        
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "stream_id": event.stream_id,
            "data": event.data,
            "metadata": event.metadata,
            "timestamp": event.timestamp
        }
        
        message = json.dumps(event_data)
        
        # Send to all connections
        disconnected = []
        for connection_id, websocket in self.connections.items():
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(connection_id)
            except Exception as e:
                logger.error(f"Failed to send to connection {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Remove disconnected connections
        for connection_id in disconnected:
            self.connections.pop(connection_id, None)

class EventProcessor:
    """
    Main event processing manager
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.event_store = EventStore(redis_client)
        self.stream_managers: Dict[str, Any] = {}
        self.processors: Dict[str, StreamProcessor] = {}
        self.processing_active = False
        self.processing_tasks: List[asyncio.Task] = []
    
    def add_stream_manager(self, stream_id: str, manager: Any):
        """Add stream manager"""
        self.stream_managers[stream_id] = manager
        logger.info(f"Added stream manager for {stream_id}")
    
    def add_processor(self, processor_id: str, processor: StreamProcessor):
        """Add event processor"""
        self.processors[processor_id] = processor
        logger.info(f"Added processor {processor_id}")
    
    async def start_processing(self):
        """Start event processing"""
        if self.processing_active:
            return
        
        self.processing_active = True
        
        # Start all processors
        for processor_id, processor in self.processors.items():
            await processor.start_processing()
        
        # Start stream consumption tasks
        for stream_id, manager in self.stream_managers.items():
            if hasattr(manager, 'consume_events'):
                task = asyncio.create_task(self._consume_stream(stream_id, manager))
                self.processing_tasks.append(task)
        
        logger.info("Event processing started")
    
    async def stop_processing(self):
        """Stop event processing"""
        self.processing_active = False
        
        # Stop all processors
        for processor in self.processors.values():
            await processor.stop_processing()
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.processing_tasks.clear()
        
        logger.info("Event processing stopped")
    
    async def _consume_stream(self, stream_id: str, manager: Any):
        """Consume events from stream"""
        try:
            async for event in manager.consume_events():
                # Store event
                await self.event_store.append_event(event)
                
                # Process event with all processors
                for processor in self.processors.values():
                    if hasattr(processor, 'submit_event'):
                        await processor.submit_event(event)
                
        except Exception as e:
            logger.error(f"Stream consumption error for {stream_id}: {e}")
    
    async def publish_event(self, event: Event, stream_id: str = None) -> bool:
        """Publish event to stream"""
        try:
            # Store event
            await self.event_store.append_event(event)
            
            # Publish to stream manager if specified
            if stream_id and stream_id in self.stream_managers:
                manager = self.stream_managers[stream_id]
                if hasattr(manager, 'publish_event'):
                    return await manager.publish_event(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            "processing_active": self.processing_active,
            "stream_managers": len(self.stream_managers),
            "processors": len(self.processors),
            "active_tasks": len(self.processing_tasks)
        }
        
        # Add processor-specific stats
        for processor_id, processor in self.processors.items():
            if hasattr(processor, 'processed_count'):
                stats[f"{processor_id}_processed"] = processor.processed_count
            if hasattr(processor, 'error_count'):
                stats[f"{processor_id}_errors"] = processor.error_count
        
        return stats

# Global event processor
event_processor = EventProcessor()

# Decorator for event handling
def event_handler(event_type: EventType):
    """Decorator for event handlers"""
    def decorator(func):
        event_processor.event_store.register_handler(event_type, func)
        return func
    return decorator

# Utility functions
def create_event(
    event_type: EventType,
    stream_id: str,
    data: Dict[str, Any],
    metadata: Dict[str, Any] = None
) -> Event:
    """Create a new event"""
    return Event(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        stream_id=stream_id,
        data=data,
        metadata=metadata or {}
    )

async def publish_event_async(
    event: Event,
    stream_id: str = None
) -> bool:
    """Publish event asynchronously"""
    return await event_processor.publish_event(event, stream_id)






























