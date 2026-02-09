from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from contextlib import asynccontextmanager
import redis.asyncio as redis
import orjson
import structlog
from prometheus_client import Counter, Histogram, Gauge
from src.core.config import EventSettings
from src.core.exceptions import BusinessException
from typing import Any, List, Dict, Optional
"""
📡 Ultra-Optimized Event Publisher
==================================

Production-grade event publishing with:
- Redis pub/sub
- Batch processing
- Reliable delivery
- Retry mechanisms
- Performance monitoring
"""





class EventPublisher:
    """
    Ultra-optimized event publisher with Redis integration,
    batch processing, and reliable message delivery.
    """
    
    def __init__(self, config: EventSettings):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Redis connection
        self.redis_client = None
        self.redis_pool = None
        
        # Event processing
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.batch_size = config.BATCH_SIZE
        self.max_retries = config.MAX_RETRIES
        
        # Subscribers
        self.subscribers = {}
        self.subscriber_tasks = {}
        
        # Performance metrics
        self.events_published = 0
        self.events_delivered = 0
        self.events_failed = 0
        self.total_publish_time = 0.0
        self.total_delivery_time = 0.0
        
        # Prometheus metrics
        self.events_published_total = Counter(
            'events_published_total',
            'Total events published',
            ['event_type']
        )
        
        self.events_delivered_total = Counter(
            'events_delivered_total',
            'Total events delivered',
            ['event_type']
        )
        
        self.events_failed_total = Counter(
            'events_failed_total',
            'Total events failed',
            ['event_type']
        )
        
        self.publish_duration = Histogram(
            'event_publish_duration_seconds',
            'Event publish duration',
            ['event_type']
        )
        
        self.delivery_duration = Histogram(
            'event_delivery_duration_seconds',
            'Event delivery duration',
            ['event_type']
        )
        
        self.queue_size = Gauge(
            'event_queue_size',
            'Current event queue size'
        )
        
        # Background tasks
        self.publisher_task = None
        self.batch_processor_task = None
        self.retry_task = None
        
        # Health status
        self.is_healthy = False
        self.last_health_check = None
        
        self.logger.info("Event Publisher initialized")
    
    async def initialize(self) -> Any:
        """Initialize event publisher"""
        
        self.logger.info("Initializing Event Publisher...")
        
        try:
            # Create Redis connection pool
            self.redis_pool = redis.ConnectionPool.from_url(
                self.config.BROKER_URL,
                max_connections=20,
                decode_responses=False
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self._test_connection()
            
            # Start background tasks
            self.publisher_task = asyncio.create_task(self._publisher_loop())
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            self.retry_task = asyncio.create_task(self._retry_processor())
            
            # Set health status
            self.is_healthy = True
            self.last_health_check = time.time()
            
            self.logger.info("Event Publisher initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Event Publisher: {e}")
            raise BusinessException(f"Event Publisher initialization failed: {e}")
    
    async def cleanup(self) -> Any:
        """Cleanup event publisher"""
        
        self.logger.info("Cleaning up Event Publisher...")
        
        # Stop background tasks
        tasks = [
            self.publisher_task,
            self.batch_processor_task,
            self.retry_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close Redis client
        if self.redis_client:
            await self.redis_client.close()
        
        # Close connection pool
        if self.redis_pool:
            await self.redis_pool.disconnect()
        
        self.logger.info("Event Publisher cleanup completed")
    
    async def publish(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Publish an event"""
        
        start_time = time.time()
        
        try:
            # Create event
            event = {
                "id": f"{event_type}_{int(time.time() * 1000)}",
                "type": event_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "retry_count": 0
            }
            
            # Add to queue
            await self.event_queue.put(event)
            
            # Update metrics
            self.events_published += 1
            self.events_published_total.labels(event_type=event_type).inc()
            
            publish_time = time.time() - start_time
            self.total_publish_time += publish_time
            self.publish_duration.labels(event_type=event_type).observe(publish_time)
            
            # Update queue size metric
            self.queue_size.set(self.event_queue.qsize())
            
            self.logger.debug(f"Event queued: {event_type}", event_id=event["id"])
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish event {event_type}: {e}")
            self.events_failed += 1
            self.events_failed_total.labels(event_type=event_type).inc()
            return False
    
    async def publish_batch(self, events: List[Dict[str, Any]]) -> bool:
        """Publish multiple events efficiently"""
        
        start_time = time.time()
        
        try:
            # Process events in parallel
            tasks = []
            for event_data in events:
                event_type = event_data.get("type", "unknown")
                data = event_data.get("data", {})
                task = self.publish(event_type, data)
                tasks.append(task)
            
            # Wait for all events to be queued
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            success_count = sum(1 for result in results if result is True)
            total_count = len(events)
            
            batch_time = time.time() - start_time
            
            self.logger.info(
                f"Batch publish completed: {success_count}/{total_count} events",
                batch_size=total_count,
                duration=batch_time
            )
            
            return success_count == total_count
            
        except Exception as e:
            self.logger.error(f"Batch publish failed: {e}")
            return False
    
    async def subscribe(self, event_type: str, handler: Callable) -> bool:
        """Subscribe to events"""
        
        try:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            self.subscribers[event_type].append(handler)
            
            # Start subscriber task if not already running
            if event_type not in self.subscriber_tasks:
                self.subscriber_tasks[event_type] = asyncio.create_task(
                    self._subscriber_loop(event_type)
                )
            
            self.logger.info(f"Subscribed to events: {event_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {event_type}: {e}")
            return False
    
    async def unsubscribe(self, event_type: str, handler: Callable) -> bool:
        """Unsubscribe from events"""
        
        try:
            if event_type in self.subscribers:
                if handler in self.subscribers[event_type]:
                    self.subscribers[event_type].remove(handler)
                    
                    # Stop subscriber task if no more handlers
                    if not self.subscribers[event_type]:
                        if event_type in self.subscriber_tasks:
                            self.subscriber_tasks[event_type].cancel()
                            del self.subscriber_tasks[event_type]
                        del self.subscribers[event_type]
                    
                    self.logger.info(f"Unsubscribed from events: {event_type}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {event_type}: {e}")
            return False
    
    async def _publisher_loop(self) -> Any:
        """Background task for publishing events"""
        
        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()
                
                # Publish to Redis
                await self._publish_to_redis(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Publisher loop error: {e}")
                await asyncio.sleep(1)  # Wait before retry
    
    async def _publish_to_redis(self, event: Dict[str, Any]):
        """Publish event to Redis"""
        
        try:
            # Serialize event
            event_data = orjson.dumps(event)
            
            # Publish to Redis channel
            channel = f"events:{event['type']}"
            result = await self.redis_client.publish(channel, event_data)
            
            if result > 0:
                self.events_delivered += 1
                self.events_delivered_total.labels(event_type=event['type']).inc()
                
                self.logger.debug(
                    f"Event published to Redis: {event['type']}",
                    event_id=event['id'],
                    subscribers=result
                )
            else:
                # No subscribers, store for retry
                await self._store_for_retry(event)
                
        except Exception as e:
            self.logger.error(f"Failed to publish to Redis: {e}")
            await self._store_for_retry(event)
    
    async def _subscriber_loop(self, event_type: str):
        """Background task for handling subscribed events"""
        
        try:
            # Subscribe to Redis channel
            pubsub = self.redis_client.pubsub()
            channel = f"events:{event_type}"
            await pubsub.subscribe(channel)
            
            self.logger.info(f"Subscribed to Redis channel: {channel}")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        # Parse event
                        event_data = orjson.loads(message['data'])
                        
                        # Call handlers
                        await self._call_handlers(event_type, event_data)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process event: {e}")
                        
        except asyncio.CancelledError:
            await pubsub.close()
        except Exception as e:
            self.logger.error(f"Subscriber loop error for {event_type}: {e}")
    
    async def _call_handlers(self, event_type: str, event_data: Dict[str, Any]):
        """Call event handlers"""
        
        start_time = time.time()
        
        try:
            handlers = self.subscribers.get(event_type, [])
            
            # Call handlers in parallel
            tasks = []
            for handler in handlers:
                task = self._call_handler(handler, event_data)
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            delivery_time = time.time() - start_time
            self.total_delivery_time += delivery_time
            self.delivery_duration.labels(event_type=event_type).observe(delivery_time)
            
        except Exception as e:
            self.logger.error(f"Failed to call handlers: {e}")
    
    async def _call_handler(self, handler: Callable, event_data: Dict[str, Any]):
        """Call a single event handler"""
        
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event_data)
            else:
                handler(event_data)
                
        except Exception as e:
            self.logger.error(f"Handler error: {e}")
    
    async def _batch_processor(self) -> Any:
        """Background task for batch processing"""
        
        while True:
            try:
                # Collect events for batch processing
                batch = []
                
                # Get events up to batch size
                for _ in range(self.batch_size):
                    try:
                        event = await asyncio.wait_for(
                            self.event_queue.get(), timeout=1.0
                        )
                        batch.append(event)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Process batch
                    await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of events"""
        
        try:
            # Group events by type
            events_by_type = {}
            for event in batch:
                event_type = event['type']
                if event_type not in events_by_type:
                    events_by_type[event_type] = []
                events_by_type[event_type].append(event)
            
            # Publish each group
            for event_type, events in events_by_type.items():
                await self._publish_batch_to_redis(event_type, events)
            
            self.logger.debug(f"Processed batch of {len(batch)} events")
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
    
    async def _publish_batch_to_redis(self, event_type: str, events: List[Dict[str, Any]]):
        """Publish batch of events to Redis"""
        
        try:
            # Use pipeline for better performance
            async with self.redis_client.pipeline() as pipe:
                channel = f"events:{event_type}"
                
                for event in events:
                    event_data = orjson.dumps(event)
                    pipe.publish(channel, event_data)
                
                results = await pipe.execute()
            
            # Update metrics
            delivered_count = sum(1 for result in results if result > 0)
            self.events_delivered += delivered_count
            self.events_delivered_total.labels(event_type=event_type).inc(delivered_count)
            
        except Exception as e:
            self.logger.error(f"Batch Redis publish failed: {e}")
            # Store failed events for retry
            for event in events:
                await self._store_for_retry(event)
    
    async def _store_for_retry(self, event: Dict[str, Any]):
        """Store event for retry"""
        
        try:
            # Increment retry count
            event['retry_count'] += 1
            
            if event['retry_count'] <= self.max_retries:
                # Store in Redis for retry
                retry_key = f"retry:{event['type']}:{event['id']}"
                event_data = orjson.dumps(event)
                
                # Set with expiration (5 minutes)
                await self.redis_client.setex(retry_key, 300, event_data)
                
                self.logger.warning(
                    f"Event stored for retry: {event['type']}",
                    event_id=event['id'],
                    retry_count=event['retry_count']
                )
            else:
                # Max retries exceeded
                self.events_failed += 1
                self.events_failed_total.labels(event_type=event['type']).inc()
                
                self.logger.error(
                    f"Event failed after max retries: {event['type']}",
                    event_id=event['id']
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store event for retry: {e}")
    
    async def _retry_processor(self) -> Any:
        """Background task for processing retry events"""
        
        while True:
            try:
                # Get retry events
                retry_keys = await self.redis_client.keys("retry:*")
                
                for retry_key in retry_keys:
                    try:
                        # Get event data
                        event_data = await self.redis_client.get(retry_key)
                        if event_data:
                            event = orjson.loads(event_data)
                            
                            # Try to publish again
                            await self._publish_to_redis(event)
                            
                            # Remove from retry queue
                            await self.redis_client.delete(retry_key)
                            
                    except Exception as e:
                        self.logger.error(f"Retry processing failed: {e}")
                
                # Wait before next retry cycle
                await asyncio.sleep(30)  # 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Retry processor error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _test_connection(self) -> Any:
        """Test Redis connection"""
        
        try:
            await self.redis_client.ping()
            self.logger.info("Redis connection test successful")
            
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            raise BusinessException(f"Redis connection failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get event publisher performance metrics"""
        
        try:
            return {
                "events_published": self.events_published,
                "events_delivered": self.events_delivered,
                "events_failed": self.events_failed,
                "delivery_rate": (
                    self.events_delivered / self.events_published * 100
                    if self.events_published > 0 else 0
                ),
                "average_publish_time": (
                    self.total_publish_time / self.events_published
                    if self.events_published > 0 else 0
                ),
                "average_delivery_time": (
                    self.total_delivery_time / self.events_delivered
                    if self.events_delivered > 0 else 0
                ),
                "queue_size": self.event_queue.qsize(),
                "subscribers": len(self.subscribers),
                "is_healthy": self.is_healthy
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        
        try:
            start_time = time.time()
            
            # Test Redis connection
            await self.redis_client.ping()
            
            response_time = time.time() - start_time
            
            # Update health status
            self.is_healthy = True
            self.last_health_check = time.time()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "queue_size": self.event_queue.qsize(),
                "subscribers": len(self.subscribers),
                "events_published": self.events_published,
                "events_delivered": self.events_delivered
            }
            
        except Exception as e:
            self.is_healthy = False
            self.logger.error(f"Health check failed: {e}")
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_health_check": self.last_health_check
            } 