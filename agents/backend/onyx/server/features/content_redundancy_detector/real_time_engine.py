"""
Real-time processing engine for live content analysis
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Stream types"""
    CONTENT_ANALYSIS = "content_analysis"
    SIMILARITY_DETECTION = "similarity_detection"
    QUALITY_ASSESSMENT = "quality_assessment"
    BATCH_PROCESSING = "batch_processing"
    SYSTEM_MONITORING = "system_monitoring"
    AI_ML_PROCESSING = "ai_ml_processing"


class StreamStatus(Enum):
    """Stream status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamEvent:
    """Stream event"""
    id: str
    stream_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stream:
    """Real-time stream"""
    id: str
    name: str
    stream_type: StreamType
    status: StreamStatus = StreamStatus.ACTIVE
    subscribers: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    event_count: int = 0
    error_count: int = 0


@dataclass
class StreamSubscriber:
    """Stream subscriber"""
    id: str
    callback: Callable[[StreamEvent], None]
    stream_types: Set[StreamType] = field(default_factory=set)
    filters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class RealTimeEngine:
    """Real-time processing engine"""
    
    def __init__(self, max_streams: int = 100, max_events_per_stream: int = 1000):
        self._streams: Dict[str, Stream] = {}
        self._subscribers: Dict[str, StreamSubscriber] = {}
        self._event_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_events_per_stream))
        self._max_streams = max_streams
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._is_running = False
        self._event_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the real-time engine"""
        if self._is_running:
            return
        
        self._is_running = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("Real-time engine started")
    
    async def stop(self) -> None:
        """Stop the real-time engine"""
        self._is_running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        self._executor.shutdown(wait=True)
        logger.info("Real-time engine stopped")
    
    async def create_stream(self, stream_id: str, name: str, stream_type: StreamType) -> Stream:
        """Create a new real-time stream"""
        if len(self._streams) >= self._max_streams:
            raise ValueError("Maximum number of streams reached")
        
        if stream_id in self._streams:
            raise ValueError(f"Stream {stream_id} already exists")
        
        stream = Stream(
            id=stream_id,
            name=name,
            stream_type=stream_type
        )
        
        self._streams[stream_id] = stream
        logger.info(f"Stream created: {stream_id} ({stream_type.value})")
        
        return stream
    
    async def delete_stream(self, stream_id: str) -> bool:
        """Delete a stream"""
        if stream_id not in self._streams:
            return False
        
        # Notify subscribers
        stream = self._streams[stream_id]
        for subscriber_id in stream.subscribers:
            if subscriber_id in self._subscribers:
                subscriber = self._subscribers[subscriber_id]
                if subscriber.is_active:
                    try:
                        event = StreamEvent(
                            id=f"stream_deleted_{int(time.time())}",
                            stream_id=stream_id,
                            event_type="stream_deleted",
                            data={"stream_id": stream_id, "stream_name": stream.name}
                        )
                        await self._notify_subscriber(subscriber, event)
                    except Exception as e:
                        logger.error(f"Error notifying subscriber {subscriber_id}: {e}")
        
        del self._streams[stream_id]
        if stream_id in self._event_history:
            del self._event_history[stream_id]
        
        logger.info(f"Stream deleted: {stream_id}")
        return True
    
    async def subscribe_to_stream(self, subscriber_id: str, stream_id: str, 
                                 callback: Callable[[StreamEvent], None],
                                 filters: Optional[Dict[str, Any]] = None) -> bool:
        """Subscribe to a stream"""
        if stream_id not in self._streams:
            return False
        
        subscriber = StreamSubscriber(
            id=subscriber_id,
            callback=callback,
            stream_types={self._streams[stream_id].stream_type},
            filters=filters or {}
        )
        
        self._subscribers[subscriber_id] = subscriber
        self._streams[stream_id].subscribers.add(subscriber_id)
        
        logger.info(f"Subscriber {subscriber_id} subscribed to stream {stream_id}")
        return True
    
    async def unsubscribe_from_stream(self, subscriber_id: str, stream_id: str) -> bool:
        """Unsubscribe from a stream"""
        if stream_id not in self._streams or subscriber_id not in self._subscribers:
            return False
        
        self._streams[stream_id].subscribers.discard(subscriber_id)
        del self._subscribers[subscriber_id]
        
        logger.info(f"Subscriber {subscriber_id} unsubscribed from stream {stream_id}")
        return True
    
    async def publish_event(self, stream_id: str, event_type: str, data: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Publish an event to a stream"""
        if stream_id not in self._streams:
            return False
        
        stream = self._streams[stream_id]
        if stream.status != StreamStatus.ACTIVE:
            return False
        
        event = StreamEvent(
            id=f"{event_type}_{int(time.time() * 1000)}",
            stream_id=stream_id,
            event_type=event_type,
            data=data,
            metadata=metadata or {}
        )
        
        # Add to event history
        self._event_history[stream_id].append(event)
        
        # Update stream stats
        stream.last_activity = time.time()
        stream.event_count += 1
        
        # Queue event for processing
        await self._event_queue.put(event)
        
        return True
    
    async def get_stream_events(self, stream_id: str, limit: int = 100) -> List[StreamEvent]:
        """Get recent events from a stream"""
        if stream_id not in self._event_history:
            return []
        
        events = list(self._event_history[stream_id])
        return events[-limit:] if limit > 0 else events
    
    async def get_stream_stats(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get stream statistics"""
        if stream_id not in self._streams:
            return None
        
        stream = self._streams[stream_id]
        return {
            "stream_id": stream_id,
            "name": stream.name,
            "stream_type": stream.stream_type.value,
            "status": stream.status.value,
            "subscribers": len(stream.subscribers),
            "event_count": stream.event_count,
            "error_count": stream.error_count,
            "created_at": stream.created_at,
            "last_activity": stream.last_activity,
            "uptime": time.time() - stream.created_at
        }
    
    async def get_all_streams(self) -> List[Dict[str, Any]]:
        """Get all streams"""
        streams = []
        for stream_id in self._streams:
            stats = await self.get_stream_stats(stream_id)
            if stats:
                streams.append(stats)
        return streams
    
    async def pause_stream(self, stream_id: str) -> bool:
        """Pause a stream"""
        if stream_id not in self._streams:
            return False
        
        self._streams[stream_id].status = StreamStatus.PAUSED
        logger.info(f"Stream paused: {stream_id}")
        return True
    
    async def resume_stream(self, stream_id: str) -> bool:
        """Resume a stream"""
        if stream_id not in self._streams:
            return False
        
        self._streams[stream_id].status = StreamStatus.ACTIVE
        logger.info(f"Stream resumed: {stream_id}")
        return True
    
    async def stop_stream(self, stream_id: str) -> bool:
        """Stop a stream"""
        if stream_id not in self._streams:
            return False
        
        self._streams[stream_id].status = StreamStatus.STOPPED
        logger.info(f"Stream stopped: {stream_id}")
        return True
    
    async def _process_events(self) -> None:
        """Process events from the queue"""
        while self._is_running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                
                # Process event
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _process_event(self, event: StreamEvent) -> None:
        """Process a single event"""
        try:
            stream = self._streams.get(event.stream_id)
            if not stream:
                return
            
            # Notify subscribers
            for subscriber_id in stream.subscribers:
                if subscriber_id in self._subscribers:
                    subscriber = self._subscribers[subscriber_id]
                    if subscriber.is_active:
                        try:
                            await self._notify_subscriber(subscriber, event)
                        except Exception as e:
                            logger.error(f"Error notifying subscriber {subscriber_id}: {e}")
                            stream.error_count += 1
            
        except Exception as e:
            logger.error(f"Error processing event {event.id}: {e}")
    
    async def _notify_subscriber(self, subscriber: StreamSubscriber, event: StreamEvent) -> None:
        """Notify a subscriber of an event"""
        # Check filters
        if not self._event_matches_filters(event, subscriber.filters):
            return
        
        # Call subscriber callback
        try:
            if asyncio.iscoroutinefunction(subscriber.callback):
                await subscriber.callback(event)
            else:
                # Run in thread pool for sync callbacks
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, subscriber.callback, event)
        except Exception as e:
            logger.error(f"Error in subscriber callback: {e}")
    
    def _event_matches_filters(self, event: StreamEvent, filters: Dict[str, Any]) -> bool:
        """Check if event matches subscriber filters"""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key == "event_type" and event.event_type != value:
                return False
            elif key == "data_key" and value not in event.data:
                return False
            elif key.startswith("data."):
                data_key = key[5:]  # Remove "data." prefix
                if data_key not in event.data or event.data[data_key] != value:
                    return False
        
        return True
    
    async def create_content_analysis_stream(self, stream_id: str, name: str) -> Stream:
        """Create a content analysis stream"""
        return await self.create_stream(stream_id, name, StreamType.CONTENT_ANALYSIS)
    
    async def create_similarity_detection_stream(self, stream_id: str, name: str) -> Stream:
        """Create a similarity detection stream"""
        return await self.create_stream(stream_id, name, StreamType.SIMILARITY_DETECTION)
    
    async def create_quality_assessment_stream(self, stream_id: str, name: str) -> Stream:
        """Create a quality assessment stream"""
        return await self.create_stream(stream_id, name, StreamType.QUALITY_ASSESSMENT)
    
    async def create_batch_processing_stream(self, stream_id: str, name: str) -> Stream:
        """Create a batch processing stream"""
        return await self.create_stream(stream_id, name, StreamType.BATCH_PROCESSING)
    
    async def create_system_monitoring_stream(self, stream_id: str, name: str) -> Stream:
        """Create a system monitoring stream"""
        return await self.create_stream(stream_id, name, StreamType.SYSTEM_MONITORING)
    
    async def create_ai_ml_processing_stream(self, stream_id: str, name: str) -> Stream:
        """Create an AI/ML processing stream"""
        return await self.create_stream(stream_id, name, StreamType.AI_ML_PROCESSING)
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        total_events = sum(len(events) for events in self._event_history.values())
        active_streams = sum(1 for stream in self._streams.values() if stream.status == StreamStatus.ACTIVE)
        total_subscribers = len(self._subscribers)
        
        return {
            "total_streams": len(self._streams),
            "active_streams": active_streams,
            "total_subscribers": total_subscribers,
            "total_events": total_events,
            "is_running": self._is_running,
            "queue_size": self._event_queue.qsize()
        }


# Global real-time engine
real_time_engine = RealTimeEngine()


# Helper functions for common use cases
async def create_analysis_stream(stream_id: str, name: str) -> Stream:
    """Create a content analysis stream"""
    return await real_time_engine.create_content_analysis_stream(stream_id, name)


async def create_similarity_stream(stream_id: str, name: str) -> Stream:
    """Create a similarity detection stream"""
    return await real_time_engine.create_similarity_detection_stream(stream_id, name)


async def create_quality_stream(stream_id: str, name: str) -> Stream:
    """Create a quality assessment stream"""
    return await real_time_engine.create_quality_assessment_stream(stream_id, name)


async def publish_analysis_event(stream_id: str, analysis_data: Dict[str, Any]) -> bool:
    """Publish content analysis event"""
    return await real_time_engine.publish_event(
        stream_id, "analysis_completed", analysis_data
    )


async def publish_similarity_event(stream_id: str, similarity_data: Dict[str, Any]) -> bool:
    """Publish similarity detection event"""
    return await real_time_engine.publish_event(
        stream_id, "similarity_detected", similarity_data
    )


async def publish_quality_event(stream_id: str, quality_data: Dict[str, Any]) -> bool:
    """Publish quality assessment event"""
    return await real_time_engine.publish_event(
        stream_id, "quality_assessed", quality_data
    )


async def publish_batch_event(stream_id: str, batch_data: Dict[str, Any]) -> bool:
    """Publish batch processing event"""
    return await real_time_engine.publish_event(
        stream_id, "batch_progress", batch_data
    )


async def publish_system_event(stream_id: str, system_data: Dict[str, Any]) -> bool:
    """Publish system monitoring event"""
    return await real_time_engine.publish_event(
        stream_id, "system_metrics", system_data
    )


async def publish_ai_ml_event(stream_id: str, ai_ml_data: Dict[str, Any]) -> bool:
    """Publish AI/ML processing event"""
    return await real_time_engine.publish_event(
        stream_id, "ai_ml_completed", ai_ml_data
    )


