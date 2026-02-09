"""
Queue Manager for Piel Mejorador AI SAM3
========================================

Advanced queue system with Redis/RabbitMQ support.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import importlib.util

logger = logging.getLogger(__name__)


class QueueBackend(Enum):
    """Queue backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"


@dataclass
class QueueMessage:
    """Queue message structure."""
    id: str
    task_id: str
    data: Dict[str, Any]
    priority: int = 0
    created_at: datetime = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "QueueMessage":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class QueueBackendInterface(ABC):
    """Interface for queue backends."""
    
    @abstractmethod
    async def enqueue(self, queue_name: str, message: QueueMessage) -> bool:
        """Enqueue a message."""
        pass
    
    @abstractmethod
    async def dequeue(self, queue_name: str, timeout: float = 0) -> Optional[QueueMessage]:
        """Dequeue a message."""
        pass
    
    @abstractmethod
    async def get_queue_size(self, queue_name: str) -> int:
        """Get queue size."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close connection."""
        pass


class MemoryQueueBackend(QueueBackendInterface):
    """In-memory queue backend."""
    
    def __init__(self):
        """Initialize memory backend."""
        self._queues: Dict[str, asyncio.Queue] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
    
    async def enqueue(self, queue_name: str, message: QueueMessage) -> bool:
        """Enqueue message."""
        if queue_name not in self._queues:
            self._queues[queue_name] = asyncio.Queue()
            self._locks[queue_name] = asyncio.Lock()
        
        await self._queues[queue_name].put(message)
        return True
    
    async def dequeue(self, queue_name: str, timeout: float = 0) -> Optional[QueueMessage]:
        """Dequeue message."""
        if queue_name not in self._queues:
            return None
        
        try:
            if timeout > 0:
                message = await asyncio.wait_for(
                    self._queues[queue_name].get(),
                    timeout=timeout
                )
            else:
                message = await self._queues[queue_name].get()
            return message
        except asyncio.TimeoutError:
            return None
    
    async def get_queue_size(self, queue_name: str) -> int:
        """Get queue size."""
        if queue_name not in self._queues:
            return 0
        return self._queues[queue_name].qsize()
    
    async def close(self):
        """Close (no-op for memory)."""
        pass


class QueueManager:
    """
    Advanced queue manager with multiple backend support.
    
    Features:
    - Multiple backends (Memory, Redis, RabbitMQ)
    - Priority queues
    - Retry logic
    - Dead letter queue
    - Message persistence
    """
    
    def __init__(self, backend: QueueBackend = QueueBackend.MEMORY, **backend_config):
        """
        Initialize queue manager.
        
        Args:
            backend: Backend type
            **backend_config: Backend-specific configuration
        """
        self.backend_type = backend
        self._backend: QueueBackendInterface = self._create_backend(backend, **backend_config)
        self._consumers: Dict[str, List[Callable]] = {}
        self._running = False
        
        self._stats = {
            "messages_enqueued": 0,
            "messages_dequeued": 0,
            "messages_failed": 0,
            "messages_retried": 0,
        }
    
    def _create_backend(
        self,
        backend: QueueBackend,
        **config
    ) -> QueueBackendInterface:
        """Create backend instance."""
        if backend == QueueBackend.MEMORY:
            return MemoryQueueBackend()
        elif backend == QueueBackend.REDIS:
            try:
                from .redis_queue_backend import RedisQueueBackend
                return RedisQueueBackend(**config)
            except ImportError:
                logger.warning("Redis not available, falling back to memory")
                return MemoryQueueBackend()
        elif backend == QueueBackend.RABBITMQ:
            try:
                from .rabbitmq_queue_backend import RabbitMQQueueBackend
                return RabbitMQQueueBackend(**config)
            except ImportError:
                logger.warning("RabbitMQ not available, falling back to memory")
                return MemoryQueueBackend()
        else:
            return MemoryQueueBackend()
    
    async def enqueue(
        self,
        queue_name: str,
        task_id: str,
        data: Dict[str, Any],
        priority: int = 0
    ) -> bool:
        """
        Enqueue a message.
        
        Args:
            queue_name: Queue name
            task_id: Task identifier
            data: Message data
            priority: Message priority (higher = more priority)
            
        Returns:
            True if enqueued
        """
        import uuid
        message = QueueMessage(
            id=str(uuid.uuid4()),
            task_id=task_id,
            data=data,
            priority=priority
        )
        
        success = await self._backend.enqueue(queue_name, message)
        if success:
            self._stats["messages_enqueued"] += 1
        
        return success
    
    async def dequeue(
        self,
        queue_name: str,
        timeout: float = 0
    ) -> Optional[QueueMessage]:
        """
        Dequeue a message.
        
        Args:
            queue_name: Queue name
            timeout: Timeout in seconds
            
        Returns:
            Message or None
        """
        message = await self._backend.dequeue(queue_name, timeout)
        if message:
            self._stats["messages_dequeued"] += 1
        return message
    
    async def register_consumer(
        self,
        queue_name: str,
        handler: Callable
    ):
        """
        Register a consumer for a queue.
        
        Args:
            queue_name: Queue name
            handler: Handler function (message: QueueMessage) -> None
        """
        if queue_name not in self._consumers:
            self._consumers[queue_name] = []
        self._consumers[queue_name].append(handler)
        logger.info(f"Registered consumer for queue: {queue_name}")
    
    async def start_consuming(self):
        """Start consuming messages from all registered queues."""
        self._running = True
        
        tasks = []
        for queue_name, handlers in self._consumers.items():
            for handler in handlers:
                task = asyncio.create_task(
                    self._consume_queue(queue_name, handler)
                )
                tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _consume_queue(
        self,
        queue_name: str,
        handler: Callable
    ):
        """Consume messages from a queue."""
        logger.info(f"Starting consumer for queue: {queue_name}")
        
        while self._running:
            try:
                message = await self.dequeue(queue_name, timeout=1.0)
                if message:
                    try:
                        await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)
                    except Exception as e:
                        logger.error(f"Error processing message {message.id}: {e}")
                        message.retry_count += 1
                        
                        if message.retry_count < message.max_retries:
                            # Retry
                            await self.enqueue(
                                queue_name,
                                message.task_id,
                                message.data,
                                message.priority
                            )
                            self._stats["messages_retried"] += 1
                        else:
                            # Dead letter queue
                            await self.enqueue(
                                f"{queue_name}_dlq",
                                message.task_id,
                                message.data,
                                -1  # Low priority
                            )
                            self._stats["messages_failed"] += 1
            except Exception as e:
                logger.error(f"Error in queue consumer: {e}")
                await asyncio.sleep(1.0)
    
    async def stop_consuming(self):
        """Stop consuming messages."""
        self._running = False
    
    async def get_queue_size(self, queue_name: str) -> int:
        """Get queue size."""
        return await self._backend.get_queue_size(queue_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            "backend": self.backend_type.value,
            "active_consumers": sum(len(handlers) for handlers in self._consumers.values()),
        }
    
    async def close(self):
        """Close queue manager."""
        await self.stop_consuming()
        await self._backend.close()

