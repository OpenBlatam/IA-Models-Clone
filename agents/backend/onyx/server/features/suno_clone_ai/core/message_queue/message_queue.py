"""
Message Queue

Utilities for message queuing.
"""

import logging
import queue
import threading
from typing import Any, Optional, Callable, Dict, List
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Message data class."""
    topic: str
    payload: Any
    timestamp: datetime = None
    message_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.message_id is None:
            import uuid
            self.message_id = str(uuid.uuid4())


class MessageQueue:
    """Message queue manager."""
    
    def __init__(
        self,
        maxsize: int = 1000
    ):
        """
        Initialize message queue.
        
        Args:
            maxsize: Maximum queue size
        """
        self.queues: Dict[str, queue.Queue] = {}
        self.maxsize = maxsize
        self.consumers: Dict[str, List[threading.Thread]] = defaultdict(list)
        self.running = False
    
    def create_topic(self, topic: str) -> None:
        """
        Create topic.
        
        Args:
            topic: Topic name
        """
        if topic not in self.queues:
            self.queues[topic] = queue.Queue(maxsize=self.maxsize)
            logger.info(f"Created topic: {topic}")
    
    def publish(
        self,
        topic: str,
        payload: Any
    ) -> str:
        """
        Publish message to topic.
        
        Args:
            topic: Topic name
            payload: Message payload
            
        Returns:
            Message ID
        """
        if topic not in self.queues:
            self.create_topic(topic)
        
        message = Message(topic=topic, payload=payload)
        
        try:
            self.queues[topic].put(message, timeout=1.0)
            logger.debug(f"Published message to {topic}: {message.message_id}")
            return message.message_id
        except queue.Full:
            logger.warning(f"Queue full for topic: {topic}")
            raise
    
    def subscribe(
        self,
        topic: str,
        handler: Callable,
        num_workers: int = 1
    ) -> None:
        """
        Subscribe to topic.
        
        Args:
            topic: Topic name
            handler: Message handler function
            num_workers: Number of worker threads
        """
        if topic not in self.queues:
            self.create_topic(topic)
        
        self.running = True
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(topic, handler),
                daemon=True,
                name=f"consumer_{topic}_{i}"
            )
            worker.start()
            self.consumers[topic].append(worker)
        
        logger.info(f"Subscribed to topic: {topic} with {num_workers} workers")
    
    def _worker(
        self,
        topic: str,
        handler: Callable
    ) -> None:
        """Worker thread for consuming messages."""
        while self.running:
            try:
                message = self.queues[topic].get(timeout=1.0)
                
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                
                self.queues[topic].task_done()
            
            except queue.Empty:
                continue
    
    def stop(self) -> None:
        """Stop all consumers."""
        self.running = False
        
        for workers in self.consumers.values():
            for worker in workers:
                worker.join(timeout=2.0)
        
        logger.info("Stopped all consumers")
    
    def get_queue_size(self, topic: str) -> int:
        """Get queue size for topic."""
        if topic not in self.queues:
            return 0
        return self.queues[topic].qsize()


def create_message_queue(**kwargs) -> MessageQueue:
    """Create message queue."""
    return MessageQueue(**kwargs)


def publish_message(
    queue: MessageQueue,
    topic: str,
    payload: Any
) -> str:
    """Publish message."""
    return queue.publish(topic, payload)


def consume_messages(
    queue: MessageQueue,
    topic: str,
    handler: Callable,
    **kwargs
) -> None:
    """Subscribe to topic."""
    queue.subscribe(topic, handler, **kwargs)

