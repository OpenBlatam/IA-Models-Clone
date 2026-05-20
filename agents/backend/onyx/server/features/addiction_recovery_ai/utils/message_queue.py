"""
Message Queue for Recovery AI
"""

import queue
import threading
from typing import Dict, List, Optional, Any, Callable
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MessageQueue:
    """Simple message queue implementation"""
    
    def __init__(self, maxsize: int = 1000):
        """
        Initialize message queue
        
        Args:
            maxsize: Maximum queue size
        """
        self.queue = queue.Queue(maxsize=maxsize)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        self.worker_thread = None
        
        logger.info(f"MessageQueue initialized with maxsize={maxsize}")
    
    def publish(self, topic: str, message: Dict[str, Any]):
        """
        Publish message to topic
        
        Args:
            topic: Topic name
            message: Message data
        """
        try:
            self.queue.put_nowait({
                "topic": topic,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            logger.debug(f"Message published to topic: {topic}")
        except queue.Full:
            logger.warning(f"Queue full, message dropped: {topic}")
    
    def subscribe(self, topic: str, callback: Callable):
        """
        Subscribe to topic
        
        Args:
            topic: Topic name
            callback: Callback function
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        
        self.subscribers[topic].append(callback)
        logger.info(f"Subscribed to topic: {topic}")
    
    def _worker_loop(self):
        """Worker loop for processing messages"""
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
                topic = item["topic"]
                message = item["message"]
                
                if topic in self.subscribers:
                    for callback in self.subscribers[topic]:
                        try:
                            callback(message)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def start(self):
        """Start message queue"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("MessageQueue started")
    
    def stop(self):
        """Stop message queue"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("MessageQueue stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queue_size": self.queue.qsize(),
            "subscribers": {topic: len(callbacks) for topic, callbacks in self.subscribers.items()},
            "running": self.running
        }


class EventStream:
    """Event streaming for real-time events"""
    
    def __init__(self):
        """Initialize event stream"""
        self.listeners: List[Callable] = []
        logger.info("EventStream initialized")
    
    def emit(self, event_type: str, data: Dict[str, Any]):
        """
        Emit event
        
        Args:
            event_type: Event type
            data: Event data
        """
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        for listener in self.listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
    
    def on(self, callback: Callable):
        """
        Register event listener
        
        Args:
            callback: Callback function
        """
        self.listeners.append(callback)
    
    def off(self, callback: Callable):
        """
        Unregister event listener
        
        Args:
            callback: Callback function
        """
        if callback in self.listeners:
            self.listeners.remove(callback)

