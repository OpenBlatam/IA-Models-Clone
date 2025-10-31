"""
Task Queue Implementation
=========================

Queue implementations for background task processing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
import heapq
import json
import redis.asyncio as redis

from .types import Task, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)

class TaskQueue(ABC):
    """Abstract base class for task queues."""
    
    @abstractmethod
    async def enqueue(self, task: Task) -> bool:
        """Add a task to the queue."""
        pass
    
    @abstractmethod
    async def dequeue(self) -> Optional[Task]:
        """Get the next task from the queue."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get the current queue size."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all tasks from the queue."""
        pass

class InMemoryTaskQueue(TaskQueue):
    """In-memory task queue implementation."""
    
    def __init__(self):
        self._queue: List[Task] = []
        self._lock = asyncio.Lock()
    
    async def enqueue(self, task: Task) -> bool:
        """Add a task to the queue."""
        try:
            async with self._lock:
                # Use priority-based ordering
                priority_value = self._get_priority_value(task.priority)
                heapq.heappush(self._queue, (priority_value, task.created_at, task))
            
            logger.debug(f"Enqueued task: {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {str(e)}")
            return False
    
    async def dequeue(self) -> Optional[Task]:
        """Get the next task from the queue."""
        try:
            async with self._lock:
                if not self._queue:
                    return None
                
                _, _, task = heapq.heappop(self._queue)
                return task
                
        except Exception as e:
            logger.error(f"Failed to dequeue task: {str(e)}")
            return None
    
    async def size(self) -> int:
        """Get the current queue size."""
        async with self._lock:
            return len(self._queue)
    
    async def clear(self) -> bool:
        """Clear all tasks from the queue."""
        try:
            async with self._lock:
                self._queue.clear()
            logger.info("Cleared task queue")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear queue: {str(e)}")
            return False
    
    def _get_priority_value(self, priority: TaskPriority) -> int:
        """Get numeric priority value for sorting."""
        priority_map = {
            TaskPriority.URGENT: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3
        }
        return priority_map.get(priority, 2)

class RedisTaskQueue(TaskQueue):
    """Redis-based task queue implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", queue_name: str = "business_agents_tasks"):
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.redis_client: Optional[redis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis task queue initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis task queue: {str(e)}")
            raise
    
    async def enqueue(self, task: Task) -> bool:
        """Add a task to the queue."""
        try:
            if not self.redis_client:
                await self.initialize()
            
            # Serialize task
            task_data = self._serialize_task(task)
            
            # Use priority-based scoring
            priority_score = self._get_priority_score(task.priority)
            score = priority_score + task.created_at.timestamp()
            
            # Add to sorted set
            await self.redis_client.zadd(self.queue_name, {task_data: score})
            
            logger.debug(f"Enqueued task in Redis: {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id} in Redis: {str(e)}")
            return False
    
    async def dequeue(self) -> Optional[Task]:
        """Get the next task from the queue."""
        try:
            if not self.redis_client:
                await self.initialize()
            
            # Get task with lowest score (highest priority)
            result = await self.redis_client.zpopmin(self.queue_name, count=1)
            
            if not result:
                return None
            
            task_data = result[0][0]
            task = self._deserialize_task(task_data)
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to dequeue task from Redis: {str(e)}")
            return None
    
    async def size(self) -> int:
        """Get the current queue size."""
        try:
            if not self.redis_client:
                await self.initialize()
            
            return await self.redis_client.zcard(self.queue_name)
            
        except Exception as e:
            logger.error(f"Failed to get queue size: {str(e)}")
            return 0
    
    async def clear(self) -> bool:
        """Clear all tasks from the queue."""
        try:
            if not self.redis_client:
                await self.initialize()
            
            await self.redis_client.delete(self.queue_name)
            logger.info("Cleared Redis task queue")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear Redis queue: {str(e)}")
            return False
    
    def _serialize_task(self, task: Task) -> str:
        """Serialize task to JSON string."""
        task_dict = {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "name": task.name,
            "description": task.description,
            "function_name": task.function_name,
            "args": task.args,
            "kwargs": task.kwargs,
            "priority": task.priority.value,
            "max_retries": task.max_retries,
            "retry_delay": task.retry_delay,
            "timeout": task.timeout,
            "created_at": task.created_at.isoformat(),
            "scheduled_at": task.scheduled_at.isoformat() if task.scheduled_at else None,
            "metadata": task.metadata,
            "tags": task.tags
        }
        return json.dumps(task_dict, default=str)
    
    def _deserialize_task(self, task_data: str) -> Task:
        """Deserialize task from JSON string."""
        task_dict = json.loads(task_data)
        
        return Task(
            task_id=task_dict["task_id"],
            task_type=TaskType(task_dict["task_type"]),
            name=task_dict["name"],
            description=task_dict["description"],
            function_name=task_dict["function_name"],
            args=tuple(task_dict["args"]),
            kwargs=task_dict["kwargs"],
            priority=TaskPriority(task_dict["priority"]),
            max_retries=task_dict["max_retries"],
            retry_delay=task_dict["retry_delay"],
            timeout=task_dict["timeout"],
            created_at=datetime.fromisoformat(task_dict["created_at"]),
            scheduled_at=datetime.fromisoformat(task_dict["scheduled_at"]) if task_dict["scheduled_at"] else None,
            metadata=task_dict["metadata"],
            tags=task_dict["tags"]
        )
    
    def _get_priority_score(self, priority: TaskPriority) -> float:
        """Get priority score for Redis sorting."""
        priority_scores = {
            TaskPriority.URGENT: 0.0,
            TaskPriority.HIGH: 0.25,
            TaskPriority.NORMAL: 0.5,
            TaskPriority.LOW: 0.75
        }
        return priority_scores.get(priority, 0.5)
