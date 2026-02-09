"""
Task Queue - Sistema de colas
"""

from typing import Any, Optional
import uuid
from .base import BaseTask
from redis.service import RedisService


class TaskQueue:
    """Sistema de colas para tareas"""

    def __init__(self, redis_service: Optional[RedisService] = None):
        """Inicializa la cola de tareas"""
        self.redis_service = redis_service
        self._queue: list = []

    async def enqueue(self, task: BaseTask, *args, **kwargs) -> str:
        """Encola una tarea"""
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "task": task,
            "args": args,
            "kwargs": kwargs
        }
        
        if self.redis_service:
            # Usar Redis para cola distribuida
            await self.redis_service.set(f"task:{task_id}", task_data)
        else:
            # Cola en memoria
            self._queue.append(task_data)
        
        return task_id

    async def dequeue(self) -> Optional[dict]:
        """Desencola una tarea"""
        if self._queue:
            return self._queue.pop(0)
        return None

