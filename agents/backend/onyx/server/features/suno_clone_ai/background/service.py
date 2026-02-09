"""
Background Service - Servicio de tareas en background
"""

from typing import Any, Dict, Optional
from .base import BaseTask
from .queue import TaskQueue
from .worker import Worker
from redis.service import RedisService
from db.service import DatabaseService
from tracing.service import TracingService


class BackgroundService:
    """Servicio para gestionar tareas en background"""

    def __init__(
        self,
        redis_service: Optional[RedisService] = None,
        db_service: Optional[DatabaseService] = None,
        tracing_service: Optional[TracingService] = None
    ):
        """Inicializa el servicio de background"""
        self.queue = TaskQueue(redis_service)
        self.worker = Worker(self.queue, tracing_service)
        self.db_service = db_service
        self.tracing_service = tracing_service

    async def enqueue_task(self, task: BaseTask, *args, **kwargs) -> str:
        """Encola una tarea"""
        return await self.queue.enqueue(task, *args, **kwargs)

    def start_worker(self) -> None:
        """Inicia el worker"""
        self.worker.start()

