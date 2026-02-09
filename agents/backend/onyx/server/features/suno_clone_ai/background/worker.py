"""
Worker - Workers para procesar tareas
"""

from typing import Optional
import asyncio
from .queue import TaskQueue
from tracing.service import TracingService


class Worker:
    """Worker para procesar tareas en background"""

    def __init__(
        self,
        queue: TaskQueue,
        tracing_service: Optional[TracingService] = None
    ):
        """Inicializa el worker"""
        self.queue = queue
        self.tracing_service = tracing_service
        self._running = False

    def start(self) -> None:
        """Inicia el worker"""
        self._running = True
        asyncio.create_task(self._process_tasks())

    async def _process_tasks(self) -> None:
        """Procesa tareas de la cola"""
        while self._running:
            task_data = await self.queue.dequeue()
            if task_data:
                try:
                    task = task_data["task"]
                    await task.execute(*task_data["args"], **task_data["kwargs"])
                except Exception as e:
                    if self.tracing_service:
                        self.tracing_service.log("error", f"Task failed: {e}")

