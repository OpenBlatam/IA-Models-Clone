"""
MCP Request Queue - Cola de requests
=====================================

Cola de requests para manejar picos de tráfico de forma eficiente.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Prioridad de requests"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QueuedRequest:
    """Request en cola"""
    request_id: str
    priority: RequestPriority
    handler: Callable
    args: tuple
    kwargs: dict
    created_at: datetime
    timeout: Optional[float] = None


class RequestQueue:
    """Cola de requests con prioridades"""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_workers: int = 10,
        default_timeout: float = 300.0
    ):
        self.max_size = max_size
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self._workers: list = []
        self._running = False
        self._processed_count = 0
        self._failed_count = 0
        self._start_time = datetime.now()
    
    async def start(self):
        """Iniciar workers"""
        if self._running:
            return
        
        self._running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Request queue started with {self.max_workers} workers")
    
    async def stop(self):
        """Detener workers"""
        self._running = False
        
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Request queue stopped")
    
    async def enqueue(
        self,
        handler: Callable,
        priority: RequestPriority = RequestPriority.NORMAL,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Agregar request a la cola"""
        if self._queue.qsize() >= self.max_size:
            raise RuntimeError(f"Queue is full (max_size={self.max_size})")
        
        request_id = f"req_{time.time()}_{id(handler)}"
        queued_request = QueuedRequest(
            request_id=request_id,
            priority=priority,
            handler=handler,
            args=args,
            kwargs=kwargs,
            created_at=datetime.now(),
            timeout=timeout or self.default_timeout
        )
        
        priority_value = -priority.value
        await self._queue.put((priority_value, queued_request))
        
        return request_id
    
    async def _worker(self, worker_id: str):
        """Worker que procesa requests"""
        while self._running:
            try:
                priority_value, queued_request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            
            try:
                elapsed = (datetime.now() - queued_request.created_at).total_seconds()
                if queued_request.timeout and elapsed > queued_request.timeout:
                    logger.warning(f"Request {queued_request.request_id} timed out")
                    self._failed_count += 1
                    continue
                
                if asyncio.iscoroutinefunction(queued_request.handler):
                    await queued_request.handler(*queued_request.args, **queued_request.kwargs)
                else:
                    queued_request.handler(*queued_request.args, **queued_request.kwargs)
                
                self._processed_count += 1
            except Exception as e:
                logger.error(f"Error processing request {queued_request.request_id}: {e}", exc_info=True)
                self._failed_count += 1
            finally:
                self._queue.task_done()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la cola"""
        uptime = (datetime.now() - self._start_time).total_seconds()
        return {
            "queue_size": self._queue.qsize(),
            "max_size": self.max_size,
            "workers": len(self._workers),
            "processed": self._processed_count,
            "failed": self._failed_count,
            "success_rate": round(
                self._processed_count / (self._processed_count + self._failed_count) if (self._processed_count + self._failed_count) > 0 else 0,
                4
            ),
            "requests_per_second": round(self._processed_count / uptime, 2) if uptime > 0 else 0,
            "uptime_seconds": int(uptime)
        }

