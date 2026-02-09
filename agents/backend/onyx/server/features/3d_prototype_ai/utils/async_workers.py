"""
Async Workers - Sistema de workers asíncronos para tareas en background
========================================================================

Soporta:
- Celery para tareas distribuidas
- RQ (Redis Queue) para tareas simples
- AsyncIO workers nativos
"""

import asyncio
import logging
from typing import Callable, Any, Optional, Dict
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class WorkerType(Enum):
    """Tipo de worker"""
    CELERY = "celery"
    RQ = "rq"
    ASYNC = "async"


class TaskStatus(Enum):
    """Estado de una tarea"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"


class AsyncWorker:
    """Worker asíncrono nativo usando asyncio"""
    
    def __init__(self, max_workers: int = 5, queue_size: int = 1000):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.workers: list = []
        self.running = False
        self.tasks: Dict[str, Dict] = {}
    
    async def start(self):
        """Inicia los workers"""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
        logger.info(f"Started {self.max_workers} async workers")
    
    async def stop(self):
        """Detiene los workers"""
        self.running = False
        
        # Esperar a que terminen las tareas pendientes
        await self.task_queue.join()
        
        # Cancelar workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Async workers stopped")
    
    async def enqueue(self, func: Callable, *args, task_id: Optional[str] = None, **kwargs) -> str:
        """Encola una tarea"""
        if task_id is None:
            import uuid
            task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "status": TaskStatus.PENDING,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.tasks[task_id] = task
        
        try:
            await asyncio.wait_for(self.task_queue.put(task), timeout=5.0)
            logger.info(f"Task {task_id} enqueued")
            return task_id
        except asyncio.TimeoutError:
            raise Exception("Task queue is full")
    
    async def _worker(self, worker_name: str):
        """Worker que procesa tareas"""
        logger.info(f"{worker_name} started")
        
        while self.running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                task_id = task["id"]
                self.tasks[task_id]["status"] = TaskStatus.RUNNING
                self.tasks[task_id]["started_at"] = datetime.utcnow().isoformat()
                
                try:
                    # Ejecutar tarea
                    if asyncio.iscoroutinefunction(task["func"]):
                        result = await task["func"](*task["args"], **task["kwargs"])
                    else:
                        result = task["func"](*task["args"], **task["kwargs"])
                    
                    self.tasks[task_id]["status"] = TaskStatus.SUCCESS
                    self.tasks[task_id]["result"] = result
                    self.tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()
                    
                    logger.info(f"Task {task_id} completed successfully")
                    
                except Exception as e:
                    self.tasks[task_id]["status"] = TaskStatus.FAILURE
                    self.tasks[task_id]["error"] = str(e)
                    self.tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()
                    
                    logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                
                finally:
                    self.task_queue.task_done()
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"{worker_name} error: {e}", exc_info=True)
        
        logger.info(f"{worker_name} stopped")
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Obtiene el estado de una tarea"""
        return self.tasks.get(task_id)
    
    def get_task_result(self, task_id: str) -> Any:
        """Obtiene el resultado de una tarea completada"""
        task = self.tasks.get(task_id)
        if task and task["status"] == TaskStatus.SUCCESS:
            return task.get("result")
        return None


class CeleryWorker:
    """Wrapper para Celery workers"""
    
    def __init__(self, broker_url: str = "redis://localhost:6379/0",
                 backend_url: str = "redis://localhost:6379/0"):
        self.broker_url = broker_url
        self.backend_url = backend_url
        self.celery_app = None
        self._setup_celery()
    
    def _setup_celery(self):
        """Configura Celery"""
        try:
            from celery import Celery
            
            self.celery_app = Celery(
                "3d_prototype_ai",
                broker=self.broker_url,
                backend=self.backend_url
            )
            
            self.celery_app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
                task_track_started=True,
                task_time_limit=30 * 60,  # 30 minutos
                task_soft_time_limit=25 * 60,  # 25 minutos
                worker_prefetch_multiplier=4,
                worker_max_tasks_per_child=1000
            )
            
            logger.info("Celery configured successfully")
            
        except ImportError:
            logger.warning("Celery not available. Install with: pip install celery")
            self.celery_app = None
    
    def task(self, *args, **kwargs):
        """Decorator para crear tareas Celery"""
        if self.celery_app is None:
            raise RuntimeError("Celery not configured")
        return self.celery_app.task(*args, **kwargs)
    
    def enqueue(self, task_name: str, *args, **kwargs):
        """Encola una tarea Celery"""
        if self.celery_app is None:
            raise RuntimeError("Celery not configured")
        return self.celery_app.send_task(task_name, args=args, kwargs=kwargs)


class RQWorker:
    """Wrapper para RQ (Redis Queue) workers"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_conn = None
        self.queue = None
        self._setup_rq()
    
    def _setup_rq(self):
        """Configura RQ"""
        try:
            import redis
            from rq import Queue
            
            self.redis_conn = redis.from_url(self.redis_url)
            self.queue = Queue(connection=self.redis_conn)
            
            logger.info("RQ configured successfully")
            
        except ImportError:
            logger.warning("RQ not available. Install with: pip install rq")
            self.redis_conn = None
            self.queue = None
    
    def enqueue(self, func: Callable, *args, **kwargs):
        """Encola una tarea RQ"""
        if self.queue is None:
            raise RuntimeError("RQ not configured")
        return self.queue.enqueue(func, *args, **kwargs)


class WorkerManager:
    """Gestor de workers que soporta múltiples tipos"""
    
    def __init__(self, worker_type: WorkerType = WorkerType.ASYNC,
                 max_workers: int = 5,
                 broker_url: Optional[str] = None):
        self.worker_type = worker_type
        self.worker: Optional[Any] = None
        
        if worker_type == WorkerType.ASYNC:
            self.worker = AsyncWorker(max_workers=max_workers)
        elif worker_type == WorkerType.CELERY:
            broker = broker_url or "redis://localhost:6379/0"
            self.worker = CeleryWorker(broker_url=broker)
        elif worker_type == WorkerType.RQ:
            redis_url = broker_url or "redis://localhost:6379/0"
            self.worker = RQWorker(redis_url=redis_url)
    
    async def start(self):
        """Inicia el worker manager"""
        if isinstance(self.worker, AsyncWorker):
            await self.worker.start()
        logger.info(f"Worker manager started with type: {self.worker_type.value}")
    
    async def stop(self):
        """Detiene el worker manager"""
        if isinstance(self.worker, AsyncWorker):
            await self.worker.stop()
        logger.info("Worker manager stopped")
    
    async def enqueue_task(self, func: Callable, *args, task_id: Optional[str] = None, **kwargs):
        """Encola una tarea"""
        if isinstance(self.worker, AsyncWorker):
            return await self.worker.enqueue(func, *args, task_id=task_id, **kwargs)
        elif isinstance(self.worker, CeleryWorker):
            # Para Celery, necesitamos registrar la función primero
            task = self.worker.task()(func)
            return task.delay(*args, **kwargs)
        elif isinstance(self.worker, RQWorker):
            return self.worker.enqueue(func, *args, **kwargs)
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Obtiene el estado de una tarea"""
        if isinstance(self.worker, AsyncWorker):
            return self.worker.get_task_status(task_id)
        # Para Celery y RQ, necesitaríamos implementar tracking
        return None




