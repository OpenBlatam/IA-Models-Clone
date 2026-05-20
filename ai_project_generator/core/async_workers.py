"""
Async Workers - Soporte para workers asíncronos (Celery, RQ, ARQ)
=================================================================

Integración con workers asíncronos para procesamiento en background
siguiendo mejores prácticas de microservicios.
"""

import logging
from typing import Optional, Callable, Any, Dict
from enum import Enum
from functools import wraps

from .microservices_config import get_worker_config, WorkerBackend

logger = logging.getLogger(__name__)

# Celery
try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None
    AsyncResult = None

# RQ
try:
    from rq import Queue, Job
    from redis import Redis
    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False
    Queue = None
    Job = None

# ARQ
try:
    from arq import create_pool
    from arq.connections import RedisSettings
    ARQ_AVAILABLE = True
except ImportError:
    ARQ_AVAILABLE = False


class WorkerManager:
    """Manager para workers asíncronos"""
    
    def __init__(self, backend: Optional[WorkerBackend] = None):
        config = get_worker_config()
        self.backend = backend or WorkerBackend(config.get("backend", "celery"))
        self.broker_url = config.get("broker_url")
        self.result_backend = config.get("result_backend")
        
        self.celery_app: Optional[Celery] = None
        self.rq_queue: Optional[Queue] = None
        self.arq_pool = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Inicializa el backend de workers"""
        if self.backend == WorkerBackend.CELERY and CELERY_AVAILABLE:
            self._init_celery()
        elif self.backend == WorkerBackend.RQ and RQ_AVAILABLE:
            self._init_rq()
        elif self.backend == WorkerBackend.ARQ and ARQ_AVAILABLE:
            self._init_arq()
        else:
            logger.warning(f"Worker backend {self.backend} not available or not configured")
    
    def _init_celery(self):
        """Inicializa Celery"""
        try:
            self.celery_app = Celery(
                "ai_project_generator",
                broker=self.broker_url or "redis://localhost:6379/0",
                backend=self.result_backend or "redis://localhost:6379/0"
            )
            self.celery_app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
                task_track_started=True,
                task_time_limit=300,
                task_soft_time_limit=240,
            )
            logger.info("Celery worker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Celery: {e}")
    
    def _init_rq(self):
        """Inicializa RQ"""
        try:
            redis_conn = Redis.from_url(
                self.broker_url or "redis://localhost:6379/0"
            )
            self.rq_queue = Queue("default", connection=redis_conn)
            logger.info("RQ worker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RQ: {e}")
    
    def _init_arq(self):
        """Inicializa ARQ"""
        try:
            # ARQ se inicializa de forma diferente (async)
            logger.info("ARQ requires async initialization")
        except Exception as e:
            logger.error(f"Failed to initialize ARQ: {e}")
    
    def enqueue_task(
        self,
        task_func: Callable,
        *args,
        **kwargs
    ) -> Optional[str]:
        """
        Encola una tarea para ejecución asíncrona.
        
        Returns:
            Task ID o None si falla
        """
        if self.backend == WorkerBackend.CELERY and self.celery_app:
            try:
                task = self.celery_app.send_task(
                    task_func.__name__,
                    args=args,
                    kwargs=kwargs
                )
                return task.id
            except Exception as e:
                logger.error(f"Failed to enqueue Celery task: {e}")
                return None
        
        elif self.backend == WorkerBackend.RQ and self.rq_queue:
            try:
                job = self.rq_queue.enqueue(
                    task_func,
                    *args,
                    **kwargs
                )
                return job.id
            except Exception as e:
                logger.error(f"Failed to enqueue RQ task: {e}")
                return None
        
        else:
            logger.warning("No worker backend available")
            return None
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene resultado de una tarea"""
        if self.backend == WorkerBackend.CELERY and self.celery_app:
            try:
                result = AsyncResult(task_id, app=self.celery_app)
                if result.ready():
                    return {
                        "status": result.status,
                        "result": result.result,
                        "success": result.successful(),
                    }
                return {"status": result.status}
            except Exception as e:
                logger.error(f"Failed to get Celery task result: {e}")
                return None
        
        elif self.backend == WorkerBackend.RQ and self.rq_queue:
            try:
                job = Job.fetch(task_id, connection=self.rq_queue.connection)
                return {
                    "status": job.get_status(),
                    "result": job.result if job.is_finished else None,
                    "success": job.is_finished and not job.is_failed,
                }
            except Exception as e:
                logger.error(f"Failed to get RQ task result: {e}")
                return None
        
        return None


# Instancia global
_worker_manager: Optional[WorkerManager] = None


def get_worker_manager() -> WorkerManager:
    """Obtiene instancia global de worker manager"""
    global _worker_manager
    if _worker_manager is None:
        _worker_manager = WorkerManager()
    return _worker_manager


def task(backend: Optional[WorkerBackend] = None):
    """
    Decorator para registrar una función como tarea de worker.
    
    Usage:
        @task()
        def my_task(arg1, arg2):
            return result
    """
    def decorator(func: Callable):
        # Registrar en Celery si está disponible
        if CELERY_AVAILABLE:
            worker_mgr = get_worker_manager()
            if worker_mgr.celery_app:
                @worker_mgr.celery_app.task(name=func.__name__)
                @wraps(func)
                def celery_wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return celery_wrapper
        
        # Si no hay Celery, retornar función normal
        return func
    
    return decorator















