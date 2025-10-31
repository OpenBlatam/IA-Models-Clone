"""
PDF Variantes - Background Workers
Base worker for async task processing (Celery/RQ compatible)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """Base worker class for background tasks"""
    
    def __init__(self, worker_name: str):
        self.worker_name = worker_name
        self.logger = logging.getLogger(f"{__name__}.{worker_name}")
        self._running = False
    
    @abstractmethod
    async def process_task(self, task_data: Dict[str, Any]) -> Any:
        """Process a task - must be implemented by subclasses"""
        pass
    
    async def execute(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with error handling and logging"""
        start_time = datetime.utcnow()
        self.logger.info(f"Starting task {task_id} of type {self.worker_name}")
        
        try:
            result = await self.process_task(task_data)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Task {task_id} completed in {duration:.2f}s")
            
            return {
                "task_id": task_id,
                "status": "success",
                "result": result,
                "duration": duration,
                "completed_at": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Task {task_id} failed after {duration:.2f}s: {e}", exc_info=True)
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "duration": duration,
                "failed_at": datetime.utcnow().isoformat()
            }


# Celery integration helper (optional)
try:
    from celery import Celery, Task
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logger.warning("Celery not available. Install with: pip install celery")


def create_celery_app(broker_url: str = "redis://localhost:6379/0") -> Optional[Celery]:
    """Create Celery app for background tasks"""
    if not CELERY_AVAILABLE:
        logger.warning("Celery not available")
        return None
    
    celery_app = Celery(
        "pdf_variantes",
        broker=broker_url,
        backend=broker_url
    )
    
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,  # 30 minutes
        task_soft_time_limit=25 * 60,  # 25 minutes
    )
    
    return celery_app


# RQ integration helper (optional)
try:
    from rq import Queue, Worker
    from redis import Redis
    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False
    logger.warning("RQ not available. Install with: pip install rq redis")


def create_rq_queue(redis_url: str = "redis://localhost:6379/0", queue_name: str = "default") -> Optional[Queue]:
    """Create RQ queue for background tasks"""
    if not RQ_AVAILABLE:
        logger.warning("RQ not available")
        return None
    
    redis_conn = Redis.from_url(redis_url)
    queue = Queue(queue_name, connection=redis_conn)
    return queue






