from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
import json
import traceback
from typing import Dict, Any, List, Optional, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import signal
import os
import sys
from pathlib import Path
import structlog
from prometheus_client import Counter, Histogram, Gauge
import psutil
import redis.asyncio as redis
from celery import Celery
from celery.utils.log import get_task_logger
from production_config import get_config, ProductionConfig
        from integration_master import IntegrationMaster
        from integration_master import IntegrationMaster
        from integration_master import IntegrationMaster
        from integration_master import IntegrationMaster
        from integration_master import IntegrationMaster
        from integration_master import IntegrationMaster
        from integration_master import IntegrationMaster
from typing import Any, List, Dict, Optional
"""
Production Worker System
========================

Background task processing system for production deployment.
Handles queue management, task distribution, error handling, and monitoring.
"""


# Production imports

# Local imports

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Task definition"""
    id: str
    name: str
    func: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes
    worker_id: Optional[str] = None

class WorkerMetrics:
    """Worker metrics for monitoring"""
    
    def __init__(self) -> Any:
        self.tasks_processed = Counter('worker_tasks_processed_total', 'Total tasks processed', ['worker_id', 'status'])
        self.task_duration = Histogram('worker_task_duration_seconds', 'Task duration', ['worker_id', 'task_name'])
        self.queue_size = Gauge('worker_queue_size', 'Current queue size', ['queue_name'])
        self.active_workers = Gauge('worker_active_workers', 'Active workers')
        self.memory_usage = Gauge('worker_memory_bytes', 'Worker memory usage')
        self.cpu_usage = Gauge('worker_cpu_percent', 'Worker CPU usage')

class TaskQueue:
    """Task queue implementation"""
    
    def __init__(self, name: str, redis_client: redis.Redis):
        
    """__init__ function."""
self.name = name
        self.redis = redis_client
        self.logger = structlog.get_logger()
    
    async def enqueue(self, task: Task) -> bool:
        """Enqueue a task"""
        try:
            task_data = {
                'id': task.id,
                'name': task.name,
                'func': task.func,
                'args': task.args,
                'kwargs': task.kwargs,
                'priority': task.priority.value,
                'status': task.status.value,
                'created_at': task.created_at,
                'retries': task.retries,
                'max_retries': task.max_retries,
                'timeout': task.timeout
            }
            
            await self.redis.lpush(f"queue:{self.name}", json.dumps(task_data))
            self.logger.info(f"Task {task.id} enqueued in {self.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enqueue task {task.id}: {e}")
            return False
    
    async def dequeue(self) -> Optional[Task]:
        """Dequeue a task"""
        try:
            # Get task with highest priority
            task_data = await self.redis.brpop(f"queue:{self.name}", timeout=1)
            if task_data:
                task_dict = json.loads(task_data[1])
                return Task(
                    id=task_dict['id'],
                    name=task_dict['name'],
                    func=task_dict['func'],
                    args=task_dict['args'],
                    kwargs=task_dict['kwargs'],
                    priority=TaskPriority(task_dict['priority']),
                    status=TaskStatus(task_dict['status']),
                    created_at=task_dict['created_at'],
                    retries=task_dict['retries'],
                    max_retries=task_dict['max_retries'],
                    timeout=task_dict['timeout']
                )
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to dequeue task: {e}")
            return None
    
    async def get_queue_size(self) -> int:
        """Get current queue size"""
        try:
            return await self.redis.llen(f"queue:{self.name}")
        except Exception as e:
            self.logger.error(f"Failed to get queue size: {e}")
            return 0

class Worker:
    """Worker implementation"""
    
    def __init__(self, worker_id: str, task_queue: TaskQueue, metrics: WorkerMetrics):
        
    """__init__ function."""
self.worker_id = worker_id
        self.task_queue = task_queue
        self.metrics = metrics
        self.logger = structlog.get_logger()
        self.is_running = False
        self.current_task = None
        self.task_handlers = {}
        
        # Performance tracking
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.start_time = time.time()
    
    def register_handler(self, task_name: str, handler: Callable):
        """Register a task handler"""
        self.task_handlers[task_name] = handler
        self.logger.info(f"Registered handler for task: {task_name}")
    
    async def start(self) -> Any:
        """Start the worker"""
        self.is_running = True
        self.metrics.active_workers.inc()
        self.logger.info(f"Worker {self.worker_id} started")
        
        while self.is_running:
            try:
                # Get task from queue
                task = await self.task_queue.dequeue()
                if task:
                    await self.process_task(task)
                else:
                    # No tasks available, wait a bit
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def stop(self) -> Any:
        """Stop the worker"""
        self.is_running = False
        self.metrics.active_workers.dec()
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    async def process_task(self, task: Task):
        """Process a single task"""
        start_time = time.time()
        self.current_task = task
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.worker_id = self.worker_id
        
        self.logger.info(f"Processing task {task.id} ({task.name})")
        
        try:
            # Check if handler exists
            if task.func not in self.task_handlers:
                raise ValueError(f"No handler registered for task: {task.func}")
            
            # Execute task
            handler = self.task_handlers[task.func]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*task.args, **task.kwargs)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, handler, *task.args, **task.kwargs)
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            duration = time.time() - start_time
            self.metrics.tasks_processed.labels(worker_id=self.worker_id, status='completed').inc()
            self.metrics.task_duration.labels(worker_id=self.worker_id, task_name=task.name).observe(duration)
            
            self.tasks_processed += 1
            self.logger.info(f"Task {task.id} completed in {duration:.2f}s")
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            duration = time.time() - start_time
            self.metrics.tasks_processed.labels(worker_id=self.worker_id, status='failed').inc()
            
            self.tasks_failed += 1
            self.logger.error(f"Task {task.id} failed: {e}")
            self.logger.error(traceback.format_exc())
            
            # Retry logic
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                await self.task_queue.enqueue(task)
                self.logger.info(f"Retrying task {task.id} (attempt {task.retries})")
        
        finally:
            self.current_task = None
            self._update_metrics()
    
    def _update_metrics(self) -> Any:
        """Update worker metrics"""
        try:
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.memory_usage.set(memory_info.rss)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.metrics.cpu_usage.set(cpu_percent)
            
        except Exception as e:
            self.logger.warning(f"Failed to update metrics: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        uptime = time.time() - self.start_time
        return {
            'worker_id': self.worker_id,
            'is_running': self.is_running,
            'uptime': uptime,
            'tasks_processed': self.tasks_processed,
            'tasks_failed': self.tasks_failed,
            'success_rate': (self.tasks_processed / (self.tasks_processed + self.tasks_failed)) if (self.tasks_processed + self.tasks_failed) > 0 else 0,
            'current_task': self.current_task.id if self.current_task else None,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.Process().cpu_percent()
        }

class WorkerPool:
    """Worker pool management"""
    
    def __init__(self, config: ProductionConfig, redis_client: redis.Redis):
        
    """__init__ function."""
self.config = config
        self.redis = redis_client
        self.logger = structlog.get_logger()
        self.workers = []
        self.metrics = WorkerMetrics()
        self.task_queue = TaskQueue("default", redis_client)
        self.is_running = False
        
        # Register default task handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> Any:
        """Register default task handlers"""
        # Text processing tasks
        self.register_handler("process_text", self._process_text_handler)
        self.register_handler("process_batch_text", self._process_batch_text_handler)
        
        # Image processing tasks
        self.register_handler("process_image", self._process_image_handler)
        self.register_handler("process_batch_images", self._process_batch_images_handler)
        
        # Vector search tasks
        self.register_handler("vector_search", self._vector_search_handler)
        
        # Optimization tasks
        self.register_handler("optimize_performance", self._optimize_performance_handler)
        
        # System tasks
        self.register_handler("health_check", self._health_check_handler)
        self.register_handler("cleanup", self._cleanup_handler)
    
    def register_handler(self, task_name: str, handler: Callable):
        """Register a task handler for all workers"""
        for worker in self.workers:
            worker.register_handler(task_name, handler)
    
    async def start(self, num_workers: Optional[int] = None):
        """Start the worker pool"""
        if num_workers is None:
            num_workers = self.config.performance.max_workers
        
        self.logger.info(f"Starting worker pool with {num_workers} workers")
        
        # Create workers
        for i in range(num_workers):
            worker_id = f"worker-{i+1}"
            worker = Worker(worker_id, self.task_queue, self.metrics)
            self.workers.append(worker)
            
            # Register handlers
            for task_name, handler in self._get_default_handlers().items():
                worker.register_handler(task_name, handler)
        
        # Start workers
        self.is_running = True
        worker_tasks = [asyncio.create_task(worker.start()) for worker in self.workers]
        
        # Monitor workers
        await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    async def stop(self) -> Any:
        """Stop the worker pool"""
        self.logger.info("Stopping worker pool")
        self.is_running = False
        
        # Stop all workers
        stop_tasks = [worker.stop() for worker in self.workers]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.workers.clear()
    
    async def enqueue_task(self, task: Task) -> bool:
        """Enqueue a task in the pool"""
        return await self.task_queue.enqueue(task)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics"""
        worker_stats = [worker.get_stats() for worker in self.workers]
        queue_size = asyncio.create_task(self.task_queue.get_queue_size())
        
        return {
            'is_running': self.is_running,
            'num_workers': len(self.workers),
            'workers': worker_stats,
            'queue_size': queue_size,
            'total_tasks_processed': sum(w['tasks_processed'] for w in worker_stats),
            'total_tasks_failed': sum(w['tasks_failed'] for w in worker_stats)
        }
    
    def _get_default_handlers(self) -> Dict[str, Callable]:
        """Get default task handlers"""
        return {
            "process_text": self._process_text_handler,
            "process_batch_text": self._process_batch_text_handler,
            "process_image": self._process_image_handler,
            "process_batch_images": self._process_batch_images_handler,
            "vector_search": self._vector_search_handler,
            "optimize_performance": self._optimize_performance_handler,
            "health_check": self._health_check_handler,
            "cleanup": self._cleanup_handler
        }
    
    # Task handlers
    async def _process_text_handler(self, text: str, operations: List[str] = None) -> Dict[str, Any]:
        """Process text task handler"""
        
        integration = IntegrationMaster()
        operations = operations or ["statistics", "sentiment", "keywords"]
        
        result = await integration.process_text(text, operations)
        return result
    
    async def _process_batch_text_handler(self, texts: List[str], operations: List[str] = None) -> List[Dict[str, Any]]:
        """Process batch text task handler"""
        
        integration = IntegrationMaster()
        operations = operations or ["statistics", "sentiment"]
        
        async def processor(text) -> Any:
            return await integration.process_text(text, operations)
        
        results = await integration.batch_process(texts, processor, self.config.performance.batch_size)
        return results
    
    async def _process_image_handler(self, image_path: str, operations: List[str] = None) -> Dict[str, Any]:
        """Process image task handler"""
        
        integration = IntegrationMaster()
        operations = operations or ["properties", "face_detection"]
        
        result = await integration.process_image(image_path, operations)
        return result
    
    async def _process_batch_images_handler(self, image_paths: List[str], operations: List[str] = None) -> List[Dict[str, Any]]:
        """Process batch images task handler"""
        
        integration = IntegrationMaster()
        operations = operations or ["properties"]
        
        async def processor(image_path) -> Any:
            return await integration.process_image(image_path, operations)
        
        results = await integration.batch_process(image_paths, processor, self.config.performance.batch_size)
        return results
    
    async def _vector_search_handler(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector search task handler"""
        
        integration = IntegrationMaster()
        results = await integration.vector_search(query, top_k)
        return results
    
    async def _optimize_performance_handler(self, task_type: str, **kwargs) -> Dict[str, Any]:
        """Performance optimization task handler"""
        
        integration = IntegrationMaster()
        results = await integration.optimize_performance(task_type, **kwargs)
        return results
    
    async def _health_check_handler(self) -> Dict[str, Any]:
        """Health check task handler"""
        
        integration = IntegrationMaster()
        health = await integration.health_check()
        return health
    
    async def _cleanup_handler(self) -> Dict[str, Any]:
        """Cleanup task handler"""
        # Cleanup temporary files
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()
        
        return {"cleaned_files": True}

class CeleryWorker:
    """Celery worker implementation for distributed task processing"""
    
    def __init__(self, config: ProductionConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger()
        
        # Initialize Celery
        self.celery = Celery(
            'notebooklm_ai_worker',
            broker=self.config.get_redis_url(),
            backend=self.config.get_redis_url(),
            include=['production_worker']
        )
        
        # Configure Celery
        self.celery.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=30 * 60,  # 30 minutes
            task_soft_time_limit=25 * 60,  # 25 minutes
            worker_prefetch_multiplier=1,
            worker_max_tasks_per_child=1000,
            worker_max_memory_per_child=200000,  # 200MB
        )
    
    def start(self) -> Any:
        """Start the Celery worker"""
        self.logger.info("Starting Celery worker")
        
        # Start worker
        self.celery.worker_main([
            'worker',
            '--loglevel=info',
            '--concurrency=4',
            '--pool=prefork'
        ])

# Global worker pool instance
worker_pool = None

async def get_worker_pool() -> WorkerPool:
    """Get or create the global worker pool instance"""
    global worker_pool
    if worker_pool is None:
        config = get_config()
        redis_client = redis.from_url(config.get_redis_url())
        worker_pool = WorkerPool(config, redis_client)
    return worker_pool

async def main():
    """Main function for worker deployment"""
    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger()
    logger.info("🚀 Starting Production Worker System")
    
    try:
        # Get configuration
        config = get_config()
        
        # Validate configuration
        if not config.validate():
            logger.error("Invalid configuration")
            sys.exit(1)
        
        # Create directories
        config.create_directories()
        
        # Get worker pool
        worker_pool = await get_worker_pool()
        
        # Start worker pool
        await worker_pool.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Worker system error: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup
        if worker_pool:
            await worker_pool.stop()
        logger.info("✅ Worker system shutdown completed")

match __name__:
    case "__main__":
    asyncio.run(main()) 