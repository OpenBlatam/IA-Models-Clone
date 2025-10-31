"""
Async Workers Configuration for Background Task Processing
Using Celery for distributed task execution
Optimized for microservices and serverless environments
"""

import os
import logging
from typing import Any, Dict, Optional
from datetime import timedelta

try:
    from celery import Celery
    from celery.schedules import crontab
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logging.warning("Celery not available. Install with: pip install celery")

from config import settings

logger = logging.getLogger(__name__)


class CeleryConfig:
    """Celery configuration optimized for microservices"""
    
    # Broker settings (RabbitMQ, Redis, etc.)
    broker_url = os.getenv(
        'CELERY_BROKER_URL',
        settings.redis_url.replace('redis://', 'redis://') if hasattr(settings, 'redis_url') else 'redis://localhost:6379/0'
    )
    
    # Result backend
    result_backend = os.getenv(
        'CELERY_RESULT_BACKEND',
        settings.redis_url.replace('redis://', 'redis://') if hasattr(settings, 'redis_url') else 'redis://localhost:6379/0'
    )
    
    # Task settings
    task_serializer = 'json'
    accept_content = ['json']
    result_serializer = 'json'
    timezone = 'UTC'
    enable_utc = True
    
    # Optimization for serverless
    task_always_eager = os.getenv('CELERY_ALWAYS_EAGER', 'false').lower() == 'true'  # For testing
    task_eager_propagates = True
    
    # Performance settings
    worker_prefetch_multiplier = 4
    worker_max_tasks_per_child = 1000  # Prevent memory leaks
    task_acks_late = True  # Acknowledge after task completion
    worker_disable_rate_limits = False
    
    # Result expiration
    result_expires = 3600  # 1 hour
    
    # Task routing
    task_routes = {
        'content_redundancy_detector.tasks.analyze_content': {'queue': 'analysis'},
        'content_redundancy_detector.tasks.batch_process': {'queue': 'batch'},
        'content_redundancy_detector.tasks.export_data': {'queue': 'export'},
        'content_redundancy_detector.tasks.send_webhook': {'queue': 'webhooks'},
    }
    
    # Task time limits
    task_time_limit = 300  # 5 minutes hard limit
    task_soft_time_limit = 240  # 4 minutes soft limit
    
    # Beat schedule for periodic tasks
    beat_schedule = {
        'cleanup-old-batches': {
            'task': 'content_redundancy_detector.tasks.cleanup_old_batches',
            'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        'generate-daily-report': {
            'task': 'content_redundancy_detector.tasks.generate_daily_report',
            'schedule': crontab(hour=0, minute=0),  # Daily at midnight
        },
        'health-check': {
            'task': 'content_redundancy_detector.tasks.health_check',
            'schedule': 60.0,  # Every minute
        },
    }


if CELERY_AVAILABLE:
    # Create Celery app
    celery_app = Celery(
        'content_redundancy_detector',
        broker=CeleryConfig.broker_url,
        backend=CeleryConfig.result_backend
    )
    celery_app.config_from_object(CeleryConfig)
    
    logger.info("Celery worker initialized")
else:
    celery_app = None


def get_celery_app() -> Optional[Celery]:
    """Get Celery app instance"""
    return celery_app


def create_task(task_name: str, *args, **kwargs) -> Optional[Any]:
    """
    Create and dispatch a Celery task
    
    Args:
        task_name: Name of the task
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        AsyncResult or None if Celery not available
    """
    if not CELERY_AVAILABLE or not celery_app:
        logger.warning("Celery not available, executing task synchronously")
        return None
    
    try:
        task = celery_app.signature(task_name, args=args, kwargs=kwargs)
        result = task.apply_async()
        return result
    except Exception as e:
        logger.error(f"Failed to create task {task_name}: {e}")
        return None


# Example task definitions (create in tasks.py)
"""
from async_workers import celery_app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@celery_app.task(name='content_redundancy_detector.tasks.analyze_content')
def analyze_content_task(content: str, threshold: float = 0.8):
    '''Background task for content analysis'''
    from services import analyze_content
    return analyze_content(content)

@celery_app.task(name='content_redundancy_detector.tasks.batch_process')
def batch_process_task(batch_id: str, jobs: list):
    '''Background task for batch processing'''
    from batch_processor import process_batch
    return process_batch(batch_id, jobs)

@celery_app.task(name='content_redundancy_detector.tasks.cleanup_old_batches')
def cleanup_old_batches():
    '''Periodic task to cleanup old batches'''
    from batch_processor import cleanup_old_batches
    cleanup_old_batches()
    return "Cleanup completed"
"""






