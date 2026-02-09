"""
Celery Configuration for Async Workers
=======================================

Handles background tasks:
- Trajectory optimization
- Model inference
- Data processing
- Report generation
"""

import os
from celery import Celery
from celery.schedules import crontab
from kombu import Queue, Exchange

# Redis URL for broker and result backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RESULT_BACKEND_URL = os.getenv("REDIS_RESULT_BACKEND", "redis://localhost:6379/1")

# Create Celery app
celery_app = Celery(
    "robot_movement_ai",
    broker=REDIS_URL,
    backend=RESULT_BACKEND_URL,
    include=[
        "aws.workers.tasks.trajectory_tasks",
        "aws.workers.tasks.model_tasks",
        "aws.workers.tasks.data_tasks",
        "aws.workers.tasks.report_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "aws.workers.tasks.trajectory_tasks.*": {"queue": "trajectory"},
        "aws.workers.tasks.model_tasks.*": {"queue": "model"},
        "aws.workers.tasks.data_tasks.*": {"queue": "data"},
        "aws.workers.tasks.report_tasks.*": {"queue": "reports"},
    },
    
    # Queue configuration
    task_queues=(
        Queue("trajectory", Exchange("trajectory"), routing_key="trajectory"),
        Queue("model", Exchange("model"), routing_key="model"),
        Queue("data", Exchange("data"), routing_key="data"),
        Queue("reports", Exchange("reports"), routing_key="reports"),
        Queue("default", Exchange("default"), routing_key="default"),
    ),
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Result settings
    result_expires=3600,  # 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Beat schedule (periodic tasks)
    beat_schedule={
        "cleanup-old-tasks": {
            "task": "aws.workers.tasks.data_tasks.cleanup_old_tasks",
            "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        "generate-daily-report": {
            "task": "aws.workers.tasks.report_tasks.generate_daily_report",
            "schedule": crontab(hour=0, minute=0),  # Daily at midnight
        },
        "health-check": {
            "task": "aws.workers.tasks.data_tasks.health_check",
            "schedule": 60.0,  # Every minute
        },
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Task decorators
def task_with_retry(*args, **kwargs):
    """Task decorator with automatic retry."""
    kwargs.setdefault("autoretry_for", (Exception,))
    kwargs.setdefault("retry_kwargs", {"max_retries": 3, "countdown": 60})
    return celery_app.task(*args, **kwargs)















