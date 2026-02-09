"""
Celery application configuration for async task processing
"""
import os
from celery import Celery
from kombu import Queue, Exchange

# Create Celery instance
celery_app = Celery(
    "faceless_video_ai",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
    include=["services.video_orchestrator", "services.batch_processor"],
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
        "services.video_orchestrator.*": {"queue": "video_generation"},
        "services.batch_processor.*": {"queue": "batch_processing"},
    },
    
    # Queue configuration
    task_queues=(
        Queue("video_generation", Exchange("video_generation"), routing_key="video_generation"),
        Queue("batch_processing", Exchange("batch_processing"), routing_key="batch_processing"),
        Queue("default", Exchange("default"), routing_key="default"),
    ),
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 minutes soft limit
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
)

# Optional: Configure periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-old-videos": {
        "task": "services.video_orchestrator.cleanup_old_videos",
        "schedule": 3600.0,  # Every hour
    },
    "update-video-status": {
        "task": "services.video_orchestrator.update_video_statuses",
        "schedule": 300.0,  # Every 5 minutes
    },
}

if __name__ == "__main__":
    celery_app.start()




