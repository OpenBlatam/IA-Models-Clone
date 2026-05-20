"""
Celery Worker - Workers asíncronos para tareas en background
============================================================

Sistema de workers distribuidos usando Celery para procesar
tareas pesadas sin bloquear la API.
"""

import os
import logging
from typing import Dict, Any, Optional
from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure

logger = logging.getLogger(__name__)

# Configuración Celery
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Crear app Celery
celery_app = Celery(
    "cursor_agent_24_7",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["core.celery_tasks"]
)

# Configuración
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutos
    task_soft_time_limit=240,  # 4 minutos
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,  # 1 hora
    task_routes={
        "core.celery_tasks.process_task": {"queue": "tasks"},
        "core.celery_tasks.process_heavy_task": {"queue": "heavy"},
        "core.celery_tasks.send_notification": {"queue": "notifications"},
    },
    task_default_queue="default",
    task_default_exchange="tasks",
    task_default_exchange_type="direct",
    task_default_routing_key="default",
)


# Signals para logging y métricas
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handler antes de ejecutar tarea."""
    logger.info(f"Starting task {task.name} with id {task_id}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handler después de ejecutar tarea."""
    logger.info(f"Completed task {task.name} with id {task_id}, state: {state}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handler cuando falla una tarea."""
    logger.error(f"Task {sender.name} failed with id {task_id}: {exception}")


def get_celery_app() -> Celery:
    """Obtener instancia de Celery."""
    return celery_app


# Funciones de utilidad
def enqueue_task(task_name: str, *args, **kwargs) -> str:
    """
    Encolar tarea en Celery.
    
    Args:
        task_name: Nombre de la tarea.
        *args: Argumentos posicionales.
        **kwargs: Argumentos con nombre.
    
    Returns:
        ID de la tarea.
    """
    result = celery_app.send_task(task_name, args=args, kwargs=kwargs)
    return result.id


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Obtener estado de una tarea.
    
    Args:
        task_id: ID de la tarea.
    
    Returns:
        Diccionario con estado de la tarea.
    """
    result = celery_app.AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
        "info": result.info if result.info else None
    }


def get_task_result(task_id: str, timeout: Optional[float] = None) -> Any:
    """
    Obtener resultado de una tarea (bloqueante).
    
    Args:
        task_id: ID de la tarea.
        timeout: Timeout en segundos (None = sin timeout).
    
    Returns:
        Resultado de la tarea.
    """
    result = celery_app.AsyncResult(task_id)
    return result.get(timeout=timeout)




