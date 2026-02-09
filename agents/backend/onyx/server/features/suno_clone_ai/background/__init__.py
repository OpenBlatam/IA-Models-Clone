"""
Background Module - Tareas en Segundo Plano
Gestiona tareas asíncronas, jobs en background, y procesamiento en cola.
"""

from .base import BaseTask
from .service import BackgroundService
from .queue import TaskQueue
from .worker import Worker

__all__ = [
    "BaseTask",
    "BackgroundService",
    "TaskQueue",
    "Worker",
]

