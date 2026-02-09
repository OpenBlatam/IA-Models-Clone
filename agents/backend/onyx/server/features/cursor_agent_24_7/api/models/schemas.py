"""
API Schemas - Modelos Pydantic
===============================

Modelos de datos para requests y responses de la API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class TaskRequest(BaseModel):
    """
    Request para agregar una tarea.
    
    Attributes:
        command: Comando a ejecutar (requerido).
    """
    command: str = Field(..., description="Comando a ejecutar", min_length=1)


class TaskResponse(BaseModel):
    """
    Response de una tarea.
    
    Attributes:
        task_id: ID único de la tarea.
        status: Estado de la tarea.
        message: Mensaje descriptivo.
    """
    task_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """
    Response del estado del agente.
    
    Attributes:
        status: Estado actual del agente.
        running: Si el agente está corriendo.
        tasks_total: Total de tareas.
        tasks_pending: Tareas pendientes.
        tasks_running: Tareas en ejecución.
        tasks_completed: Tareas completadas.
        tasks_failed: Tareas fallidas.
    """
    status: str
    running: bool
    tasks_total: int
    tasks_pending: int
    tasks_running: int
    tasks_completed: int
    tasks_failed: int


class TaskDetailResponse(BaseModel):
    """
    Response con detalles de una tarea.
    
    Attributes:
        id: ID de la tarea.
        command: Comando ejecutado.
        status: Estado actual.
        timestamp: Fecha y hora de creación.
        result: Resultado de la ejecución (si está disponible).
        error: Mensaje de error (si falló).
    """
    id: str
    command: str
    status: str
    timestamp: str
    result: Optional[str] = None
    error: Optional[str] = None


class TasksListResponse(BaseModel):
    """
    Response con lista de tareas.
    
    Attributes:
        tasks: Lista de tareas.
        total: Total de tareas.
    """
    tasks: List[TaskDetailResponse]
    total: int

