"""
API Schemas
===========

Esquemas Pydantic para la API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class TaskCreateRequest(BaseModel):
    """Request para crear una tarea."""
    repository: str = Field(..., description="Repositorio en formato owner/repo")
    instruction: str = Field(..., description="Instrucción a ejecutar")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata adicional")


class TaskResponse(BaseModel):
    """Response de una tarea."""
    id: str
    repository: str
    instruction: str
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response de lista de tareas."""
    tasks: List[TaskResponse]
    status: Dict[str, int]


class AgentStatusResponse(BaseModel):
    """Response del estado del agente."""
    status: str
    running_tasks: int
    queue: Dict[str, int]
    timestamp: str


class AgentControlRequest(BaseModel):
    """Request para controlar el agente."""
    action: str = Field(..., description="Acción: start, stop, pause, resume")
