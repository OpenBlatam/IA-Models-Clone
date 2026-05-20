"""
Tipos TypeScript compartidos para frontend.
Estos tipos pueden ser exportados a TypeScript.
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal
from datetime import datetime


# Task Types
class TaskDict(TypedDict, total=False):
    """Tipo de tarea."""
    id: str
    repository_owner: str
    repository_name: str
    instruction: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error: Optional[str]


class TaskCreateRequest(TypedDict):
    """Request para crear tarea."""
    repository_owner: str
    repository_name: str
    instruction: str
    metadata: Optional[Dict[str, Any]]


class TaskResponse(TypedDict):
    """Respuesta de tarea."""
    task: TaskDict
    message: str


class TaskListResponse(TypedDict):
    """Respuesta de lista de tareas."""
    tasks: List[TaskDict]
    total: int
    limit: int
    offset: int


# Agent Types
class AgentStatus(TypedDict):
    """Estado del agente."""
    is_running: bool
    current_task_id: Optional[str]
    last_activity: Optional[str]
    metadata: Dict[str, Any]


class AgentStatusResponse(TypedDict):
    """Respuesta de estado del agente."""
    status: AgentStatus


# Repository Types
class RepositoryInfo(TypedDict, total=False):
    """Información de repositorio."""
    name: str
    full_name: str
    description: Optional[str]
    url: str
    default_branch: str
    language: Optional[str]
    stars: int
    forks: int
    is_private: bool
    created_at: Optional[str]
    updated_at: Optional[str]


class RepositoryListResponse(TypedDict):
    """Respuesta de lista de repositorios."""
    repositories: List[RepositoryInfo]
    total: int


# LLM Types
class LLMRequest(TypedDict, total=False):
    """Request para LLM."""
    prompt: str
    model: Optional[str]
    system_prompt: Optional[str]
    temperature: float
    max_tokens: Optional[int]


class LLMResponse(TypedDict, total=False):
    """Respuesta de LLM."""
    model: str
    content: str
    usage: Optional[Dict[str, Any]]
    finish_reason: Optional[str]
    error: Optional[str]
    latency_ms: Optional[float]


class CodeAnalysisRequest(TypedDict):
    """Request para análisis de código."""
    code: str
    language: Optional[str]
    analysis_type: Literal["general", "bugs", "performance", "security", "style"]


# Health Check Types
class ServiceStatus(TypedDict, total=False):
    """Estado de un servicio."""
    status: Literal["ok", "error", "warning", "disabled"]
    message: str


class HealthResponse(TypedDict):
    """Respuesta de health check."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    services: Dict[str, bool]
    details: Dict[str, ServiceStatus]


# Error Types
class APIError(TypedDict):
    """Error de API."""
    error: bool
    detail: str
    message: Optional[str]
    code: Optional[str]


# WebSocket Types (para futuras implementaciones)
class WebSocketMessage(TypedDict, total=False):
    """Mensaje WebSocket."""
    type: str
    data: Dict[str, Any]
    timestamp: str


class TaskUpdateMessage(WebSocketMessage):
    """Mensaje de actualización de tarea."""
    type: Literal["task_update"]
    data: TaskDict


class AgentStatusMessage(WebSocketMessage):
    """Mensaje de estado del agente."""
    type: Literal["agent_status"]
    data: AgentStatus



