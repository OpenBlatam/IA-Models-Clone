"""
Esquemas Pydantic para las rutas de la API con validaciones mejoradas.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Literal
from core.constants import TaskStatus, InstructionConfig


class CreateTaskRequest(BaseModel):
    """
    Request para crear una tarea con validaciones.
    
    Attributes:
        repository_owner: Propietario del repositorio (máximo 100 caracteres)
        repository_name: Nombre del repositorio (máximo 100 caracteres)
        instruction: Instrucción a ejecutar (5-5000 caracteres)
        metadata: Metadatos adicionales (opcional)
    """
    repository_owner: str = Field(
        ...,
        description="Propietario del repositorio",
        min_length=1,
        max_length=100
    )
    repository_name: str = Field(
        ...,
        description="Nombre del repositorio",
        min_length=1,
        max_length=100
    )
    instruction: str = Field(
        ...,
        min_length=InstructionConfig.MIN_LENGTH,
        max_length=InstructionConfig.MAX_LENGTH,
        description=f"Instrucción a ejecutar ({InstructionConfig.MIN_LENGTH}-{InstructionConfig.MAX_LENGTH} caracteres)"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos adicionales")
    
    @validator('repository_owner', 'repository_name')
    def validate_repository_fields(cls, v):
        """Validar campos de repositorio."""
        if not v or not v.strip():
            raise ValueError("El campo no puede estar vacío")
        v = v.strip()
        if v.startswith('.') or v.endswith('.'):
            raise ValueError("No puede comenzar o terminar con punto")
        if v.startswith('-') or v.endswith('-'):
            raise ValueError("No puede comenzar o terminar con guion")
        return v
    
    @validator('instruction')
    def validate_instruction(cls, v):
        """Validar instrucción."""
        if not v or not v.strip():
            raise ValueError("La instrucción no puede estar vacía")
        v = v.strip()
        if len(v) < InstructionConfig.MIN_LENGTH:
            raise ValueError(f"La instrucción debe tener al menos {InstructionConfig.MIN_LENGTH} caracteres")
        if len(v) > InstructionConfig.MAX_LENGTH:
            raise ValueError(f"La instrucción no puede tener más de {InstructionConfig.MAX_LENGTH} caracteres")
        return v
    
    class Config:
        """Configuración del modelo."""
        json_schema_extra = {
            "example": {
                "repository_owner": "octocat",
                "repository_name": "Hello-World",
                "instruction": "Create a new file called README.md with project description",
                "metadata": {"priority": "high"}
            }
        }


class TaskResponse(BaseModel):
    """
    Response con información de tarea con validaciones.
    
    Attributes:
        id: ID único de la tarea
        repository_owner: Propietario del repositorio
        repository_name: Nombre del repositorio
        instruction: Instrucción de la tarea
        status: Estado de la tarea (debe ser un estado válido)
        created_at: Timestamp de creación
        updated_at: Timestamp de última actualización
        started_at: Timestamp de inicio (opcional)
        completed_at: Timestamp de finalización (opcional)
        result: Resultado de la tarea (opcional)
        error: Mensaje de error (opcional)
        metadata: Metadatos adicionales (opcional)
    """
    id: str = Field(..., description="ID único de la tarea", min_length=1)
    repository_owner: str = Field(..., description="Propietario del repositorio")
    repository_name: str = Field(..., description="Nombre del repositorio")
    instruction: str = Field(..., description="Instrucción de la tarea")
    status: str = Field(..., description="Estado de la tarea")
    created_at: str = Field(..., description="Timestamp de creación")
    updated_at: str = Field(..., description="Timestamp de última actualización")
    started_at: Optional[str] = Field(None, description="Timestamp de inicio")
    completed_at: Optional[str] = Field(None, description="Timestamp de finalización")
    result: Optional[Dict[str, Any]] = Field(None, description="Resultado de la tarea")
    error: Optional[str] = Field(None, description="Mensaje de error")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos adicionales")
    
    @validator('status')
    def validate_status(cls, v):
        """Validar que el estado sea válido."""
        if not TaskStatus.is_valid(v):
            raise ValueError(
                f"Estado inválido: {v}. "
                f"Estados válidos: {', '.join(TaskStatus.ALL_STATES)}"
            )
        return v
    
    @validator('id')
    def validate_id(cls, v):
        """Validar formato básico de ID."""
        if not v or not v.strip():
            raise ValueError("El ID no puede estar vacío")
        if len(v) != 36:  # UUID v4 length
            raise ValueError(f"ID con formato inválido: {v}")
        return v.strip()
    
    class Config:
        """Configuración del modelo."""
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "repository_owner": "octocat",
                "repository_name": "Hello-World",
                "instruction": "Create README.md",
                "status": "pending",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
                "metadata": {}
            }
        }


class RepositoryInfoRequest(BaseModel):
    """Request para obtener información de repositorio."""
    owner: str = Field(..., description="Propietario del repositorio")
    repo: str = Field(..., description="Nombre del repositorio")


class RepositoryInfoResponse(BaseModel):
    """Response con información de repositorio."""
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


class AgentControlRequest(BaseModel):
    """
    Request para controlar el agente con validaciones.
    
    Attributes:
        action: Acción a realizar (debe ser una acción válida)
    """
    action: Literal["start", "stop", "pause", "resume"] = Field(
        ...,
        description="Acción a realizar: start, stop, pause, resume"
    )
    
    class Config:
        """Configuración del modelo."""
        json_schema_extra = {
            "example": {
                "action": "start"
            }
        }


class AgentStatusResponse(BaseModel):
    """Response con estado del agente."""
    is_running: bool
    current_task_id: Optional[str] = None
    last_activity: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WorkerMetricsResponse(BaseModel):
    """
    Response con métricas del worker con validaciones.
    
    Attributes:
        tasks_processed: Total de tareas procesadas (>= 0)
        tasks_succeeded: Total de tareas exitosas (>= 0)
        tasks_failed: Total de tareas fallidas (>= 0)
        last_task_time: Timestamp de última tarea (opcional)
        average_task_duration: Duración promedio de tareas en segundos (>= 0)
        circuit_state: Estado del circuit breaker
        consecutive_failures: Fallos consecutivos (>= 0)
        is_running: Si el worker está corriendo
    """
    tasks_processed: int = Field(..., description="Total de tareas procesadas", ge=0)
    tasks_succeeded: int = Field(..., description="Total de tareas exitosas", ge=0)
    tasks_failed: int = Field(..., description="Total de tareas fallidas", ge=0)
    last_task_time: Optional[str] = Field(None, description="Timestamp de última tarea")
    average_task_duration: float = Field(..., description="Duración promedio en segundos", ge=0.0)
    circuit_state: Literal["closed", "open", "half_open"] = Field(
        ...,
        description="Estado del circuit breaker"
    )
    consecutive_failures: int = Field(..., description="Fallos consecutivos", ge=0)
    is_running: bool = Field(..., description="Si el worker está corriendo")
    
    @validator('tasks_processed', 'tasks_succeeded', 'tasks_failed', 'consecutive_failures')
    def validate_non_negative_int(cls, v):
        """Validar que los valores sean no negativos."""
        if v < 0:
            raise ValueError(f"El valor debe ser no negativo, recibido: {v}")
        return v
    
    @validator('average_task_duration')
    def validate_non_negative_float(cls, v):
        """Validar que la duración sea no negativa."""
        if v < 0.0:
            raise ValueError(f"La duración debe ser no negativa, recibido: {v}")
        return v
    
    class Config:
        """Configuración del modelo."""
        json_schema_extra = {
            "example": {
                "tasks_processed": 100,
                "tasks_succeeded": 95,
                "tasks_failed": 5,
                "last_task_time": "2024-01-01T00:00:00",
                "average_task_duration": 2.5,
                "circuit_state": "closed",
                "consecutive_failures": 0,
                "is_running": True
            }
        }


class AgentMetricsResponse(BaseModel):
    """Response con métricas completas del agente."""
    worker_metrics: WorkerMetricsResponse
    agent_state: AgentStatusResponse
    task_statistics: Optional[Dict[str, Any]] = None
    timestamp: str


