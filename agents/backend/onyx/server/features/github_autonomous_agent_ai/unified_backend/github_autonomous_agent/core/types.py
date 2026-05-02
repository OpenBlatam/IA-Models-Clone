"""
Type definitions y aliases para el core con mejoras y documentación.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from typing_extensions import TypedDict, NotRequired
from datetime import datetime

# Type aliases para mejor legibilidad
TaskDict = Dict[str, Any]
"""Diccionario que representa una tarea completa."""

AgentStateDict = Dict[str, Any]
"""Diccionario que representa el estado del agente."""

RepositoryInfoDict = Dict[str, Any]
"""Diccionario que representa información de un repositorio."""

InstructionParamsDict = Dict[str, Any]
"""Diccionario que representa parámetros de una instrucción."""

MetadataDict = Dict[str, Any]
"""Diccionario que representa metadatos adicionales."""

# Type definitions para respuestas
class TaskResult(TypedDict, total=False):
    """
    Resultado de una tarea con campos tipados.
    
    Attributes:
        success: Indica si la operación fue exitosa
        task_id: ID de la tarea
        result: Resultado de la tarea (opcional)
        error: Mensaje de error (opcional)
        error_type: Tipo de error (opcional)
    """
    success: bool
    task_id: str
    result: Optional[TaskDict]
    error: Optional[str]
    error_type: Optional[str]

# Type definitions para estados
class TaskStatusType:
    """
    Tipo para estados de tarea.
    
    Estos valores deben coincidir con los definidos en core.constants.TaskStatus.
    """
    PENDING: str = "pending"
    RUNNING: str = "running"
    COMPLETED: str = "completed"
    FAILED: str = "failed"
    CANCELLED: str = "cancelled"
    
    @classmethod
    def all_statuses(cls) -> List[str]:
        """Retorna todos los estados válidos."""
        return [
            cls.PENDING,
            cls.RUNNING,
            cls.COMPLETED,
            cls.FAILED,
            cls.CANCELLED
        ]


