"""
Funciones helper para operaciones comunes con validaciones mejoradas.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from config.logging_config import get_logger
from core.constants import TaskStatus, ErrorMessages

logger = get_logger(__name__)


def generate_task_id() -> str:
    """
    Generar un ID único para una tarea.
    
    Returns:
        String con UUID único
    """
    return str(uuid.uuid4())


def create_task_dict(
    repository_owner: str,
    repository_name: str,
    instruction: str,
    metadata: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Crear diccionario de tarea con valores por defecto y validaciones.
    
    Args:
        repository_owner: Propietario del repositorio
        repository_name: Nombre del repositorio
        instruction: Instrucción a ejecutar
        metadata: Metadatos adicionales
        task_id: ID de la tarea (se genera si no se proporciona)
        
    Returns:
        Diccionario con información de la tarea
        
    Raises:
        ValueError: Si los parámetros requeridos son inválidos
    """
    # Validaciones
    if not repository_owner or not isinstance(repository_owner, str) or not repository_owner.strip():
        raise ValueError("El propietario del repositorio es requerido y no puede estar vacío")
    
    if not repository_name or not isinstance(repository_name, str) or not repository_name.strip():
        raise ValueError("El nombre del repositorio es requerido y no puede estar vacío")
    
    if not instruction or not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("La instrucción es requerida y no puede estar vacía")
    
    # Validar task_id si se proporciona
    if task_id:
        try:
            # Validar formato UUID
            uuid.UUID(task_id)
        except (ValueError, TypeError):
            logger.warning(f"Task ID proporcionado no es un UUID válido: {task_id}, generando uno nuevo")
            task_id = None
    
    now = datetime.now().isoformat()
    task_dict = {
        "id": task_id or generate_task_id(),
        "repository_owner": repository_owner.strip(),
        "repository_name": repository_name.strip(),
        "instruction": instruction.strip(),
        "status": TaskStatus.PENDING,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
        "metadata": metadata or {}
    }
    
    logger.debug(f"Tarea creada: {task_dict['id']} para {repository_owner}/{repository_name}")
    return task_dict


def create_agent_state(
    is_running: bool = False,
    current_task_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crear diccionario de estado del agente con validaciones.
    
    Args:
        is_running: Si el agente está corriendo
        current_task_id: ID de la tarea actual (opcional)
        metadata: Metadatos adicionales (opcional)
        
    Returns:
        Diccionario con estado del agente
        
    Raises:
        ValueError: Si el task_id no es un UUID válido
    """
    # Validar task_id si se proporciona
    if current_task_id:
        try:
            uuid.UUID(current_task_id)
        except (ValueError, TypeError):
            logger.warning(f"Task ID no es un UUID válido: {current_task_id}")
            raise ValueError(f"Task ID inválido: {current_task_id}")
    
    state = {
        "id": "main",
        "is_running": bool(is_running),
        "current_task_id": current_task_id,
        "last_activity": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    logger.debug(f"Estado del agente creado: is_running={is_running}, task_id={current_task_id}")
    return state


def format_error_response(
    error: str,
    error_type: Optional[str] = None,
    task_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Formatear respuesta de error de forma consistente con validaciones.
    
    Args:
        error: Mensaje de error
        error_type: Tipo de error (opcional)
        task_id: ID de la tarea (opcional)
        details: Detalles adicionales del error (opcional)
        timestamp: Timestamp del error (opcional, se genera si no se proporciona)
        
    Returns:
        Diccionario con respuesta de error formateada
    """
    if not error or not isinstance(error, str):
        error = ErrorMessages.VALIDATION_ERROR
    
    response = {
        "success": False,
        "error": error,
        "timestamp": timestamp or datetime.now().isoformat()
    }
    
    if error_type:
        response["error_type"] = str(error_type)
    
    if task_id:
        # Validar formato UUID
        try:
            uuid.UUID(task_id)
            response["task_id"] = task_id
        except (ValueError, TypeError):
            logger.warning(f"Task ID inválido en error response: {task_id}")
    
    if details:
        response["details"] = details
    
    return response


def format_success_response(
    result: Any,
    task_id: Optional[str] = None,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Formatear respuesta de éxito de forma consistente con validaciones.
    
    Args:
        result: Resultado de la operación
        task_id: ID de la tarea (opcional)
        message: Mensaje de éxito (opcional)
        metadata: Metadatos adicionales (opcional)
        timestamp: Timestamp del éxito (opcional, se genera si no se proporciona)
        
    Returns:
        Diccionario con respuesta de éxito formateada
    """
    response = {
        "success": True,
        "result": result,
        "timestamp": timestamp or datetime.now().isoformat()
    }
    
    if task_id:
        # Validar formato UUID
        try:
            uuid.UUID(task_id)
            response["task_id"] = task_id
        except (ValueError, TypeError):
            logger.warning(f"Task ID inválido en success response: {task_id}")
    
    if message:
        response["message"] = str(message)
    
    if metadata:
        response["metadata"] = metadata
    
    return response

