"""
Task Utilities - Utilidades para tareas
=======================================

Utilidades para trabajar con tareas del agente.
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime


def create_task_id(index: int = 0) -> str:
    """
    Crear ID único para una tarea.
    
    Args:
        index: Índice opcional para incluir en el ID.
    
    Returns:
        ID único de tarea.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"task_{timestamp}_{unique_id}_{index}"


def count_tasks_by_status(tasks: Dict[str, Any], status: str) -> int:
    """
    Contar tareas por estado.
    
    Args:
        tasks: Diccionario de tareas (task_id -> task).
        status: Estado a contar.
    
    Returns:
        Número de tareas con el estado especificado.
    """
    return sum(1 for task in tasks.values() if getattr(task, 'status', None) == status)


def tasks_to_dict_list(
    tasks: List[Any],
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Convertir lista de tareas a lista de diccionarios.
    
    Args:
        tasks: Lista de objetos Task.
        limit: Límite máximo de tareas a retornar (None = todas).
    
    Returns:
        Lista de diccionarios con información de las tareas.
    """
    task_dicts = []
    
    for task in tasks:
        task_dict = {
            "id": task.id,
            "command": task.command,
            "status": task.status,
            "timestamp": task.timestamp.isoformat() if isinstance(task.timestamp, datetime) else str(task.timestamp),
        }
        
        if hasattr(task, 'result') and task.result:
            task_dict["result"] = task.result[:200] if len(task.result) > 200 else task.result
        
        if hasattr(task, 'error') and task.error:
            task_dict["error"] = task.error[:200] if len(task.error) > 200 else task.error
        
        task_dicts.append(task_dict)
        
        if limit and len(task_dicts) >= limit:
            break
    
    return task_dicts


def filter_tasks_by_status(
    tasks: Dict[str, Any],
    status: str
) -> List[Any]:
    """
    Filtrar tareas por estado.
    
    Args:
        tasks: Diccionario de tareas (task_id -> task).
        status: Estado a filtrar.
    
    Returns:
        Lista de tareas con el estado especificado.
    """
    return [task for task in tasks.values() if getattr(task, 'status', None) == status]


def get_task_summary(tasks: Dict[str, Any]) -> Dict[str, Any]:
    """
    Obtener resumen de tareas.
    
    Args:
        tasks: Diccionario de tareas (task_id -> task).
    
    Returns:
        Diccionario con resumen de tareas.
    """
    return {
        "total": len(tasks),
        "pending": count_tasks_by_status(tasks, "pending"),
        "running": count_tasks_by_status(tasks, "running"),
        "completed": count_tasks_by_status(tasks, "completed"),
        "failed": count_tasks_by_status(tasks, "failed"),
    }


def format_task_command(command: str, max_length: int = 50) -> str:
    """
    Formatear comando de tarea para mostrar.
    
    Args:
        command: Comando a formatear.
        max_length: Longitud máxima (default: 50).
    
    Returns:
        Comando formateado.
    """
    if not command:
        return ""
    
    if len(command) <= max_length:
        return command
    
    return command[:max_length] + "..."


def validate_task_id(task_id: str) -> bool:
    """
    Validar formato de ID de tarea.
    
    Args:
        task_id: ID a validar.
    
    Returns:
        True si el ID es válido, False en caso contrario.
    """
    if not task_id or not isinstance(task_id, str):
        return False
    
    # Formato esperado: task_YYYYMMDDHHMMSS_UUID_index
    parts = task_id.split("_")
    if len(parts) < 3:
        return False
    
    if parts[0] != "task":
        return False
    
    return True




