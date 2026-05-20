"""
Async Utilities - Utilidades para programación asíncrona
=======================================================

Utilidades para trabajar con asyncio y tareas asíncronas de forma consistente.
"""

import asyncio
from typing import Dict, List, Any, Optional, TypeVar, Awaitable
from .error_handling import safe_async_call

T = TypeVar('T')


def get_completed_tasks(active_tasks: Dict[str, asyncio.Task]) -> List[str]:
    """
    Obtener lista de IDs de tareas completadas.
    
    Args:
        active_tasks: Diccionario de tareas activas (task_id -> Task).
    
    Returns:
        Lista de IDs de tareas completadas.
    """
    return [
        task_id
        for task_id, task in active_tasks.items()
        if task.done()
    ]


def cleanup_completed_tasks(active_tasks: Dict[str, asyncio.Task]) -> int:
    """
    Limpiar tareas completadas de un diccionario de tareas activas.
    
    Args:
        active_tasks: Diccionario de tareas activas (task_id -> Task).
    
    Returns:
        Número de tareas limpiadas.
    """
    completed = get_completed_tasks(active_tasks)
    for task_id in completed:
        del active_tasks[task_id]
    return len(completed)




async def gather_tasks_safely(
    tasks: Dict[str, asyncio.Task],
    return_exceptions: bool = True,
    logger_instance: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Ejecutar múltiples tareas de forma segura usando gather.
    
    Args:
        tasks: Diccionario de tareas (task_id -> Task).
        return_exceptions: Si True, retorna excepciones en lugar de lanzarlas.
        logger_instance: Logger a usar (opcional).
    
    Returns:
        Diccionario con resultados (task_id -> resultado o excepción).
    """
    if not tasks:
        return {}
    
    results = await asyncio.gather(
        *tasks.values(),
        return_exceptions=return_exceptions
    )
    
    return dict(zip(tasks.keys(), results))


def can_add_task(
    active_tasks: Dict[str, asyncio.Task],
    max_concurrent: int
) -> bool:
    """
    Verificar si se puede agregar una nueva tarea.
    
    Args:
        active_tasks: Diccionario de tareas activas.
        max_concurrent: Número máximo de tareas concurrentes.
    
    Returns:
        True si se puede agregar, False en caso contrario.
    """
    return len(active_tasks) < max_concurrent



