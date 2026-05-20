"""
Task Executor Module
===================

Ejecutor de tareas con gestión de concurrencia y tracking.
Proporciona ejecución estructurada de tareas con límites de concurrencia.
"""

import asyncio
import logging
from typing import List, Callable, Any, Optional, Coroutine
from datetime import datetime

from .models import AgentTask
from .async_utils import AsyncTaskManager, gather_with_limit

logger = logging.getLogger(__name__)


class TaskExecutor:
    """
    Ejecutor de tareas con gestión de concurrencia.
    
    Proporciona ejecución estructurada de tareas con límites de concurrencia,
    tracking y manejo de errores.
    """
    
    def __init__(
        self,
        async_task_manager: Optional[AsyncTaskManager] = None,
        max_concurrent: int = 5
    ):
        """
        Inicializar ejecutor de tareas.
        
        Args:
            async_task_manager: Manager de tareas asíncronas
            max_concurrent: Número máximo de tareas concurrentes
        """
        self.async_task_manager = async_task_manager or AsyncTaskManager()
        self.max_concurrent = max_concurrent
    
    async def execute_task(
        self,
        task: AgentTask,
        processor: Callable[[AgentTask], Coroutine],
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> Any:
        """
        Ejecutar una tarea individual.
        
        Args:
            task: Tarea a ejecutar
            processor: Función que procesa la tarea
            on_complete: Callback cuando se complete exitosamente
            on_error: Callback cuando ocurra un error
            
        Returns:
            Resultado de la ejecución
        """
        task_name = f"task_{task.task_id}"
        
        async def execute_with_callbacks():
            try:
                result = await processor(task)
                
                if on_complete:
                    try:
                        if asyncio.iscoroutinefunction(on_complete):
                            await on_complete(task, result)
                        else:
                            on_complete(task, result)
                    except Exception as e:
                        logger.error(f"Error in on_complete callback: {e}", exc_info=True)
                
                return result
            except Exception as e:
                logger.error(f"Error executing task {task.task_id}: {e}", exc_info=True)
                
                if on_error:
                    try:
                        if asyncio.iscoroutinefunction(on_error):
                            await on_error(task, e)
                        else:
                            on_error(task, e)
                    except Exception as callback_error:
                        logger.error(f"Error in on_error callback: {callback_error}", exc_info=True)
                
                raise
        
        return await self.async_task_manager.create_task(
            execute_with_callbacks(),
            name=task_name
        )
    
    async def execute_tasks_concurrent(
        self,
        tasks: List[AgentTask],
        processor: Callable[[AgentTask], Coroutine],
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> List[Any]:
        """
        Ejecutar múltiples tareas con límite de concurrencia.
        
        Args:
            tasks: Lista de tareas a ejecutar
            processor: Función que procesa cada tarea
            on_complete: Callback cuando una tarea se complete exitosamente
            on_error: Callback cuando ocurra un error
            
        Returns:
            Lista de resultados
        """
        if not tasks:
            return []
        
        # Crear coroutines para cada tarea
        coroutines = [
            self.execute_task(task, processor, on_complete, on_error)
            for task in tasks
        ]
        
        # Ejecutar con límite de concurrencia
        results = await gather_with_limit(
            coroutines,
            limit=self.max_concurrent,
            return_exceptions=True
        )
        
        # Filtrar excepciones
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {tasks[i].task_id} failed: {result}", exc_info=True)
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def execute_task_background(
        self,
        task: AgentTask,
        processor: Callable[[AgentTask], Coroutine],
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> asyncio.Task:
        """
        Ejecutar una tarea en background (fire-and-forget).
        
        Args:
            task: Tarea a ejecutar
            processor: Función que procesa la tarea
            on_complete: Callback cuando se complete exitosamente
            on_error: Callback cuando ocurra un error
            
        Returns:
            Task creada
        """
        async def background_execution():
            try:
                result = await processor(task)
                if on_complete:
                    if asyncio.iscoroutinefunction(on_complete):
                        await on_complete(task, result)
                    else:
                        on_complete(task, result)
                return result
            except Exception as e:
                logger.error(f"Error in background task {task.task_id}: {e}", exc_info=True)
                if on_error:
                    try:
                        if asyncio.iscoroutinefunction(on_error):
                            await on_error(task, e)
                        else:
                            on_error(task, e)
                    except Exception as callback_error:
                        logger.error(f"Error in on_error callback: {callback_error}", exc_info=True)
                raise
        
        return self.async_task_manager.create_task(
            background_execution(),
            name=f"bg_task_{task.task_id}"
        )
    
    def get_active_count(self) -> int:
        """
        Obtener número de tareas activas.
        
        Returns:
            Número de tareas activas
        """
        return self.async_task_manager.get_active_count()
    
    def cleanup_completed(self) -> None:
        """Limpiar tareas completadas."""
        self.async_task_manager.cleanup_completed()
    
    def cancel_all(self) -> None:
        """Cancelar todas las tareas."""
        self.async_task_manager.cancel_all()


def create_task_executor(
    async_task_manager: Optional[AsyncTaskManager] = None,
    max_concurrent: int = 5
) -> TaskExecutor:
    """
    Factory function para crear TaskExecutor.
    
    Args:
        async_task_manager: Manager de tareas asíncronas
        max_concurrent: Número máximo de tareas concurrentes
        
    Returns:
        Instancia de TaskExecutor
    """
    return TaskExecutor(async_task_manager, max_concurrent)


