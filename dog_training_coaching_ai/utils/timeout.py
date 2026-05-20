"""
Timeout Utilities
=================
Utilidades para manejo de timeouts.
"""

import asyncio
from typing import Callable, Any, Optional
from contextlib import asynccontextmanager


class TimeoutError(Exception):
    """Excepción de timeout."""
    pass


async def with_timeout(
    coro: Callable,
    timeout: float,
    timeout_message: Optional[str] = None
) -> Any:
    """
    Ejecutar coroutine con timeout.
    
    Args:
        coro: Coroutine a ejecutar
        timeout: Timeout en segundos
        timeout_message: Mensaje de error personalizado
        
    Returns:
        Resultado de la coroutine
        
    Raises:
        TimeoutError: Si se excede el timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        message = timeout_message or f"Operation timed out after {timeout} seconds"
        raise TimeoutError(message)


@asynccontextmanager
async def timeout_context(timeout: float):
    """
    Context manager para timeout.
    
    Args:
        timeout: Timeout en segundos
        
    Yields:
        None
        
    Raises:
        TimeoutError: Si se excede el timeout
    """
    try:
        yield
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout} seconds")


class TimeoutManager:
    """Manager para múltiples timeouts."""
    
    def __init__(self):
        self.active_timeouts: dict = {}
    
    async def execute_with_timeout(
        self,
        task_id: str,
        coro: Callable,
        timeout: float
    ) -> Any:
        """
        Ejecutar tarea con timeout y tracking.
        
        Args:
            task_id: ID único de la tarea
            coro: Coroutine a ejecutar
            timeout: Timeout en segundos
            
        Returns:
            Resultado de la coroutine
        """
        task = asyncio.create_task(coro)
        self.active_timeouts[task_id] = {
            "task": task,
            "timeout": timeout,
            "start_time": asyncio.get_event_loop().time()
        }
        
        try:
            result = await with_timeout(task, timeout)
            del self.active_timeouts[task_id]
            return result
        except TimeoutError:
            if task_id in self.active_timeouts:
                task = self.active_timeouts[task_id]["task"]
                task.cancel()
                del self.active_timeouts[task_id]
            raise
    
    def cancel_task(self, task_id: str):
        """Cancelar tarea por ID."""
        if task_id in self.active_timeouts:
            task = self.active_timeouts[task_id]["task"]
            task.cancel()
            del self.active_timeouts[task_id]
    
    def get_active_tasks(self) -> list:
        """Obtener lista de tareas activas."""
        return list(self.active_timeouts.keys())

