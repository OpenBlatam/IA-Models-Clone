"""
Async Context Utilities
=======================
Utilidades para manejo de contextos asíncronos.
"""

from typing import Any, Optional, Dict, Callable
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)


class AsyncContextManager:
    """Manager para contextos asíncronos."""
    
    def __init__(self):
        self.contexts: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def set(self, key: str, value: Any):
        """Establecer valor en contexto."""
        async with self._lock:
            self.contexts[key] = value
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor del contexto."""
        return self.contexts.get(key, default)
    
    async def delete(self, key: str):
        """Eliminar valor del contexto."""
        async with self._lock:
            if key in self.contexts:
                del self.contexts[key]
    
    async def clear(self):
        """Limpiar todo el contexto."""
        async with self._lock:
            self.contexts.clear()
    
    def get_all(self) -> Dict[str, Any]:
        """Obtener todo el contexto."""
        return self.contexts.copy()


@asynccontextmanager
async def async_timeout_context(timeout: float):
    """
    Context manager para timeout.
    
    Args:
        timeout: Timeout en segundos
        
    Yields:
        None
    """
    try:
        yield
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout} seconds")


@asynccontextmanager
async def async_retry_context(
    max_attempts: int = 3,
    delay: float = 1.0,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Context manager para reintentos.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay entre intentos
        on_retry: Callback en cada reintento
        
    Yields:
        None
    """
    attempt = 0
    last_exception = None
    
    while attempt < max_attempts:
        try:
            yield
            return
        except Exception as e:
            last_exception = e
            attempt += 1
            
            if attempt < max_attempts:
                if on_retry:
                    on_retry(attempt, e)
                await asyncio.sleep(delay * attempt)
            else:
                raise last_exception
    
    raise last_exception


class AsyncTaskGroup:
    """Grupo de tareas asíncronas."""
    
    def __init__(self):
        self.tasks: List[asyncio.Task] = []
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, Exception] = {}
    
    async def add_task(self, name: str, coro: Callable):
        """
        Agregar tarea al grupo.
        
        Args:
            name: Nombre de la tarea
            coro: Coroutine a ejecutar
        """
        task = asyncio.create_task(coro)
        task.name = name
        self.tasks.append(task)
    
    async def wait_all(self, timeout: Optional[float] = None):
        """
        Esperar a que todas las tareas completen.
        
        Args:
            timeout: Timeout opcional
        """
        if timeout:
            done, pending = await asyncio.wait(
                self.tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancelar pendientes
            for task in pending:
                task.cancel()
        else:
            done = await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Procesar resultados
        for task in self.tasks:
            try:
                if hasattr(task, 'name'):
                    if task.exception():
                        self.errors[task.name] = task.exception()
                    else:
                        self.results[task.name] = task.result()
            except Exception as e:
                if hasattr(task, 'name'):
                    self.errors[task.name] = e
    
    def get_results(self) -> Dict[str, Any]:
        """Obtener resultados."""
        return self.results
    
    def get_errors(self) -> Dict[str, Exception]:
        """Obtener errores."""
        return self.errors

