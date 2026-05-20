"""
Async Helpers - Utilidades para operaciones asíncronas
======================================================

Funciones helper para facilitar el trabajo con operaciones asíncronas.
"""

import asyncio
import logging
from typing import Callable, Any, Optional, List, Dict, TypeVar, Coroutine
from functools import wraps
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncTaskManager:
    """
    Gestor de tareas asíncronas.
    
    Permite ejecutar y gestionar tareas asíncronas con control de concurrencia.
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Inicializar gestor de tareas.
        
        Args:
            max_concurrent: Número máximo de tareas concurrentes
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def execute(
        self,
        task_id: str,
        coro: Coroutine,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Ejecutar tarea asíncrona con control de concurrencia.
        
        Args:
            task_id: ID único de la tarea
            coro: Coroutine a ejecutar
            timeout: Timeout en segundos (opcional)
        
        Returns:
            Resultado de la tarea
        
        Raises:
            asyncio.TimeoutError: Si se excede el timeout
        """
        async with self.semaphore:
            async with self._lock:
                if task_id in self.active_tasks:
                    raise ValueError(f"Task {task_id} already exists")
            
            try:
                if timeout:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro
                
                async with self._lock:
                    self.completed_tasks.append({
                        "task_id": task_id,
                        "status": "completed",
                        "completed_at": datetime.utcnow()
                    })
                
                return result
            except asyncio.TimeoutError:
                async with self._lock:
                    self.completed_tasks.append({
                        "task_id": task_id,
                        "status": "timeout",
                        "completed_at": datetime.utcnow()
                    })
                raise
            except Exception as e:
                async with self._lock:
                    self.completed_tasks.append({
                        "task_id": task_id,
                        "status": "failed",
                        "error": str(e),
                        "completed_at": datetime.utcnow()
                    })
                raise
            finally:
                async with self._lock:
                    self.active_tasks.pop(task_id, None)
    
    async def execute_batch(
        self,
        tasks: List[Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Ejecutar múltiples tareas en paralelo.
        
        Args:
            tasks: Lista de diccionarios con task_id y coro
            timeout: Timeout por tarea (opcional)
        
        Returns:
            Lista de resultados
        """
        async def execute_one(task_info: Dict[str, Any]) -> Any:
            return await self.execute(
                task_id=task_info["task_id"],
                coro=task_info["coro"],
                timeout=timeout
            )
        
        results = await asyncio.gather(
            *[execute_one(task) for task in tasks],
            return_exceptions=True
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del gestor.
        
        Returns:
            Diccionario con estadísticas
        """
        async def _get_stats():
            async with self._lock:
                return {
                    "max_concurrent": self.max_concurrent,
                    "active": len(self.active_tasks),
                    "completed": len(self.completed_tasks),
                    "available": self.max_concurrent - len(self.active_tasks)
                }
        
        # Ejecutar de forma síncrona si es posible
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Si el loop está corriendo, retornar stats básicos
                return {
                    "max_concurrent": self.max_concurrent,
                    "active": len(self.active_tasks),
                    "completed": len(self.completed_tasks),
                }
            return loop.run_until_complete(_get_stats())
        except RuntimeError:
            return {
                "max_concurrent": self.max_concurrent,
                "active": len(self.active_tasks),
                "completed": len(self.completed_tasks),
            }


async def run_with_timeout(
    coro: Coroutine,
    timeout: float,
    default: Any = None
) -> Any:
    """
    Ejecutar coroutine con timeout y valor por defecto.
    
    Args:
        coro: Coroutine a ejecutar
        timeout: Timeout en segundos
        default: Valor por defecto si hay timeout
    
    Returns:
        Resultado de la coroutine o default
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default


async def gather_with_errors(
    *coros: Coroutine,
    return_exceptions: bool = True
) -> List[Any]:
    """
    Ejecutar múltiples coroutines y manejar errores.
    
    Args:
        *coros: Coroutines a ejecutar
        return_exceptions: Si retornar excepciones en lugar de lanzarlas
    
    Returns:
        Lista de resultados
    """
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)


async def retry_async(
    func: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Reintentar función asíncrona con exponential backoff.
    
    Args:
        func: Función a ejecutar
        max_attempts: Número máximo de intentos
        delay: Delay inicial en segundos
        backoff: Factor de backoff
        exceptions: Tipos de excepciones a capturar
    
    Returns:
        Resultado de la función
    
    Raises:
        Exception: Si todos los intentos fallan
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_attempts} attempts failed")
    
    raise last_exception


@asynccontextmanager
async def async_lock(lock: asyncio.Lock):
    """
    Context manager para asyncio.Lock.
    
    Args:
        lock: Lock a adquirir
    
    Yields:
        None
    """
    await lock.acquire()
    try:
        yield
    finally:
        lock.release()


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator para reintentar funciones asíncronas.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial
        backoff: Factor de backoff
        exceptions: Excepciones a capturar
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def attempt():
                return await func(*args, **kwargs)
            
            return await retry_async(
                attempt,
                max_attempts=max_attempts,
                delay=delay,
                backoff=backoff,
                exceptions=exceptions
            )
        
        return wrapper
    return decorator


async def wait_for_condition_async(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    error_message: Optional[str] = None
) -> bool:
    """
    Esperar hasta que una condición sea verdadera (async).
    
    Args:
        condition: Función que retorna True cuando se cumple
        timeout: Tiempo máximo en segundos
        interval: Intervalo entre verificaciones
        error_message: Mensaje de error personalizado
    
    Returns:
        True si se cumplió
    
    Raises:
        TimeoutError: Si no se cumple en el tiempo especificado
    """
    start_time = asyncio.get_event_loop().time()
    
    while True:
        if condition():
            return True
        
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed >= timeout:
            if error_message:
                raise TimeoutError(error_message)
            raise TimeoutError(f"Condition not met within {timeout} seconds")
        
        await asyncio.sleep(interval)


async def batch_process(
    items: List[Any],
    processor: Callable[[Any], Coroutine],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Procesar items en lotes con control de concurrencia.
    
    Args:
        items: Lista de items a procesar
        processor: Función async que procesa un item
        batch_size: Tamaño del lote
        max_concurrent: Máximo de items concurrentes
    
    Returns:
        Lista de resultados
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    
    async def process_with_semaphore(item: Any) -> Any:
        async with semaphore:
            return await processor(item)
    
    # Procesar en lotes
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_with_semaphore(item) for item in batch],
            return_exceptions=True
        )
        results.extend(batch_results)
    
    return results


class PeriodicTask:
    """
    Tarea periódica que se ejecuta en intervalos regulares.
    """
    
    def __init__(
        self,
        func: Callable,
        interval: float,
        name: Optional[str] = None
    ):
        """
        Inicializar tarea periódica.
        
        Args:
            func: Función a ejecutar
            interval: Intervalo en segundos
            name: Nombre de la tarea (opcional)
        """
        self.func = func
        self.interval = interval
        self.name = name or func.__name__
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Iniciar tarea periódica"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"Started periodic task: {self.name}")
    
    async def stop(self) -> None:
        """Detener tarea periódica"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped periodic task: {self.name}")
    
    async def _run(self) -> None:
        """Loop de ejecución"""
        while self._running:
            try:
                if asyncio.iscoroutinefunction(self.func):
                    await self.func()
                else:
                    self.func()
            except Exception as e:
                logger.error(f"Error in periodic task {self.name}: {e}", exc_info=True)
            
            await asyncio.sleep(self.interval)

