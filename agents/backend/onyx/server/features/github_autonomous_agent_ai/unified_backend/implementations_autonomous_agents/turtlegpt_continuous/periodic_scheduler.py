"""
Periodic Scheduler Module
========================

Programación y ejecución centralizada de tareas periódicas.
Proporciona una interfaz estructurada para operaciones que se ejecutan en intervalos regulares.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class PeriodicTaskStatus(Enum):
    """Estado de una tarea periódica."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PeriodicTask:
    """
    Representa una tarea peri?dica.
    
    Attributes:
        name: Nombre de la tarea
        func: Funci?n a ejecutar
        interval: Intervalo entre ejecuciones en segundos
        status: Estado actual de la tarea
        last_run: ?ltima vez que se ejecut?
        next_run: Pr?xima ejecuci?n programada
        run_count: N?mero de veces ejecutada
        error_count: N?mero de errores
        enabled: Si la tarea est? habilitada
    """
    
    def __init__(
        self,
        name: str,
        func: Callable,
        interval: float,
        initial_delay: float = 0.0,
        enabled: bool = True
    ):
        """
        Inicializar tarea periódica.
        
        Args:
            name: Nombre de la tarea
            func: Funci?n a ejecutar
            interval: Intervalo entre ejecuciones en segundos
            initial_delay: Retraso inicial antes de la primera ejecuci?n
            enabled: Si la tarea est? habilitada
        """
        self.name = name
        self.func = func
        self.interval = interval
        self.initial_delay = initial_delay
        self.enabled = enabled
        
        self.status = PeriodicTaskStatus.PENDING
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.run_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
        
        # Calcular pr?xima ejecuci?n
        if initial_delay > 0:
            self.next_run = datetime.now() + timedelta(seconds=initial_delay)
        else:
            self.next_run = datetime.now()
        
        self._task: Optional[asyncio.Task] = None
    
    def should_run(self) -> bool:
        """
        Verificar si la tarea debe ejecutarse.
        
        Returns:
            True si debe ejecutarse
        """
        if not self.enabled:
            return False
        
        if self.next_run is None:
            return True
        
        return datetime.now() >= self.next_run
    
    def update_schedule(self) -> None:
        """Actualizar programación para próxima ejecución."""
        self.next_run = datetime.now() + timedelta(seconds=self.interval)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtener informaci?n de la tarea.
        
        Returns:
            Dict con informaci?n de la tarea
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "interval": self.interval,
            "enabled": self.enabled,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_error": self.last_error
        }


class PeriodicScheduler:
    """
    Programador de tareas periódicas.
    
    Proporciona una interfaz estructurada para programar
    y ejecutar tareas que se repiten en intervalos regulares.
    """
    
    def __init__(self):
        """Inicializar programador."""
        self.tasks: Dict[str, PeriodicTask] = {}
        self._stop_event = asyncio.Event()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register_task(
        self,
        name: str,
        func: Callable,
        interval: float,
        initial_delay: float = 0.0,
        enabled: bool = True
    ) -> PeriodicTask:
        """
        Registrar una tarea periódica.
        
        Args:
            name: Nombre de la tarea
            func: Funci?n a ejecutar
            interval: Intervalo entre ejecuciones en segundos
            initial_delay: Retraso inicial antes de la primera ejecuci?n
            enabled: Si la tarea est? habilitada
            
        Returns:
            Tarea peri?dica creada
        """
        task = PeriodicTask(
            name=name,
            func=func,
            interval=interval,
            initial_delay=initial_delay,
            enabled=enabled
        )
        
        self.tasks[name] = task
        logger.debug(f"Registered periodic task: {name} (interval: {interval}s)")
        
        return task
    
    def unregister_task(self, name: str) -> bool:
        """
        Desregistrar una tarea periódica.
        
        Args:
            name: Nombre de la tarea
            
        Returns:
            True si se desregistr? exitosamente
        """
        if name in self.tasks:
            task = self.tasks[name]
            if task._task and not task._task.done():
                task._task.cancel()
            
            del self.tasks[name]
            logger.debug(f"Unregistered periodic task: {name}")
            return True
        
        return False
    
    def enable_task(self, name: str) -> bool:
        """
        Habilitar una tarea.
        
        Args:
            name: Nombre de la tarea
            
        Returns:
            True si se habilit? exitosamente
        """
        if name in self.tasks:
            self.tasks[name].enabled = True
            logger.debug(f"Enabled periodic task: {name}")
            return True
        return False
    
    def disable_task(self, name: str) -> bool:
        """
        Deshabilitar una tarea.
        
        Args:
            name: Nombre de la tarea
            
        Returns:
            True si se deshabilitó exitosamente
        """
        if name in self.tasks:
            self.tasks[name].enabled = False
            logger.debug(f"Disabled periodic task: {name}")
            return True
        return False
    
    async def start(self) -> None:
        """Iniciar el programador."""
        if self._running:
            logger.warning("Scheduler is already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Periodic scheduler started")
    
    async def stop(self) -> None:
        """Detener el programador."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        # Cancelar todas las tareas
        for task in self.tasks.values():
            if task._task and not task._task.done():
                task._task.cancel()
        
        # Esperar a que termine el scheduler
        if self._scheduler_task:
            try:
                await asyncio.wait_for(self._scheduler_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Scheduler task did not stop in time")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")
        
        logger.info("Periodic scheduler stopped")
    
    async def _scheduler_loop(self) -> None:
        """Loop principal del programador."""
        while not self._stop_event.is_set():
            try:
                # Verificar tareas que deben ejecutarse
                for task in self.tasks.values():
                    if task.should_run():
                        await self._execute_task(task)
                
                # Esperar un poco antes de verificar nuevamente
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)
    
    async def _execute_task(self, task: PeriodicTask) -> None:
        """
        Ejecutar una tarea peri?dica.
        
        Args:
            task: Tarea a ejecutar
        """
        if task.status == PeriodicTaskStatus.RUNNING:
            return  # Ya está ejecutándose
        
        task.status = PeriodicTaskStatus.RUNNING
        task.last_run = datetime.now()
        
        async def execute():
            try:
                if asyncio.iscoroutinefunction(task.func):
                    await task.func()
                else:
                    task.func()
                
                task.status = PeriodicTaskStatus.COMPLETED
                task.run_count += 1
                task.last_error = None
                
            except Exception as e:
                task.status = PeriodicTaskStatus.FAILED
                task.error_count += 1
                task.last_error = str(e)
                logger.error(f"Error executing periodic task '{task.name}': {e}", exc_info=True)
            
            finally:
                task.update_schedule()
                task.status = PeriodicTaskStatus.PENDING
        
        task._task = asyncio.create_task(execute())
    
    def get_task(self, name: str) -> Optional[PeriodicTask]:
        """
        Obtener una tarea por nombre.
        
        Args:
            name: Nombre de la tarea
            
        Returns:
            Tarea periódica o None
        """
        return self.tasks.get(name)
    
    def get_all_tasks(self) -> List[PeriodicTask]:
        """
        Obtener todas las tareas registradas.
        
        Returns:
            Lista de tareas
        """
        return list(self.tasks.values())
    
    def get_task_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Obtener informaci?n de una tarea.
        
        Args:
            name: Nombre de la tarea
            
        Returns:
            Informaci?n de la tarea o None
        """
        task = self.tasks.get(name)
        return task.get_info() if task else None
    
    def get_all_tasks_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener información de todas las tareas.
        
        Returns:
            Dict con informaci?n de todas las tareas
        """
        return {
            name: task.get_info()
            for name, task in self.tasks.items()
        }
    
    def is_running(self) -> bool:
        """
        Verificar si el programador est? ejecut?ndose.
        
        Returns:
            True si est? ejecut?ndose
        """
        return self._running
    
    def get_task_count(self) -> int:
        """
        Obtener número de tareas registradas.
        
        Returns:
            N?mero de tareas
        """
        return len(self.tasks)


def create_periodic_scheduler() -> PeriodicScheduler:
    """
    Factory function para crear PeriodicScheduler.
    
    Returns:
        Instancia de PeriodicScheduler
    """
    return PeriodicScheduler()


