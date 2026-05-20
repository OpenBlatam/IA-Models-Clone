"""
Continuous Processing Service
==============================
Servicio para procesamiento continuo que no se detiene hasta que se le indique.
Basado en patrones de continuous-agent.
"""

import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable
from datetime import datetime
from enum import Enum
from ..core.datetime_utils import get_utc_now

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..core.constants import MIN_INTERVAL_SECONDS


class ProcessorStatus(str, Enum):
    """Estado del procesador."""
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ContinuousProcessor:
    """
    Procesador continuo que ejecuta tareas en bucle hasta que se detiene explícitamente.
    
    Patrones extraídos de continuous-agent:
    - Auto-refresh continuo con intervalo configurable
    - Flag isActive para controlar ejecución
    - Reconnection handling
    - Estado persistente
    """
    
    def __init__(
        self,
        process_function: Callable[[], Awaitable[Any]],
        interval_seconds: float = 5.0,
        name: str = "ContinuousProcessor"
    ):
        """
        Inicializar procesador continuo.
        
        Args:
            process_function: Función async a ejecutar en cada ciclo
            interval_seconds: Intervalo entre ejecuciones (similar a refreshInterval)
            name: Nombre del procesador
        """
        self.process_function = process_function
        self.interval_seconds = interval_seconds
        self.name = name
        
        # Estado del procesador (similar a isActive en continuous-agent)
        self._is_active = False
        self._status = ProcessorStatus.IDLE
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # Estadísticas
        self._start_time: Optional[datetime] = None
        self._last_execution: Optional[datetime] = None
        self._execution_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        
        # Lock para operaciones que modifican estado
        self._lock = asyncio.Lock()
    
    @property
    def is_active(self) -> bool:
        """Verificar si el procesador está activo."""
        return self._is_active
    
    @property
    def status(self) -> ProcessorStatus:
        """Obtener estado actual."""
        return self._status
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del procesador."""
        uptime_seconds = None
        if self._start_time:
            uptime_seconds = (get_utc_now() - self._start_time).total_seconds()
        
        return {
            "is_active": self._is_active,
            "status": self._status.value,
            "interval_seconds": self.interval_seconds,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "last_execution": self._last_execution.isoformat() if self._last_execution else None,
            "last_error": self._last_error,
            "uptime_seconds": uptime_seconds,
            "start_time": self._start_time.isoformat() if self._start_time else None,
        }
    
    async def start(self) -> None:
        """Iniciar procesamiento continuo."""
        async with self._lock:
            if self._is_active:
                logger.warning("Processor already active", name=self.name)
                return
            
            if self._task and not self._task.done():
                logger.warning("Task already exists", name=self.name)
                return
            
            self._is_active = True
            self._status = ProcessorStatus.RUNNING
            self._stop_event.clear()
            self._start_time = get_utc_now()
            self._error_count = 0
            self._last_error = None
        
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Continuous processing started", name=self.name, interval=self.interval_seconds)
    
    async def stop(self) -> None:
        """Detener procesamiento continuo."""
        async with self._lock:
            if not self._is_active:
                logger.warning("Processor already inactive", name=self.name)
                return
            
            self._status = ProcessorStatus.STOPPING
            self._is_active = False
        
        self._stop_event.set()
        
        if self._task and not self._task.done():
            from ..core.constants import PROCESSOR_STOP_TIMEOUT
            try:
                await asyncio.wait_for(self._task, timeout=PROCESSOR_STOP_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("Timeout stopping processor, canceling task", name=self.name)
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        
        async with self._lock:
            self._status = ProcessorStatus.STOPPED
        
        logger.info("Continuous processing stopped", name=self.name)
    
    async def _run_loop(self) -> None:
        """Bucle principal de procesamiento continuo."""
        logger.info("Processing loop started", name=self.name)
        
        try:
            while self._is_active and not self._stop_event.is_set():
                try:
                    await self.process_function()
                    
                    async with self._lock:
                        self._last_execution = get_utc_now()
                        self._execution_count += 1
                        if self._status == ProcessorStatus.ERROR:
                            self._status = ProcessorStatus.RUNNING
                    
                    logger.debug("Execution completed", name=self.name, count=self._execution_count)
                    
                except Exception as e:
                    from ..core.logging_helpers import log_error, truncate_error_message
                    
                    error_msg = truncate_error_message(e)
                    
                    async with self._lock:
                        self._error_count += 1
                        self._last_error = error_msg
                        self._status = ProcessorStatus.ERROR
                    
                    log_error(
                        "Execution error",
                        e,
                        context={
                            "name": self.name,
                            "execution_count": self._execution_count,
                            "error_type": type(e).__name__
                        }
                    )
                
                # Optimized: simple sleep with check
                await asyncio.sleep(self.interval_seconds)
                if self._stop_event.is_set():
                    break
        
        except asyncio.CancelledError:
            logger.info("Loop cancelled", name=self.name)
            async with self._lock:
                self._status = ProcessorStatus.STOPPED
            raise
        
        except Exception as e:
            from ..core.logging_helpers import log_error
            log_error(
                "Fatal loop error",
                e,
                context={"name": self.name, "execution_count": self._execution_count}
            )
            async with self._lock:
                self._status = ProcessorStatus.ERROR
                self._is_active = False
        
        finally:
            logger.info("Processing loop finished", name=self.name)
    
    async def restart(self, new_interval: Optional[float] = None) -> None:
        """
        Reiniciar procesador (stop + start).
        
        Stops the processor, waits briefly, then starts it again.
        Useful for applying configuration changes.
        
        Args:
            new_interval: Optional new interval to set before restarting
        """
        await self.stop()
        # Small pause to ensure clean shutdown
        await asyncio.sleep(0.5)
        
        if new_interval is not None:
            self.update_interval(new_interval)
        
        await self.start()
    
    def update_interval(self, new_interval: float) -> None:
        """
        Actualizar intervalo de procesamiento.
        
        Args:
            new_interval: Nuevo intervalo en segundos (validado contra MIN_INTERVAL_SECONDS)
            
        Raises:
            ValueError: Si el intervalo es menor que MIN_INTERVAL_SECONDS
        """
        if new_interval < MIN_INTERVAL_SECONDS:
            from ..core.exceptions import ValidationError
            raise ValidationError(
                f"Interval must be at least {MIN_INTERVAL_SECONDS} seconds, got {new_interval}"
            )
        self.interval_seconds = new_interval
        logger.info("Interval updated", name=self.name, interval=self.interval_seconds)

