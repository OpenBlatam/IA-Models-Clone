"""
Worker - Worker para ejecutar tareas de forma continua.
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

from config.settings import settings
from config.logging_config import get_logger
from core.github_client import GitHubClient
from core.storage import TaskStorage
from core.task_processor import TaskProcessor
from core.constants import TaskStatus, ErrorMessages, SuccessMessages
from core.exceptions import TaskProcessingError, StorageError

logger = get_logger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"  # Funcionando normalmente
    OPEN = "open"  # Circuito abierto, no procesar tareas
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó


class WorkerManager:
    """Gestor de workers para ejecución continua de tareas."""

    def __init__(
        self,
        storage: Optional[TaskStorage] = None,
        task_processor: Optional[TaskProcessor] = None
    ):
        """
        Inicializar gestor de workers.
        
        Args:
            storage: Almacenamiento de tareas (opcional, se crea si no se proporciona)
            task_processor: Procesador de tareas (opcional, se crea si no se proporciona)
        """
        self.storage = storage or TaskStorage()
        self.task_processor = task_processor
        self.is_running = False
        self.worker_task: Optional[asyncio.Task] = None
        
        # Circuit breaker state
        self.circuit_state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.max_failures = settings.CIRCUIT_BREAKER_MAX_FAILURES
        self.circuit_open_until: Optional[datetime] = None
        self.circuit_timeout = timedelta(seconds=settings.CIRCUIT_BREAKER_TIMEOUT)
        
        # Métricas
        self.metrics: Dict[str, Any] = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "last_task_time": None,
            "average_task_duration": 0.0
        }

    async def start(self):
        """
        Iniciar el worker con validaciones y manejo de errores mejorado.
        
        Raises:
            StorageError: Si hay un error al inicializar el storage
            ValueError: Si la configuración es inválida
        """
        if self.is_running:
            logger.warning("Worker ya está en ejecución, ignorando solicitud de inicio")
            return

        try:
            # Inicializar base de datos
            logger.info("Inicializando base de datos...")
            await self.storage.init_db()
            logger.debug("Base de datos inicializada correctamente")
            
            # Inicializar task processor si no está disponible
            if not self.task_processor:
                if settings.GITHUB_TOKEN:
                    logger.info("Creando task processor con GitHub client...")
                    try:
                        github_client = GitHubClient()
                        self.task_processor = TaskProcessor(github_client, self.storage)
                        logger.debug("Task processor creado exitosamente")
                    except Exception as e:
                        logger.error(f"Error al crear task processor: {e}", exc_info=True)
                        raise ValueError(f"No se pudo inicializar el task processor: {e}") from e
                else:
                    logger.warning(
                        "GitHub token no configurado. "
                        "El worker se iniciará pero algunas funciones pueden no estar disponibles."
                    )
            
            # Actualizar estado
            self.is_running = True
            await self.storage.update_agent_state({
                "id": "main",
                "is_running": True,
                "current_task_id": None,
                "last_activity": datetime.now().isoformat(),
                "metadata": {}
            })
            
            # Iniciar loop de worker
            self.worker_task = asyncio.create_task(self._worker_loop())
            logger.info("✅ Worker iniciado correctamente")
            
        except StorageError as e:
            logger.error(f"Error de almacenamiento al iniciar worker: {e}", exc_info=True)
            self.is_running = False
            raise
        except Exception as e:
            logger.error(f"Error inesperado al iniciar worker: {e}", exc_info=True)
            self.is_running = False
            raise ValueError(f"Error al iniciar worker: {e}") from e

    async def stop(self):
        """
        Detener el worker con manejo de errores mejorado.
        
        Raises:
            StorageError: Si hay un error al actualizar el estado
        """
        if not self.is_running:
            logger.debug("Worker no está en ejecución, ignorando solicitud de detención")
            return

        logger.info("Deteniendo worker...")
        self.is_running = False
        
        try:
            await self.storage.update_agent_state({
                "id": "main",
                "is_running": False,
                "current_task_id": None,
                "last_activity": datetime.now().isoformat(),
                "metadata": {}
            })
        except StorageError as e:
            logger.error(f"Error al actualizar estado al detener worker: {e}", exc_info=True)
            # Continuar con la detención aunque falle la actualización del estado

        # Cancelar task del worker
        if self.worker_task:
            logger.debug("Cancelando task del worker...")
            self.worker_task.cancel()
            try:
                await asyncio.wait_for(self.worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout al esperar que el worker termine, forzando cancelación")
            except asyncio.CancelledError:
                logger.debug("Worker task cancelado correctamente")
            except Exception as e:
                logger.warning(f"Error al cancelar worker task: {e}", exc_info=True)
        
        logger.info("🛑 Worker detenido correctamente")

    def _check_circuit_breaker(self) -> bool:
        """
        Verificar si el circuit breaker permite procesar tareas.
        
        Returns:
            True si se puede procesar, False si el circuito está abierto
        """
        if self.circuit_state == CircuitState.CLOSED:
            return True
        
        if self.circuit_state == CircuitState.OPEN:
            if self.circuit_open_until and datetime.now() < self.circuit_open_until:
                return False
            # Intentar medio abrir el circuito
            self.circuit_state = CircuitState.HALF_OPEN
            self.consecutive_failures = 0
            logger.info("Circuit breaker: Intentando recuperación (half-open)")
            return True
        
        # HALF_OPEN: permitir intento
        return True
    
    def _record_success(self):
        """Registrar éxito y cerrar el circuit breaker si estaba abierto."""
        if self.circuit_state == CircuitState.HALF_OPEN:
            self.circuit_state = CircuitState.CLOSED
            logger.info("Circuit breaker: Servicio recuperado, circuito cerrado")
        
        self.consecutive_failures = 0
        self.circuit_open_until = None
    
    def _record_failure(self):
        """Registrar fallo y abrir el circuit breaker si es necesario."""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.max_failures:
            self.circuit_state = CircuitState.OPEN
            self.circuit_open_until = datetime.now() + self.circuit_timeout
            logger.warning(
                f"Circuit breaker: Abierto después de {self.consecutive_failures} fallos. "
                f"Reintentando en {self.circuit_timeout.total_seconds()} segundos"
            )
    
    async def _process_task_with_timeout(self, task: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Procesar una tarea con timeout.
        
        Args:
            task: Tarea a procesar
            timeout: Timeout en segundos (default: desde settings)
            
        Returns:
            Resultado de la ejecución
        """
        if timeout is None:
            timeout = settings.TASK_TIMEOUT
        
        try:
            return await asyncio.wait_for(
                self.task_processor.execute_task(task),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            error_msg = f"Tarea {task['id']} excedió el timeout de {timeout} segundos"
            logger.error(error_msg)
            await self.storage.update_task_status(
                task["id"],
                TaskStatus.FAILED,
                error=error_msg
            )
            return {
                "success": False,
                "task_id": task["id"],
                "error": error_msg,
                "error_type": "TimeoutError"
            }
    
    async def _worker_loop(self):
        """Loop principal del worker con circuit breaker y mejor manejo de errores."""
        logger.info("🔄 Iniciando loop de worker...")
        
        backoff_seconds = 1  # Backoff exponencial inicial

        while self.is_running:
            try:
                # Verificar circuit breaker
                if not self._check_circuit_breaker():
                    wait_seconds = (self.circuit_open_until - datetime.now()).total_seconds()
                    if wait_seconds > 0:
                        logger.info(f"Circuit breaker abierto, esperando {wait_seconds:.1f} segundos...")
                        await asyncio.sleep(min(wait_seconds, settings.TASK_POLL_INTERVAL))
                    continue
                
                # Resetear backoff en caso de éxito
                backoff_seconds = 1
                
                state = await self.storage.get_agent_state()
                if not state.get("is_running", False):
                    logger.info("Agente pausado, esperando...")
                    await asyncio.sleep(settings.TASK_POLL_INTERVAL)
                    continue

                pending_tasks = await self.storage.get_pending_tasks(order_asc=True)
                
                if pending_tasks:
                    task = pending_tasks[0]
                    task_start_time = datetime.now()
                    logger.info(f"Procesando tarea {task['id']}")

                    await self.storage.update_agent_state({
                        "id": "main",
                        "is_running": True,
                        "current_task_id": task["id"],
                        "last_activity": datetime.now().isoformat(),
                        "metadata": {}
                    })

                    try:
                        if self.task_processor:
                            result = await self._process_task_with_timeout(task)
                            task_duration = (datetime.now() - task_start_time).total_seconds()
                            
                            # Actualizar métricas de duración
                            self._update_average_duration(task_duration)
                            
                            if result.get('success', False):
                                self._record_success()
                                self.metrics["tasks_succeeded"] += 1
                                logger.info(
                                    f"✅ Tarea {task['id']} completada exitosamente "
                                    f"en {task_duration:.2f}s"
                                )
                            else:
                                self._record_failure()
                                self.metrics["tasks_failed"] += 1
                                error_msg = result.get('error', 'Unknown error')
                                logger.warning(
                                    f"❌ Tarea {task['id']} falló después de {task_duration:.2f}s: {error_msg}"
                                )
                        else:
                            error_msg = ErrorMessages.TASK_PROCESSOR_NOT_INITIALIZED
                            logger.error(f"Tarea {task['id']}: {error_msg}")
                            await self.storage.update_task_status(
                                task["id"],
                                TaskStatus.FAILED,
                                error=error_msg
                            )
                            self._record_failure()
                            self.metrics["tasks_failed"] += 1
                        
                        # Actualizar métricas
                        self.metrics["tasks_processed"] += 1
                        self.metrics["last_task_time"] = datetime.now().isoformat()
                        
                    except TaskProcessingError as e:
                        self._record_failure()
                        self.metrics["tasks_failed"] += 1
                        logger.error(f"Error al procesar tarea {task['id']}: {e}", exc_info=True)
                        await self.storage.update_task_status(
                            task["id"],
                            TaskStatus.FAILED,
                            error=str(e)
                        )
                    except StorageError as e:
                        self._record_failure()
                        logger.error(f"Error de almacenamiento al procesar tarea {task['id']}: {e}", exc_info=True)
                        # Esperar más tiempo si hay problemas de almacenamiento
                        await asyncio.sleep(backoff_seconds)
                        backoff_seconds = min(backoff_seconds * 2, 60)  # Max 60 segundos
                    except Exception as e:
                        self._record_failure()
                        self.metrics["tasks_failed"] += 1
                        logger.error(f"Error inesperado al procesar tarea {task['id']}: {e}", exc_info=True)
                        await self.storage.update_task_status(
                            task["id"],
                            TaskStatus.FAILED,
                            error=f"Error inesperado: {str(e)}"
                        )

                    await self.storage.update_agent_state({
                        "id": "main",
                        "is_running": True,
                        "current_task_id": None,
                        "last_activity": datetime.now().isoformat(),
                        "metadata": {}
                    })
                else:
                    await asyncio.sleep(settings.TASK_POLL_INTERVAL)

            except asyncio.CancelledError:
                logger.info("Worker cancelado")
                break
            except Exception as e:
                logger.error(f"Error crítico en worker loop: {e}", exc_info=True)
                self._record_failure()
                await asyncio.sleep(min(backoff_seconds, settings.TASK_POLL_INTERVAL))
                backoff_seconds = min(backoff_seconds * 2, 60)

        logger.info("Loop de worker finalizado")
    
    def _update_average_duration(self, duration: float):
        """
        Actualizar la duración promedio de tareas.
        
        Args:
            duration: Duración de la tarea en segundos
        """
        total_tasks = self.metrics["tasks_processed"]
        current_avg = self.metrics["average_task_duration"]
        
        if total_tasks == 0:
            self.metrics["average_task_duration"] = duration
        else:
            # Media móvil exponencial para calcular promedio
            alpha = 0.1  # Factor de suavizado
            self.metrics["average_task_duration"] = (
                alpha * duration + (1 - alpha) * current_avg
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas del worker.
        
        Returns:
            Diccionario con métricas incluyendo:
            - tasks_processed: Total de tareas procesadas
            - tasks_succeeded: Tareas exitosas
            - tasks_failed: Tareas fallidas
            - last_task_time: Timestamp de última tarea
            - average_task_duration: Duración promedio en segundos
            - circuit_state: Estado del circuit breaker
            - consecutive_failures: Fallos consecutivos
            - is_running: Si el worker está corriendo
        """
        return {
            **self.metrics,
            "circuit_state": self.circuit_state.value,
            "consecutive_failures": self.consecutive_failures,
            "is_running": self.is_running,
            "success_rate": (
                self.metrics["tasks_succeeded"] / self.metrics["tasks_processed"]
                if self.metrics["tasks_processed"] > 0
                else 0.0
            )
        }

