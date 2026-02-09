"""
Task Executor - Ejecutor de tareas
===================================

Ejecuta comandos recibidos desde Cursor con manejo robusto de errores,
timeouts y soporte para múltiples tipos de comandos.
"""

import asyncio
import logging
from enum import Enum
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime

from .domain.exceptions import TaskExecutionException, TaskTimeoutException
from .infrastructure.monitoring.observability import observe_async

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Estados de una tarea"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Resultado de ejecución de tarea"""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class TaskExecutor:
    """
    Ejecutor de tareas con soporte para timeouts y cancelación.
    
    Attributes:
        timeout: Tiempo máximo de ejecución en segundos (default: 300.0)
        active_executions: Diccionario de tareas activas por task_id
    """
    
    def __init__(self, timeout: float = 300.0) -> None:
        """
        Inicializar ejecutor de tareas.
        
        Args:
            timeout: Tiempo máximo de ejecución en segundos
        """
        self.timeout = timeout
        self.active_executions: Dict[str, asyncio.Task] = {}
        
    @observe_async(operation_name="task_execution", log_args=False, track_metrics=True)
    async def execute(self, command: str, task_id: str) -> ExecutionResult:
        """
        Ejecutar un comando con timeout y manejo de errores.
        
        Args:
            command: Comando a ejecutar
            task_id: Identificador único de la tarea
            
        Returns:
            ExecutionResult con el resultado de la ejecución
            
        Raises:
            asyncio.TimeoutError: Si la ejecución excede el timeout
            Exception: Si ocurre un error durante la ejecución
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"🔨 Executing command: {command[:100]}...")
            
            # Ejecutar comando con timeout
            result = await asyncio.wait_for(
                self._run_command(command),
                timeout=self.timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=True,
                output=result,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Task timeout after {self.timeout}s"
            logger.error(f"⏱️ {error_msg}")
            
            # Lanzar excepción específica para mejor manejo
            raise TaskTimeoutException(
                timeout=self.timeout,
                task_id=task_id,
                command=command
            ) from None
            
        except TaskExecutionException:
            # Re-lanzar excepciones específicas
            raise
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"❌ Execution error: {error_msg}")
            
            # Convertir a excepción específica
            raise TaskExecutionException(
                message=error_msg,
                task_id=task_id,
                command=command
            ) from e
            
    async def _run_command(self, command: str) -> str:
        """Ejecutar comando real"""
        from .command_executor import CommandExecutor
        
        executor = CommandExecutor(timeout=self.timeout)
        result = await executor.execute(command)
        
        if result["success"]:
            return result["output"]
        else:
            raise Exception(result.get("error", "Unknown error"))
        
    async def cancel(self, task_id: str) -> bool:
        """
        Cancelar una tarea en ejecución.
        
        Args:
            task_id: Identificador de la tarea a cancelar
            
        Returns:
            True si la tarea fue cancelada, False si no se encontró
        """
        if task_id in self.active_executions:
            task = self.active_executions[task_id]
            task.cancel()
            del self.active_executions[task_id]
            logger.info(f"🚫 Task cancelled: {task_id}")
            return True
        return False

