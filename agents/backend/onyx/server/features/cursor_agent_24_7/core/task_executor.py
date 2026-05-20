"""
Task Executor - Ejecutor de tareas
===================================

Ejecuta comandos recibidos desde Cursor con manejo de timeouts,
cancelación y gestión de errores.
"""

import asyncio
import logging
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

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
    """
    Resultado de ejecución de tarea.
    
    Attributes:
        success: True si la ejecución fue exitosa.
        output: Salida del comando (si fue exitoso).
        error: Mensaje de error (si falló).
        execution_time: Tiempo de ejecución en segundos.
    """
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class TaskExecutor:
    """
    Ejecutor de tareas con soporte para timeouts y cancelación.
    
    Gestiona la ejecución de comandos con control de tiempo y capacidad
    de cancelar tareas en ejecución.
    """
    
    def __init__(self, timeout: float = 300.0) -> None:
        """
        Inicializar ejecutor de tareas.
        
        Args:
            timeout: Timeout en segundos para cada tarea (default: 300.0).
        
        Raises:
            ValueError: Si el timeout es inválido.
        """
        if timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {timeout}")
        
        self.timeout: float = timeout
        self.active_executions: Dict[str, asyncio.Task] = {}
    
    async def execute(self, command: str, task_id: str) -> ExecutionResult:
        """
        Ejecutar un comando con timeout.
        
        Args:
            command: Comando a ejecutar.
            task_id: ID único de la tarea.
        
        Returns:
            ExecutionResult con el resultado de la ejecución.
        
        Raises:
            ValueError: Si el comando o task_id están vacíos.
        """
        if not command or not command.strip():
            raise ValueError("Command cannot be empty")
        if not task_id or not task_id.strip():
            raise ValueError("Task ID cannot be empty")
        
        start_time = datetime.now()
        
        try:
            logger.info(f"🔨 Executing command: {command[:100]}...")
            
            exec_task = asyncio.create_task(self._run_command(command))
            self.active_executions[task_id] = exec_task
            
            result = await asyncio.wait_for(exec_task, timeout=self.timeout)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if task_id in self.active_executions:
                del self.active_executions[task_id]
            
            return ExecutionResult(
                success=True,
                output=result,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Task timeout after {self.timeout}s"
            logger.error(f"⏱️ {error_msg}")
            
            if task_id in self.active_executions:
                self.active_executions[task_id].cancel()
                del self.active_executions[task_id]
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
            
        except asyncio.CancelledError:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = "Task was cancelled"
            logger.warning(f"🚫 {error_msg}")
            
            if task_id in self.active_executions:
                del self.active_executions[task_id]
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"❌ Execution error: {error_msg}", exc_info=True)
            
            if task_id in self.active_executions:
                del self.active_executions[task_id]
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _run_command(self, command: str) -> str:
        """
        Ejecutar comando real usando CommandExecutor.
        
        Args:
            command: Comando a ejecutar.
        
        Returns:
            Salida del comando.
        
        Raises:
            RuntimeError: Si la ejecución falla.
        """
        try:
            from .command_executor import CommandExecutor
        except ImportError as e:
            logger.error(f"Failed to import CommandExecutor: {e}")
            raise RuntimeError(f"CommandExecutor not available: {e}") from e
        
        executor = CommandExecutor(timeout=self.timeout)
        result = await executor.execute(command)
        
        if not isinstance(result, dict):
            raise RuntimeError(f"Invalid result format from CommandExecutor: {type(result)}")
        
        if result.get("success", False):
            output = result.get("output", "")
            if output is None:
                return ""
            return str(output)
        else:
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Command execution failed: {error}")
    
    async def cancel(self, task_id: str) -> bool:
        """
        Cancelar una tarea en ejecución.
        
        Args:
            task_id: ID de la tarea a cancelar.
        
        Returns:
            True si la tarea fue cancelada, False si no se encontró.
        """
        if not task_id or not task_id.strip():
            logger.warning("Attempted to cancel task with empty task_id")
            return False
        
        if task_id not in self.active_executions:
            logger.debug(f"Task {task_id} not found in active executions")
            return False
        
        try:
            task = self.active_executions[task_id]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self.active_executions[task_id]
            logger.info(f"🚫 Task cancelled: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}", exc_info=True)
            if task_id in self.active_executions:
                del self.active_executions[task_id]
            return False

