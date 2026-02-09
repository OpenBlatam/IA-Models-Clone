"""
GitHub Autonomous Agent
=======================

Agente principal que ejecuta tareas en repositorios de GitHub.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from ..config.settings import settings
from .github_client import GitHubClient
from .task_executor import TaskExecutor
from .task_queue import TaskQueue

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Estado del agente."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class GitHubAutonomousAgent:
    """Agente autónomo para GitHub."""
    
    def __init__(self):
        self.status = AgentStatus.IDLE
        self.github_client = GitHubClient(token=settings.GITHUB_TOKEN)
        self.task_queue = TaskQueue()
        self.task_executor = TaskExecutor(
            github_client=self.github_client,
            max_concurrent=settings.AGENT_MAX_CONCURRENT_TASKS
        )
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._stop_event = asyncio.Event()
        
    async def start(self) -> None:
        """Iniciar el agente."""
        if self.status == AgentStatus.RUNNING:
            logger.warning("El agente ya está en ejecución")
            return
            
        logger.info("🚀 Iniciando GitHub Autonomous Agent...")
        self.status = AgentStatus.RUNNING
        self._stop_event.clear()
        
        await self.task_queue.initialize()
        await self._run_loop()
        
    async def stop(self) -> None:
        """Detener el agente."""
        logger.info("⏹️  Deteniendo GitHub Autonomous Agent...")
        self.status = AgentStatus.STOPPED
        self._stop_event.set()
        
        for task_id, task in self.running_tasks.items():
            if not task.done():
                logger.info(f"Cancelando tarea {task_id}...")
                task.cancel()
                
        await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        self.running_tasks.clear()
        
    async def pause(self) -> None:
        """Pausar el agente."""
        logger.info("⏸️  Pausando GitHub Autonomous Agent...")
        self.status = AgentStatus.PAUSED
        
    async def resume(self) -> None:
        """Reanudar el agente."""
        logger.info("▶️  Reanudando GitHub Autonomous Agent...")
        self.status = AgentStatus.RUNNING
        
    async def add_task(
        self,
        repository: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Agregar una nueva tarea a la cola."""
        task_id = await self.task_queue.add_task(
            repository=repository,
            instruction=instruction,
            metadata=metadata or {}
        )
        logger.info(f"✅ Tarea {task_id} agregada a la cola")
        return task_id
        
    async def get_status(self) -> Dict[str, Any]:
        """Obtener el estado actual del agente."""
        queue_status = await self.task_queue.get_status()
        
        return {
            "status": self.status.value,
            "running_tasks": len(self.running_tasks),
            "queue": queue_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def _run_loop(self) -> None:
        """Loop principal del agente. Solo se detiene cuando se llama explícitamente a stop()."""
        logger.info("🔄 Iniciando loop principal del agente...")
        logger.info("⚠️  El agente NO se detendrá automáticamente. Solo se detendrá cuando presiones el botón de parar.")
        
        # El loop continúa indefinidamente hasta que _stop_event sea establecido explícitamente
        while not self._stop_event.is_set():
            try:
                # Si el agente está pausado, esperar pero no detenerse
                if self.status == AgentStatus.PAUSED:
                    logger.debug("Agente pausado, esperando...")
                    await asyncio.sleep(settings.AGENT_POLL_INTERVAL)
                    continue
                
                # Si no está en estado RUNNING, intentar reanudar automáticamente
                if self.status != AgentStatus.RUNNING:
                    if self.status == AgentStatus.STOPPED:
                        # Si está detenido, salir del loop (solo cuando se llama stop())
                        logger.info("Agente detenido por el usuario")
                        break
                    # Para otros estados (IDLE, ERROR), intentar reanudar
                    logger.warning(f"Estado inesperado: {self.status}, reanudando...")
                    self.status = AgentStatus.RUNNING
                    
                if len(self.running_tasks) >= settings.AGENT_MAX_CONCURRENT_TASKS:
                    await asyncio.sleep(settings.AGENT_POLL_INTERVAL)
                    continue
                    
                task = await self.task_queue.get_next_task()
                
                if task:
                    logger.info(f"📋 Procesando tarea {task['id']}...")
                    task_coro = self.task_executor.execute_task(task)
                    task_obj = asyncio.create_task(task_coro)
                    self.running_tasks[task['id']] = task_obj
                    
                    task_obj.add_done_callback(
                        lambda t, task_id=task['id']: self._on_task_done(task_id, t)
                    )
                else:
                    await asyncio.sleep(settings.AGENT_POLL_INTERVAL)
                    
            except asyncio.CancelledError:
                # Si se cancela pero no se ha llamado stop(), continuar
                if not self._stop_event.is_set():
                    logger.warning("Loop cancelado pero continuando (solo se detiene con stop())")
                    await asyncio.sleep(settings.AGENT_POLL_INTERVAL)
                    continue
                else:
                    logger.info("Loop principal detenido por el usuario")
                    break
            except Exception as e:
                # En caso de error, loguear pero continuar (no detenerse)
                logger.error(f"Error en el loop principal: {e}", exc_info=True)
                logger.warning("El agente continuará ejecutándose a pesar del error")
                await asyncio.sleep(settings.AGENT_POLL_INTERVAL)
                # Intentar restaurar el estado a RUNNING si estaba en ERROR
                if self.status == AgentStatus.ERROR:
                    self.status = AgentStatus.RUNNING
                    logger.info("Estado restaurado a RUNNING después del error")
                
    def _on_task_done(self, task_id: str, task: asyncio.Task) -> None:
        """Callback cuando una tarea termina."""
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
            
        if task.cancelled():
            logger.info(f"Tarea {task_id} cancelada")
        elif task.exception():
            logger.error(f"Error en tarea {task_id}: {task.exception()}")
        else:
            logger.info(f"✅ Tarea {task_id} completada")

