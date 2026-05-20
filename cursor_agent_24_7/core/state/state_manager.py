"""
State Manager - Gestión de estado persistente del agente
========================================================

Gestiona el guardado y carga del estado del agente de forma
robusta y eficiente.
"""

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    from ..agent import CursorAgent, Task

# Usar orjson para JSON más rápido
try:
    import orjson as json
    HAS_ORJSON = True
except ImportError:
    import json  # Fallback a json estándar
    HAS_ORJSON = False

logger = logging.getLogger(__name__)


class StateManager:
    """
    Gestor de estado persistente del agente.
    
    Gestiona el guardado y carga del estado del agente,
    incluyendo tareas, configuración y métricas.
    """
    
    def __init__(self, agent: "CursorAgent", state_file: str) -> None:
        """
        Inicializar gestor de estado.
        
        Args:
            agent: Instancia del agente.
            state_file: Ruta al archivo de estado.
        """
        self.agent = agent
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def save(self) -> None:
        """
        Guardar estado del agente.
        
        Guarda el estado actual del agente incluyendo:
        - Estado del agente (status, running)
        - Tareas pendientes y completadas
        - Configuración
        
        Raises:
            RuntimeError: Si hay error al guardar el estado.
        """
        try:
            state: Dict[str, Any] = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "status": self.agent.status.value,
                "running": self.agent.running,
                "config": {
                    "check_interval": self.agent.config.check_interval,
                    "max_concurrent_tasks": self.agent.config.max_concurrent_tasks,
                    "task_timeout": self.agent.config.task_timeout,
                    "auto_restart": self.agent.config.auto_restart,
                    "persistent_storage": self.agent.config.persistent_storage,
                    "storage_path": self.agent.config.storage_path,
                    "command_file": self.agent.config.command_file,
                    "watch_directory": self.agent.config.watch_directory,
                },
                "tasks": self._serialize_tasks(),
            }
            
            # Guardar con orjson si está disponible
            if HAS_ORJSON:
                content = json.dumps(state).decode('utf-8')
            else:
                content = json.dumps(state, indent=2)
            
            # Escribir a archivo de forma atómica
            temp_file = self.state_file.with_suffix('.tmp')
            temp_file.write_text(content, encoding='utf-8')
            temp_file.replace(self.state_file)
            
            logger.debug(f"State saved to {self.state_file}")
        
        except Exception as e:
            logger.error(f"Error saving state: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save state: {e}") from e
    
    async def load(self) -> None:
        """
        Cargar estado del agente.
        
        Carga el estado guardado del agente y restaura:
        - Tareas pendientes
        - Configuración (si aplica)
        
        Si el archivo no existe o hay error, continúa sin estado.
        """
        if not self.state_file.exists():
            logger.debug("No state file found, starting fresh")
            return
        
        try:
            content = self.state_file.read_text(encoding='utf-8')
            
            # Cargar con orjson si está disponible
            if HAS_ORJSON:
                state = json.loads(content)
            else:
                state = json.loads(content)
            
            # Validar versión
            if state.get("version") != "1.0":
                logger.warning(f"State version mismatch: {state.get('version')}")
                return
            
            # Restaurar tareas
            if "tasks" in state:
                self._deserialize_tasks(state["tasks"])
            
            logger.info(f"State loaded from {self.state_file}")
            logger.debug(f"Restored {len(self.agent.tasks)} tasks")
        
        except Exception as e:
            logger.warning(f"Error loading state: {e}", exc_info=True)
            # Continuar sin estado en caso de error
    
    def _serialize_tasks(self) -> list[Dict[str, Any]]:
        """
        Serializar tareas para guardado.
        
        Returns:
            Lista de diccionarios con información de las tareas.
        """
        return [
            {
                "id": task.id,
                "command": task.command,
                "status": task.status,
                "timestamp": task.timestamp.isoformat(),
                "result": task.result,
                "error": task.error,
            }
            for task in self.agent.tasks.values()
            if task.status in ("pending", "running")  # Solo guardar tareas activas
        ]
    
    def _deserialize_tasks(self, tasks_data: list[Dict[str, Any]]) -> None:
        """
        Deserializar tareas desde estado guardado.
        
        Args:
            tasks_data: Lista de diccionarios con información de tareas.
        """
        from ..agent import Task
        
        for task_data in tasks_data:
            try:
                task = Task(
                    id=task_data["id"],
                    command=task_data["command"],
                    timestamp=datetime.fromisoformat(task_data["timestamp"]),
                    status=task_data.get("status", "pending"),
                    result=task_data.get("result"),
                    error=task_data.get("error"),
                )
                self.agent.tasks[task.id] = task
                
                # Solo restaurar tareas pendientes a la cola
                if task.status == "pending":
                    # No usar await aquí, se agregará en el loop
                    self.agent.task_queue.put_nowait(task)
            
            except Exception as e:
                logger.warning(f"Error deserializing task {task_data.get('id')}: {e}")

