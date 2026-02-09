"""
Storage - Sistema de persistencia para tareas y estado.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiosqlite
from config.settings import settings
from config.logging_config import get_logger
from core.utils import parse_json_field, serialize_json_field
from core.exceptions import StorageError
from core.constants import TaskStatus

logger = get_logger(__name__)


class TaskStorage:
    """Almacenamiento persistente de tareas."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializar almacenamiento.

        Args:
            db_path: Ruta a la base de datos (opcional)
        """
        self.db_path = db_path or settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
        self.storage_path = Path(settings.STORAGE_PATH)
        self.tasks_path = Path(settings.TASKS_STORAGE_PATH)
        self.logs_path = Path(settings.LOGS_STORAGE_PATH)

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.tasks_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

    async def init_db(self):
        """Inicializar base de datos."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    repository_owner TEXT NOT NULL,
                    repository_name TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT,
                    metadata TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS agent_state (
                    id TEXT PRIMARY KEY,
                    is_running INTEGER NOT NULL DEFAULT 0,
                    current_task_id TEXT,
                    last_activity TEXT,
                    metadata TEXT
                )
            """)
            await db.commit()

    async def save_task(self, task: Dict[str, Any]) -> bool:
        """
        Guardar una tarea.

        Args:
            task: Diccionario con información de la tarea

        Returns:
            True si se guardó exitosamente
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO tasks 
                    (id, repository_owner, repository_name, instruction, status,
                     created_at, updated_at, started_at, completed_at, result, error, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.get("id"),
                    task.get("repository_owner"),
                    task.get("repository_name"),
                    task.get("instruction"),
                    task.get("status", TaskStatus.PENDING),
                    task.get("created_at", datetime.now().isoformat()),
                    task.get("updated_at", datetime.now().isoformat()),
                    task.get("started_at"),
                    task.get("completed_at"),
                    serialize_json_field(task.get("result")),
                    task.get("error"),
                    serialize_json_field(task.get("metadata", {}))
                ))
                await db.commit()

            task_file = self.tasks_path / f"{task['id']}.json"
            task_file.write_text(json.dumps(task, indent=2))
            return True
        except Exception as e:
            logger.error(f"Error al guardar tarea: {e}")
            return False

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener una tarea por ID.

        Args:
            task_id: ID de la tarea

        Returns:
            Diccionario con información de la tarea o None
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM tasks WHERE id = ?", (task_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        task = dict(row)
                        task["result"] = parse_json_field(task.get("result"))
                        task["metadata"] = parse_json_field(task.get("metadata"))
                        return task
                    return None
        except Exception as e:
            logger.error(f"Error al obtener tarea: {e}")
            return None

    async def get_pending_tasks(self, order_asc: bool = True) -> List[Dict[str, Any]]:
        """
        Obtener todas las tareas pendientes.

        Args:
            order_asc: Si es True, ordena por fecha de creación ascendente (más antiguas primero).
                       Si es False, ordena descendente (más recientes primero).

        Returns:
            Lista de tareas pendientes
        """
        tasks = await self.get_tasks(status=TaskStatus.PENDING, limit=1000)
        if order_asc:
            tasks.sort(key=lambda x: x.get("created_at", ""))
        return tasks
    
    async def get_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener tareas con filtros opcionales.

        Args:
            status: Estado de las tareas a filtrar (opcional)
            limit: Número máximo de tareas a retornar

        Returns:
            Lista de tareas
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                query = "SELECT * FROM tasks"
                params = []
                
                if status:
                    query += " WHERE status = ?"
                    params.append(status)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    tasks = []
                    for row in rows:
                        task = dict(row)
                        task["result"] = parse_json_field(task.get("result"))
                        task["metadata"] = parse_json_field(task.get("metadata"))
                        tasks.append(task)
                    return tasks
        except Exception as e:
            logger.error(f"Error al obtener tareas: {e}")
            return []

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Actualizar estado de una tarea.

        Args:
            task_id: ID de la tarea
            status: Nuevo estado
            result: Resultado de la tarea (opcional)
            error: Mensaje de error (opcional)

        Returns:
            True si se actualizó exitosamente
        """
        try:
            status_str = status if isinstance(status, str) else str(status)
            
            updates = {
                "status": status_str,
                "updated_at": datetime.now().isoformat()
            }

            if status_str == TaskStatus.RUNNING:
                updates["started_at"] = datetime.now().isoformat()
            elif status_str in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                updates["completed_at"] = datetime.now().isoformat()

            if result:
                updates["result"] = serialize_json_field(result)
            if error:
                updates["error"] = error

            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [task_id]

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"UPDATE tasks SET {set_clause} WHERE id = ?",
                    values
                )
                await db.commit()

            task = await self.get_task(task_id)
            if task:
                task_file = self.tasks_path / f"{task_id}.json"
                task_file.write_text(json.dumps(task, indent=2))

            return True
        except Exception as e:
            logger.error(f"Error al actualizar estado de tarea: {e}")
            return False

    async def get_agent_state(self) -> Dict[str, Any]:
        """
        Obtener estado del agente.

        Returns:
            Diccionario con el estado del agente
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM agent_state WHERE id = 'main'"
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        state = dict(row)
                        state["metadata"] = parse_json_field(state.get("metadata"))
                        state["is_running"] = bool(state["is_running"])
                        return state
                    return {
                        "id": "main",
                        "is_running": False,
                        "current_task_id": None,
                        "last_activity": None,
                        "metadata": {}
                    }
        except Exception as e:
            logger.error(f"Error al obtener estado del agente: {e}")
            return {
                "id": "main",
                "is_running": False,
                "current_task_id": None,
                "last_activity": None,
                "metadata": {}
            }

    async def update_agent_state(self, state: Dict[str, Any]) -> bool:
        """
        Actualizar estado del agente.

        Args:
            state: Diccionario con el nuevo estado

        Returns:
            True si se actualizó exitosamente
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO agent_state 
                    (id, is_running, current_task_id, last_activity, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    state.get("id", "main"),
                    1 if state.get("is_running", False) else 0,
                    state.get("current_task_id"),
                    state.get("last_activity", datetime.now().isoformat()),
                    serialize_json_field(state.get("metadata", {}))
                ))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error al actualizar estado del agente: {e}")
            return False

    async def delete_task(self, task_id: str) -> bool:
        """
        Eliminar una tarea.

        Args:
            task_id: ID de la tarea a eliminar

        Returns:
            True si se eliminó exitosamente, False si no existe
            
        Raises:
            StorageError: Si hay un error al eliminar la tarea
        """
        try:
            # Verificar que la tarea existe
            task = await self.get_task(task_id)
            if not task:
                logger.warning(f"Intento de eliminar tarea inexistente: {task_id}")
                return False
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                await db.commit()
            
            # Eliminar archivo de log asociado si existe
            try:
                log_file = self.logs_path / f"{task_id}.log"
                if log_file.exists():
                    log_file.unlink()
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo de log para tarea {task_id}: {e}")
            
            # Eliminar archivo JSON de tarea si existe
            try:
                task_file = self.tasks_path / f"{task_id}.json"
                if task_file.exists():
                    task_file.unlink()
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo JSON para tarea {task_id}: {e}")
            
            logger.info(f"Tarea {task_id} eliminada exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar tarea {task_id}: {e}", exc_info=True)
            raise StorageError(f"Error al eliminar tarea: {e}", operation="delete_task") from e

    async def save_log(self, task_id: str, log_message: str, level: str = "INFO"):
        """
        Guardar un log.

        Args:
            task_id: ID de la tarea
            log_message: Mensaje del log
            level: Nivel del log
        """
        try:
            log_file = self.logs_path / f"{task_id}.log"
            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] [{level}] {log_message}\n"
            # Usar append mode para agregar logs
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Error al guardar log: {e}")

