"""
Task Queue
==========

Sistema de cola de tareas para el agente.
"""

import asyncio
import logging
import json
import aiosqlite
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)


class TaskQueue:
    """Cola de tareas persistente."""
    
    def __init__(self):
        self.db_path = Path(settings.TASKS_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        
    async def initialize(self) -> None:
        """Inicializar la base de datos."""
        if self._initialized:
            return
            
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    repository TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    metadata TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT
                )
            """)
            await db.commit()
            
        self._initialized = True
        logger.info("✅ Cola de tareas inicializada")
        
    async def add_task(
        self,
        repository: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Agregar una nueva tarea."""
        import uuid
        task_id = str(uuid.uuid4())
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO tasks (id, repository, instruction, metadata, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                repository,
                instruction,
                json.dumps(metadata or {}),
                "pending",
                datetime.utcnow().isoformat()
            ))
            await db.commit()
            
        logger.info(f"Tarea {task_id} agregada a la cola")
        return task_id
        
    async def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Obtener la siguiente tarea pendiente."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM tasks
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
            """) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    task = dict(row)
                    task_id = task["id"]
                    
                    await db.execute("""
                        UPDATE tasks
                        SET status = 'running', started_at = ?
                        WHERE id = ?
                    """, (datetime.utcnow().isoformat(), task_id))
                    await db.commit()
                    
                    task["metadata"] = json.loads(task["metadata"] or "{}")
                    return task
                    
        return None
        
    async def complete_task(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Marcar una tarea como completada."""
        async with aiosqlite.connect(self.db_path) as db:
            status = "completed" if not error else "failed"
            
            await db.execute("""
                UPDATE tasks
                SET status = ?, completed_at = ?, result = ?, error = ?
                WHERE id = ?
            """, (
                status,
                datetime.utcnow().isoformat(),
                json.dumps(result or {}),
                error
            ))
            await db.commit()
            
    async def get_status(self) -> Dict[str, Any]:
        """Obtener el estado de la cola."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT status, COUNT(*) as count
                FROM tasks
                GROUP BY status
            """) as cursor:
                rows = await cursor.fetchall()
                
                status_counts = {row[0]: row[1] for row in rows}
                
                return {
                    "pending": status_counts.get("pending", 0),
                    "running": status_counts.get("running", 0),
                    "completed": status_counts.get("completed", 0),
                    "failed": status_counts.get("failed", 0),
                }




