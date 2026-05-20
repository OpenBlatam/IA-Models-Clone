"""
Sistema de cola de tareas
"""

import json
import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import redis
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings

logger = logging.getLogger(__name__)


class TaskQueue:
    """Cola de tareas usando Redis"""
    
    def __init__(self):
        """Inicializar cola de tareas"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Conectado a Redis")
        except Exception as e:
            logger.warning(f"No se pudo conectar a Redis: {e}. Usando cola en memoria.")
            self.redis_client = None
            self._memory_queue: List[Dict[str, Any]] = []
            self._memory_tasks: Dict[str, Dict[str, Any]] = {}
    
    def _serialize_task(self, task: Dict[str, Any]) -> str:
        """Serializar tarea a JSON"""
        task_copy = task.copy()
        if "created_at" in task_copy and isinstance(task_copy["created_at"], datetime):
            task_copy["created_at"] = task_copy["created_at"].isoformat()
        if "started_at" in task_copy and isinstance(task_copy["started_at"], datetime):
            task_copy["started_at"] = task_copy["started_at"].isoformat()
        if "completed_at" in task_copy and isinstance(task_copy["completed_at"], datetime):
            task_copy["completed_at"] = task_copy["completed_at"].isoformat()
        return json.dumps(task_copy)
    
    def _deserialize_task(self, task_str: str) -> Dict[str, Any]:
        """Deserializar tarea desde JSON"""
        task = json.loads(task_str)
        for date_field in ["created_at", "started_at", "completed_at"]:
            if date_field in task and task[date_field]:
                task[date_field] = datetime.fromisoformat(task[date_field])
        return task
    
    def add_task(self, task_data: Dict[str, Any]) -> str:
        """
        Agregar tarea a la cola
        
        Args:
            task_data: Datos de la tarea
            
        Returns:
            ID de la tarea
        """
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "status": "pending",
            "created_at": datetime.now(),
            **task_data
        }
        
        if self.redis_client:
            self.redis_client.set(f"task:{task_id}", self._serialize_task(task))
            priority = task_data.get("priority", "medium")
            priority_score = {"low": 1, "medium": 2, "high": 3, "urgent": 4}.get(priority, 2)
            self.redis_client.zadd("task_queue", {task_id: priority_score})
        else:
            self._memory_tasks[task_id] = task
            self._memory_queue.append(task_id)
        
        logger.info(f"Tarea agregada: {task_id}")
        return task_id
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Obtener siguiente tarea de la cola
        
        Returns:
            Tarea o None si no hay tareas
        """
        if self.redis_client:
            task_ids = self.redis_client.zrevrange("task_queue", 0, 0)
            if not task_ids:
                return None
            
            task_id = task_ids[0]
            task_str = self.redis_client.get(f"task:{task_id}")
            if not task_str:
                return None
            
            task = self._deserialize_task(task_str)
            if task["status"] == "pending":
                return task
        else:
            for task_id in self._memory_queue:
                task = self._memory_tasks.get(task_id)
                if task and task["status"] == "pending":
                    return task
        
        return None
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualizar tarea
        
        Args:
            task_id: ID de la tarea
            updates: Actualizaciones a aplicar
            
        Returns:
            True si se actualizó, False si no existe
        """
        if self.redis_client:
            task_str = self.redis_client.get(f"task:{task_id}")
            if not task_str:
                return False
            
            task = self._deserialize_task(task_str)
            task.update(updates)
            self.redis_client.set(f"task:{task_id}", self._serialize_task(task))
            
            if updates.get("status") in ["completed", "failed", "cancelled"]:
                self.redis_client.zrem("task_queue", task_id)
        else:
            if task_id not in self._memory_tasks:
                return False
            
            self._memory_tasks[task_id].update(updates)
            if updates.get("status") in ["completed", "failed", "cancelled"]:
                if task_id in self._memory_queue:
                    self._memory_queue.remove(task_id)
        
        return True
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener tarea por ID
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Tarea o None si no existe
        """
        if self.redis_client:
            task_str = self.redis_client.get(f"task:{task_id}")
            if not task_str:
                return None
            return self._deserialize_task(task_str)
        else:
            return self._memory_tasks.get(task_id)
    
    def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Listar tareas
        
        Args:
            status: Filtrar por estado
            limit: Límite de resultados
            
        Returns:
            Lista de tareas
        """
        if self.redis_client:
            if status:
                all_tasks = []
                for task_id in self.redis_client.zrevrange("task_queue", 0, -1):
                    task_str = self.redis_client.get(f"task:{task_id}")
                    if task_str:
                        task = self._deserialize_task(task_str)
                        if task["status"] == status:
                            all_tasks.append(task)
                return all_tasks[:limit]
            else:
                all_tasks = []
                for task_id in self.redis_client.zrevrange("task_queue", 0, limit - 1):
                    task_str = self.redis_client.get(f"task:{task_id}")
                    if task_str:
                        all_tasks.append(self._deserialize_task(task_str))
                return all_tasks
        else:
            tasks = list(self._memory_tasks.values())
            if status:
                tasks = [t for t in tasks if t["status"] == status]
            return tasks[:limit]
    
    def remove_task(self, task_id: str) -> bool:
        """
        Eliminar tarea
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            True si se eliminó, False si no existe
        """
        if self.redis_client:
            deleted = bool(self.redis_client.delete(f"task:{task_id}"))
            self.redis_client.zrem("task_queue", task_id)
            return deleted
        else:
            if task_id in self._memory_tasks:
                del self._memory_tasks[task_id]
                if task_id in self._memory_queue:
                    self._memory_queue.remove(task_id)
                return True
            return False

