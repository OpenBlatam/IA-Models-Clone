"""
Task Queue - Sistema de colas para procesamiento asíncrono
===========================================================
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from enum import Enum
import uuid
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Estados de tarea"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskQueue:
    """
    Sistema de colas para procesamiento asíncrono de tareas.
    """
    
    def __init__(self, max_workers: int = 4, queue_dir: str = "data/queue"):
        """
        Inicializar cola de tareas.
        
        Args:
            max_workers: Número máximo de workers
            queue_dir: Directorio para almacenar tareas
        """
        self.max_workers = max_workers
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.workers: List[asyncio.Task] = []
        self.running = False
    
    async def start(self):
        """Inicia los workers"""
        if self.running:
            return
        
        self.running = True
        
        # Iniciar workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Task queue iniciada con {self.max_workers} workers")
    
    async def stop(self):
        """Detiene los workers"""
        self.running = False
        
        # Esperar a que terminen las tareas actuales
        await self.queue.join()
        
        # Cancelar workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Task queue detenida")
    
    async def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """
        Encola una tarea.
        
        Args:
            task_type: Tipo de tarea
            payload: Datos de la tarea
            priority: Prioridad (mayor = más prioritario)
            
        Returns:
            ID de la tarea
        """
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "type": task_type,
            "payload": payload,
            "priority": priority,
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None
        }
        
        self.tasks[task_id] = task
        await self.queue.put((priority, task_id))
        
        # Guardar en disco
        self._save_task(task)
        
        logger.info(f"Tarea encolada: {task_id} ({task_type})")
        
        return task_id
    
    async def _worker(self, worker_name: str):
        """Worker que procesa tareas"""
        logger.info(f"Worker iniciado: {worker_name}")
        
        while self.running:
            try:
                # Obtener tarea de la cola
                priority, task_id = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                task = self.tasks.get(task_id)
                if not task:
                    self.queue.task_done()
                    continue
                
                # Procesar tarea
                await self._process_task(task, worker_name)
                
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error en worker {worker_name}: {e}")
    
    async def _process_task(self, task: Dict[str, Any], worker_name: str):
        """Procesa una tarea individual"""
        task_id = task["id"]
        task_type = task["type"]
        
        try:
            # Actualizar estado
            task["status"] = TaskStatus.PROCESSING.value
            task["started_at"] = datetime.now().isoformat()
            self._save_task(task)
            
            logger.info(f"Procesando tarea {task_id} ({task_type}) en {worker_name}")
            
            # Ejecutar handler según tipo
            result = await self._execute_task(task_type, task["payload"])
            
            # Completar tarea
            task["status"] = TaskStatus.COMPLETED.value
            task["completed_at"] = datetime.now().isoformat()
            task["result"] = result
            
            logger.info(f"Tarea completada: {task_id}")
            
        except Exception as e:
            logger.error(f"Error procesando tarea {task_id}: {e}")
            task["status"] = TaskStatus.FAILED.value
            task["completed_at"] = datetime.now().isoformat()
            task["error"] = str(e)
        
        finally:
            self._save_task(task)
    
    async def _execute_task(self, task_type: str, payload: Dict[str, Any]) -> Any:
        """Ejecuta una tarea según su tipo"""
        # Handlers por tipo de tarea
        handlers = {
            "improve_code": self._handle_improve_code,
            "train_model": self._handle_train_model,
            "batch_process": self._handle_batch_process,
        }
        
        handler = handlers.get(task_type)
        if not handler:
            raise ValueError(f"Tipo de tarea desconocido: {task_type}")
        
        return await handler(payload)
    
    async def _handle_improve_code(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handler para mejora de código"""
        # Implementación básica - en producción usaría CodeImprover real
        return {
            "success": True,
            "message": "Code improvement task completed"
        }
    
    async def _handle_train_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handler para entrenamiento de modelo"""
        return {
            "success": True,
            "message": "Model training task completed"
        }
    
    async def _handle_batch_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handler para procesamiento en lote"""
        return {
            "success": True,
            "message": "Batch processing task completed"
        }
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de una tarea"""
        return self.tasks.get(task_id)
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Lista tareas"""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t["status"] == status.value]
        
        # Ordenar por fecha de creación (más recientes primero)
        tasks.sort(key=lambda x: x["created_at"], reverse=True)
        
        return tasks[:limit]
    
    def _save_task(self, task: Dict[str, Any]):
        """Guarda tarea en disco"""
        try:
            task_file = self.queue_dir / f"{task['id']}.json"
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando tarea: {e}")

