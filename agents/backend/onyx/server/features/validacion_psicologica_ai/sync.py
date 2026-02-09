"""
Sistema de Sincronización
========================
Sincronización de datos entre sistemas
"""

from typing import Dict, Any, List, Optional, Callable
from uuid import UUID
from datetime import datetime, timedelta
from enum import Enum
import structlog
import asyncio

logger = structlog.get_logger()


class SyncStatus(str, Enum):
    """Estado de sincronización"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SyncTask:
    """Tarea de sincronización"""
    
    def __init__(
        self,
        id: UUID,
        source: str,
        target: str,
        data_type: str,
        sync_type: str = "full"
    ):
        self.id = id
        self.source = source
        self.target = target
        self.data_type = data_type
        self.sync_type = sync_type
        self.status = SyncStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.records_synced: int = 0


class SyncManager:
    """Gestor de sincronización"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._tasks: Dict[UUID, SyncTask] = {}
        self._sync_handlers: Dict[str, Callable] = {}
        logger.info("SyncManager initialized")
    
    def register_sync_handler(
        self,
        data_type: str,
        handler: Callable
    ) -> None:
        """
        Registrar handler de sincronización
        
        Args:
            data_type: Tipo de dato
            handler: Función handler
        """
        self._sync_handlers[data_type] = handler
        logger.info("Sync handler registered", data_type=data_type)
    
    async def sync(
        self,
        source: str,
        target: str,
        data_type: str,
        sync_type: str = "full"
    ) -> UUID:
        """
        Iniciar sincronización
        
        Args:
            source: Fuente de datos
            target: Destino de datos
            data_type: Tipo de dato
            sync_type: Tipo de sincronización (full, incremental)
            
        Returns:
            ID de tarea de sincronización
        """
        from uuid import uuid4
        
        task = SyncTask(
            id=uuid4(),
            source=source,
            target=target,
            data_type=data_type,
            sync_type=sync_type
        )
        
        self._tasks[task.id] = task
        
        # Ejecutar sincronización en background
        asyncio.create_task(self._execute_sync(task))
        
        logger.info("Sync task created", task_id=str(task.id), data_type=data_type)
        
        return task.id
    
    async def _execute_sync(self, task: SyncTask) -> None:
        """Ejecutar sincronización"""
        task.status = SyncStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        try:
            handler = self._sync_handlers.get(task.data_type)
            if not handler:
                raise ValueError(f"No handler for data type: {task.data_type}")
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task.source, task.target, task.sync_type)
            else:
                result = handler(task.source, task.target, task.sync_type)
            
            task.records_synced = result.get("records_synced", 0)
            task.status = SyncStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            logger.info(
                "Sync completed",
                task_id=str(task.id),
                records_synced=task.records_synced
            )
            
        except Exception as e:
            task.status = SyncStatus.FAILED
            task.error = str(e)
            logger.error("Sync failed", task_id=str(task.id), error=str(e))
    
    def get_sync_status(self, task_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Obtener estado de sincronización
        
        Args:
            task_id: ID de tarea
            
        Returns:
            Estado de sincronización
        """
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": str(task.id),
            "source": task.source,
            "target": task.target,
            "data_type": task.data_type,
            "sync_type": task.sync_type,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "records_synced": task.records_synced,
            "error": task.error
        }


# Instancia global del gestor de sincronización
sync_manager = SyncManager()

