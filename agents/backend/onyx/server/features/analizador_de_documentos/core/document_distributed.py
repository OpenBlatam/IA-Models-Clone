"""
Document Distributed - Procesamiento Distribuido
================================================

Sistema de procesamiento distribuido para análisis masivo.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)


@dataclass
class DistributedTask:
    """Tarea distribuida."""
    task_id: str
    document_id: str
    content: str
    task_type: str
    status: str = "pending"  # 'pending', 'processing', 'completed', 'failed'
    result: Optional[Any] = None
    error: Optional[str] = None
    assigned_worker: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class DistributedResult:
    """Resultado de procesamiento distribuido."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    processing_time: float
    results: List[Any] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


class DistributedProcessor:
    """Procesador distribuido."""
    
    def __init__(self, analyzer, max_workers: Optional[int] = None):
        """Inicializar procesador."""
        self.analyzer = analyzer
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.tasks: Dict[str, DistributedTask] = {}
        self.worker_pool: Optional[ThreadPoolExecutor] = None
    
    async def process_distributed(
        self,
        documents: List[Dict[str, Any]],
        task_processor: Callable,
        chunk_size: int = 100
    ) -> DistributedResult:
        """
        Procesar documentos de forma distribuida.
        
        Args:
            documents: Lista de documentos
            task_processor: Función procesadora
            chunk_size: Tamaño de chunk
        
        Returns:
            DistributedResult con resultados
        """
        start_time = datetime.now()
        
        # Crear tareas
        tasks = []
        for i, doc in enumerate(documents):
            task = DistributedTask(
                task_id=f"task_{i}",
                document_id=doc.get("id", f"doc_{i}"),
                content=doc.get("content", ""),
                task_type=doc.get("type", "analysis")
            )
            tasks.append(task)
            self.tasks[task.task_id] = task
        
        # Procesar en chunks
        results = []
        errors = []
        
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            
            # Procesar chunk en paralelo
            chunk_results = await self._process_chunk(chunk, task_processor)
            
            results.extend(chunk_results["results"])
            errors.extend(chunk_results["errors"])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DistributedResult(
            total_tasks=len(tasks),
            completed_tasks=len(results),
            failed_tasks=len(errors),
            processing_time=processing_time,
            results=results,
            errors=errors
        )
    
    async def _process_chunk(
        self,
        chunk: List[DistributedTask],
        processor: Callable
    ) -> Dict[str, List]:
        """Procesar chunk de tareas."""
        results = []
        errors = []
        
        # Crear tareas async
        async_tasks = []
        for task in chunk:
            task.status = "processing"
            task.started_at = datetime.now()
            
            async_task = self._process_single_task(task, processor)
            async_tasks.append(async_task)
        
        # Ejecutar en paralelo
        chunk_results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        for i, result in enumerate(chunk_results):
            task = chunk[i]
            
            if isinstance(result, Exception):
                task.status = "failed"
                task.error = str(result)
                errors.append({
                    "task_id": task.task_id,
                    "error": str(result)
                })
            else:
                task.status = "completed"
                task.result = result
                task.completed_at = datetime.now()
                results.append(result)
        
        return {"results": results, "errors": errors}
    
    async def _process_single_task(
        self,
        task: DistributedTask,
        processor: Callable
    ) -> Any:
        """Procesar tarea individual."""
        try:
            if asyncio.iscoroutinefunction(processor):
                result = await processor(task.content, task.document_id)
            else:
                result = processor(task.content, task.document_id)
            
            return result
        except Exception as e:
            logger.error(f"Error procesando tarea {task.task_id}: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Obtener estado de tarea."""
        return self.tasks.get(task_id)
    
    def get_all_tasks_status(self) -> Dict[str, str]:
        """Obtener estado de todas las tareas."""
        return {
            task_id: task.status
            for task_id, task in self.tasks.items()
        }


__all__ = [
    "DistributedProcessor",
    "DistributedTask",
    "DistributedResult"
]


