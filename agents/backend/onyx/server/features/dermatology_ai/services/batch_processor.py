"""
Sistema de procesamiento por lotes
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class BatchStatus(str, Enum):
    """Estado del batch"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BatchJob:
    """Trabajo en batch"""
    id: str
    items: List[Any]
    processor: Callable
    status: BatchStatus = BatchStatus.PENDING
    results: List[Any] = None
    errors: List[Dict] = None
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.results is None:
            self.results = []
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "status": self.status.value,
            "total_items": len(self.items),
            "processed": len(self.results),
            "errors": len(self.errors),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class BatchProcessor:
    """Procesador de lotes"""
    
    def __init__(self, max_workers: int = 5, use_processes: bool = False):
        """
        Inicializa el procesador
        
        Args:
            max_workers: Número máximo de workers
            use_processes: Usar procesos en lugar de threads
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.jobs: Dict[str, BatchJob] = {}
        self.executor = None
    
    async def process_batch(self, items: List[Any], processor: Callable,
                           batch_id: Optional[str] = None) -> str:
        """
        Procesa un lote de items
        
        Args:
            items: Lista de items a procesar
            processor: Función procesadora
            batch_id: ID del batch (opcional)
            
        Returns:
            ID del batch
        """
        import uuid
        batch_id = batch_id or str(uuid.uuid4())
        
        job = BatchJob(
            id=batch_id,
            items=items,
            processor=processor
        )
        
        self.jobs[batch_id] = job
        
        # Procesar en background
        asyncio.create_task(self._process_job(job))
        
        return batch_id
    
    async def _process_job(self, job: BatchJob):
        """Procesa un job"""
        job.status = BatchStatus.PROCESSING
        job.started_at = datetime.now().isoformat()
        
        try:
            if self.use_processes:
                results = await self._process_with_processes(job)
            else:
                results = await self._process_with_threads(job)
            
            job.results = results
            job.status = BatchStatus.COMPLETED if len(job.errors) == 0 else BatchStatus.PARTIAL
            job.completed_at = datetime.now().isoformat()
        
        except Exception as e:
            job.status = BatchStatus.FAILED
            job.errors.append({"error": str(e), "item_index": -1})
            job.completed_at = datetime.now().isoformat()
    
    async def _process_with_threads(self, job: BatchJob) -> List[Any]:
        """Procesa con threads"""
        results = []
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, job.processor, item)
                for item in job.items
            ]
            
            for i, task in enumerate(tasks):
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    job.errors.append({
                        "error": str(e),
                        "item_index": i
                    })
                    results.append(None)
        
        return results
    
    async def _process_with_processes(self, job: BatchJob) -> List[Any]:
        """Procesa con procesos"""
        results = []
        loop = asyncio.get_event_loop()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, job.processor, item)
                for item in job.items
            ]
            
            for i, task in enumerate(tasks):
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    job.errors.append({
                        "error": str(e),
                        "item_index": i
                    })
                    results.append(None)
        
        return results
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchJob]:
        """Obtiene estado de un batch"""
        return self.jobs.get(batch_id)
    
    def get_all_batches(self) -> List[BatchJob]:
        """Obtiene todos los batches"""
        return list(self.jobs.values())






