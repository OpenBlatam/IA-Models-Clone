"""
Batch Processor - Sistema de procesamiento en lotes avanzado
==============================================================
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchJob:
    """Job de procesamiento en lote"""
    id: str
    items: List[T]
    processor: Callable[[T], R]
    batch_size: int
    created_at: datetime
    status: str = "pending"
    results: List[R] = None
    errors: List[Dict[str, Any]] = None


class AdvancedBatchProcessor:
    """Sistema avanzado de procesamiento en lotes"""
    
    def __init__(self, max_concurrent_batches: int = 5):
        self.max_concurrent_batches = max_concurrent_batches
        self.batches: Dict[str, BatchJob] = {}
        self.active_batches: int = 0
    
    async def process_batch(self, items: List[T], processor: Callable[[T], R],
                           batch_size: int = 10, job_id: Optional[str] = None) -> BatchJob:
        """Procesa un lote de items"""
        from uuid import uuid4
        
        job_id = job_id or str(uuid4())
        
        job = BatchJob(
            id=job_id,
            items=items,
            processor=processor,
            batch_size=batch_size,
            created_at=datetime.now()
        )
        
        self.batches[job_id] = job
        
        # Procesar en sub-lotes
        results = []
        errors = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Esperar si hay demasiados lotes activos
            while self.active_batches >= self.max_concurrent_batches:
                await asyncio.sleep(0.1)
            
            self.active_batches += 1
            job.status = "processing"
            
            try:
                # Procesar sub-lote
                if asyncio.iscoroutinefunction(processor):
                    batch_results = await asyncio.gather(
                        *[processor(item) for item in batch],
                        return_exceptions=True
                    )
                else:
                    batch_results = [processor(item) for item in batch]
                
                # Separar resultados y errores
                for idx, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        errors.append({
                            "item_index": i + idx,
                            "error": str(result),
                            "item": str(batch[idx])
                        })
                    else:
                        results.append(result)
            
            except Exception as e:
                logger.error(f"Error procesando lote: {e}")
                errors.append({
                    "batch_index": i,
                    "error": str(e)
                })
            
            finally:
                self.active_batches -= 1
        
        job.results = results
        job.errors = errors
        job.status = "completed" if not errors else "completed_with_errors"
        
        logger.info(f"Lote {job_id} procesado: {len(results)} exitosos, {len(errors)} errores")
        
        return job
    
    def get_batch_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de un lote"""
        job = self.batches.get(job_id)
        if not job:
            return None
        
        return {
            "id": job.id,
            "status": job.status,
            "total_items": len(job.items),
            "processed_items": len(job.results) if job.results else 0,
            "errors": len(job.errors) if job.errors else 0,
            "created_at": job.created_at.isoformat()
        }
    
    def get_batch_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene resultados de un lote"""
        job = self.batches.get(job_id)
        if not job:
            return None
        
        return {
            "id": job.id,
            "status": job.status,
            "results": job.results,
            "errors": job.errors,
            "summary": {
                "total": len(job.items),
                "successful": len(job.results) if job.results else 0,
                "failed": len(job.errors) if job.errors else 0
            }
        }
    
    def list_batches(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Lista lotes"""
        batches = list(self.batches.values())
        
        if status:
            batches = [b for b in batches if b.status == status]
        
        batches.sort(key=lambda x: x.created_at, reverse=True)
        
        return [self.get_batch_status(b.id) for b in batches[:limit]]




