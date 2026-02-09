"""
Procesamiento por Lotes para Validación Psicológica AI
=======================================================
Optimización de procesamiento de múltiples validaciones
"""

from typing import List, Dict, Any, Optional, Callable
from uuid import UUID, uuid4
from datetime import datetime
import structlog
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .models import PsychologicalValidation, ValidationStatus
from .service import PsychologicalValidationService

logger = structlog.get_logger()


@dataclass
class BatchJob:
    """Trabajo de procesamiento por lotes"""
    id: UUID
    validation_ids: List[UUID]
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


class BatchProcessor:
    """Procesador por lotes"""
    
    def __init__(
        self,
        service: PsychologicalValidationService,
        max_concurrent: int = 5,
        batch_size: int = 10
    ):
        """
        Inicializar procesador
        
        Args:
            service: Servicio de validación
            max_concurrent: Máximo de validaciones concurrentes
            batch_size: Tamaño de lote por defecto
        """
        self.service = service
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self._jobs: Dict[UUID, BatchJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        logger.info(
            "BatchProcessor initialized",
            max_concurrent=max_concurrent,
            batch_size=batch_size
        )
    
    async def process_batch(
        self,
        validation_ids: List[UUID],
        job_id: Optional[UUID] = None
    ) -> BatchJob:
        """
        Procesar lote de validaciones
        
        Args:
            validation_ids: IDs de validaciones a procesar
            job_id: ID del trabajo (opcional)
            
        Returns:
            Trabajo de procesamiento
        """
        job_id = job_id or uuid4()
        job = BatchJob(
            id=job_id,
            validation_ids=validation_ids,
            status="pending",
            created_at=datetime.utcnow()
        )
        
        self._jobs[job_id] = job
        job.status = "processing"
        job.started_at = datetime.utcnow()
        
        logger.info(
            "Batch processing started",
            job_id=str(job_id),
            count=len(validation_ids)
        )
        
        # Procesar en lotes más pequeños
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_validation(val_id: UUID):
            async with semaphore:
                try:
                    validation = await self.service.run_validation(val_id)
                    return {
                        "validation_id": str(val_id),
                        "status": "success",
                        "confidence_score": (
                            validation.profile.confidence_score
                            if validation.profile else None
                        )
                    }
                except Exception as e:
                    logger.error(
                        "Error processing validation in batch",
                        validation_id=str(val_id),
                        error=str(e)
                    )
                    return {
                        "validation_id": str(val_id),
                        "status": "error",
                        "error": str(e)
                    }
        
        # Procesar todas las validaciones
        tasks = [process_validation(val_id) for val_id in validation_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separar resultados y errores
        for result in results:
            if isinstance(result, Exception):
                job.errors.append({"error": str(result)})
            elif result.get("status") == "error":
                job.errors.append(result)
            else:
                job.results.append(result)
        
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        
        logger.info(
            "Batch processing completed",
            job_id=str(job_id),
            successful=len(job.results),
            failed=len(job.errors)
        )
        
        return job
    
    def get_job(self, job_id: UUID) -> Optional[BatchJob]:
        """
        Obtener trabajo por ID
        
        Args:
            job_id: ID del trabajo
            
        Returns:
            Trabajo o None
        """
        return self._jobs.get(job_id)
    
    def get_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[BatchJob]:
        """
        Obtener trabajos
        
        Args:
            status: Filtrar por estado
            limit: Límite de resultados
            
        Returns:
            Lista de trabajos
        """
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]
    
    def get_job_stats(self, job_id: UUID) -> Dict[str, Any]:
        """
        Obtener estadísticas de un trabajo
        
        Args:
            job_id: ID del trabajo
            
        Returns:
            Estadísticas
        """
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        total = len(job.validation_ids)
        successful = len(job.results)
        failed = len(job.errors)
        
        duration = None
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()
        
        return {
            "job_id": str(job_id),
            "status": job.status,
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "duration_seconds": duration,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }


# Instancia global del procesador por lotes
batch_processor = None  # Se inicializa con el servicio

