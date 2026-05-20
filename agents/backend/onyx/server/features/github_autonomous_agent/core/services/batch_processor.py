"""
Procesador de Tareas en Lote con Control de Concurrencia.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict

from config.logging_config import get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """
    Procesador de tareas en lote con mejoras.
    
    Attributes:
        max_concurrent: Máximo de tareas concurrentes
        batch_size: Tamaño de lote
        processor_func: Función para procesar tareas
        semaphore: Semáforo para control de concurrencia
        stats: Estadísticas del procesador
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        batch_size: int = 10,
        processor_func: Optional[Callable] = None
    ):
        """
        Inicializar procesador de lotes con validaciones.
        
        Args:
            max_concurrent: Máximo de tareas concurrentes (debe ser entero positivo)
            batch_size: Tamaño de lote (debe ser entero positivo)
            processor_func: Función para procesar tareas (opcional)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError(f"max_concurrent debe ser un entero positivo, recibido: {max_concurrent}")
        
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size debe ser un entero positivo, recibido: {batch_size}")
        
        if processor_func is not None and not callable(processor_func):
            raise ValueError(f"processor_func debe ser callable si se proporciona, recibido: {type(processor_func)}")
        
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.processor_func = processor_func
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = {
            "total_processed": 0,
            "total_succeeded": 0,
            "total_failed": 0,
            "batches_processed": 0,
            "average_batch_time": 0.0
        }
        
        logger.info(
            f"✅ BatchProcessor inicializado: max_concurrent={max_concurrent}, "
            f"batch_size={batch_size}, processor_func={'configured' if processor_func else 'None'}"
        )
    
    async def process_batch(
        self,
        tasks: List[Dict[str, Any]],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Procesar lote de tareas con validaciones.
        
        Args:
            tasks: Lista de tareas (debe ser lista)
            on_progress: Callback de progreso (completadas, total) (opcional, debe ser callable si se proporciona)
            
        Returns:
            Resultado del procesamiento con estadísticas
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not isinstance(tasks, list):
            raise ValueError(f"tasks debe ser una lista, recibido: {type(tasks)}")
        
        if on_progress is not None:
            if not callable(on_progress):
                raise ValueError(f"on_progress debe ser callable si se proporciona, recibido: {type(on_progress)}")
        
        if not tasks:
            logger.debug("Lista de tareas vacía, retornando resultado vacío")
            return {
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "results": []
            }
        
        logger.info(f"🔄 Iniciando procesamiento de lote: {len(tasks)} tareas")
        
        results = []
        succeeded = 0
        failed = 0
        
        # Procesar en batches
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            batch_start = datetime.now()
            
            # Procesar batch concurrentemente
            batch_results = await asyncio.gather(
                *[self._process_task(task) for task in batch],
                return_exceptions=True
            )
            
            batch_duration = (datetime.now() - batch_start).total_seconds()
            
            # Procesar resultados
            for result in batch_results:
                if isinstance(result, Exception):
                    failed += 1
                    results.append({
                        "success": False,
                        "error": str(result)
                    })
                else:
                    if result.get("success"):
                        succeeded += 1
                    else:
                        failed += 1
                    results.append(result)
            
            # Actualizar progreso
            if on_progress:
                on_progress(len(results), len(tasks))
            
            self.stats["batches_processed"] += 1
            self.stats["total_processed"] += len(batch)
            
            # Actualizar tiempo promedio
            total_batches = self.stats["batches_processed"]
            current_avg = self.stats["average_batch_time"]
            self.stats["average_batch_time"] = (
                (current_avg * (total_batches - 1) + batch_duration) / total_batches
            )
        
        self.stats["total_succeeded"] += succeeded
        self.stats["total_failed"] += failed
        
        result = {
            "total": len(tasks),
            "succeeded": succeeded,
            "failed": failed,
            "results": results
        }
        
        logger.info(
            f"✅ Procesamiento de lote completado: {succeeded}/{len(tasks)} exitosas, "
            f"{failed} fallidas, {self.stats['batches_processed']} batches procesados, "
            f"tiempo promedio: {self.stats['average_batch_time']:.2f}s"
        )
        
        return result
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar una tarea individual.
        
        Args:
            task: Tarea a procesar
            
        Returns:
            Resultado del procesamiento
        """
        async with self.semaphore:
            try:
                if self.processor_func:
                    if asyncio.iscoroutinefunction(self.processor_func):
                        result = await self.processor_func(task)
                    else:
                        result = self.processor_func(task)
                    return {
                        "success": True,
                        "task_id": task.get("id"),
                        "result": result
                    }
                else:
                    return {
                        "success": False,
                        "error": "No processor function configured"
                    }
            except Exception as e:
                logger.error(f"Error procesando tarea {task.get('id')}: {e}", exc_info=True)
                return {
                    "success": False,
                    "task_id": task.get("id"),
                    "error": str(e)
                }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            **self.stats,
            "max_concurrent": self.max_concurrent,
            "batch_size": self.batch_size
        }

