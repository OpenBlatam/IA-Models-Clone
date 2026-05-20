"""
MCP Batch Operations - Operaciones en lote
==========================================
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class BatchRequest(BaseModel):
    """Request para operaciones en lote"""
    requests: List[Dict[str, Any]] = Field(..., description="Lista de requests")
    stop_on_error: bool = Field(default=False, description="Detener en primer error")
    max_concurrent: int = Field(default=10, description="Máximo de operaciones concurrentes")


class BatchResponse(BaseModel):
    """Response para operaciones en lote"""
    results: List[Dict[str, Any]] = Field(..., description="Resultados de las operaciones")
    total: int = Field(..., description="Total de operaciones")
    successful: int = Field(..., description="Operaciones exitosas")
    failed: int = Field(..., description="Operaciones fallidas")
    duration: float = Field(..., description="Duración total en segundos")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchProcessor:
    """
    Procesador de operaciones en lote
    
    Permite ejecutar múltiples operaciones MCP en paralelo
    con control de concurrencia y manejo de errores.
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Inicializa el procesador de lotes.
        
        Args:
            max_concurrent: Máximo de operaciones concurrentes (debe ser > 0)
            
        Raises:
            ValueError: Si max_concurrent es inválido
        """
        if not isinstance(max_concurrent, int) or max_concurrent <= 0:
            raise ValueError("max_concurrent must be a positive integer")
        
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        operations: List[Dict[str, Any]],
        processor_func: Callable[[Dict[str, Any]], Any],
        stop_on_error: bool = False,
    ) -> BatchResponse:
        """
        Procesa un lote de operaciones
        
        Args:
            operations: Lista de operaciones a procesar
            processor_func: Función que procesa cada operación
            stop_on_error: Detener en primer error
            
        Returns:
            BatchResponse con resultados
        """
        start_time = datetime.now(timezone.utc)
        results = []
        successful = 0
        failed = 0
        
        async def process_one(operation: Dict[str, Any], index: int) -> Dict[str, Any]:
            """Procesa una operación"""
            async with self.semaphore:
                try:
                    result = await processor_func(operation)
                    return {
                        "index": index,
                        "success": True,
                        "result": result,
                        "error": None,
                    }
                except Exception as e:
                    logger.error(f"Batch operation {index} failed: {e}")
                    return {
                        "index": index,
                        "success": False,
                        "result": None,
                        "error": str(e),
                    }
        
        # Procesar operaciones
        tasks = [process_one(op, i) for i, op in enumerate(operations)]
        
        if stop_on_error:
            # Procesar secuencialmente, detener en error
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                if result["success"]:
                    successful += 1
                else:
                    failed += 1
                    break
        else:
            # Procesar en paralelo
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed:
                if isinstance(result, Exception):
                    failed += 1
                    results.append({
                        "index": len(results),
                        "success": False,
                        "result": None,
                        "error": str(result),
                    })
                elif result["success"]:
                    successful += 1
                    results.append(result)
                else:
                    failed += 1
                    results.append(result)
        
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return BatchResponse(
            results=results,
            total=len(operations),
            successful=successful,
            failed=failed,
            duration=duration,
        )


async def batch_query(
    operations: List[Dict[str, Any]],
    query_func: Callable[[Dict[str, Any]], Any],
    max_concurrent: int = 10,
    stop_on_error: bool = False,
) -> BatchResponse:
    """
    Ejecuta múltiples queries en lote.
    
    Args:
        operations: Lista de operaciones (cada una con resource_id, operation, parameters)
        query_func: Función async que ejecuta cada query
        max_concurrent: Máximo de operaciones concurrentes (debe ser > 0)
        stop_on_error: Detener en primer error
        
    Returns:
        BatchResponse con resultados
        
    Raises:
        ValueError: Si max_concurrent es inválido o operations está vacía
        TypeError: Si query_func no es callable
    """
    if not operations:
        raise ValueError("operations cannot be empty")
    if not callable(query_func):
        raise TypeError("query_func must be callable")
    if not isinstance(max_concurrent, int) or max_concurrent <= 0:
        raise ValueError("max_concurrent must be a positive integer")
    processor = BatchProcessor(max_concurrent=max_concurrent)
    return await processor.process_batch(
        operations=operations,
        processor_func=query_func,
        stop_on_error=stop_on_error,
    )

