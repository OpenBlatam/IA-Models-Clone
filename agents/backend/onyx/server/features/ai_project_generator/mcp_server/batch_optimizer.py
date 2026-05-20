"""
MCP Batch Optimizer - Optimizador de batch processing
=======================================================
"""

import logging
import asyncio
from typing import List, Any, Callable, Dict, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class BatchOptimizer:
    """
    Optimizador de batch processing
    
    Optimiza el procesamiento de batches para mejor rendimiento.
    """
    
    def __init__(
        self,
        max_batch_size: int = 100,
        max_concurrent_batches: int = 10,
        auto_batch: bool = True,
    ):
        """
        Args:
            max_batch_size: Tamaño máximo del batch
            max_concurrent_batches: Máximo de batches concurrentes
            auto_batch: Si agrupa automáticamente requests similares
        """
        self.max_batch_size = max_batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.auto_batch = auto_batch
        self._pending_requests: Dict[str, List[Any]] = defaultdict(list)
        self._pending_futures: Dict[str, List[asyncio.Future]] = defaultdict(list)
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        group_key: Optional[Callable] = None,
    ) -> List[Any]:
        """
        Procesa items en batch optimizado
        
        Args:
            items: Lista de items
            processor: Función procesadora
            group_key: Función para agrupar items (opcional)
            
        Returns:
            Lista de resultados
        """
        if not items:
            return []
        
        # Agrupar si es necesario
        if group_key and self.auto_batch:
            groups = defaultdict(list)
            for item in items:
                key = group_key(item)
                groups[key].append(item)
            
            # Procesar cada grupo
            tasks = []
            for group_items in groups.values():
                tasks.append(self._process_group(group_items, processor))
            
            results = await asyncio.gather(*tasks)
            return [item for group_results in results for item in group_results]
        
        # Procesar como un solo batch
        return await self._process_group(items, processor)
    
    async def _process_group(
        self,
        items: List[Any],
        processor: Callable,
    ) -> List[Any]:
        """Procesa un grupo de items"""
        # Dividir en batches
        batches = [
            items[i:i + self.max_batch_size]
            for i in range(0, len(items), self.max_batch_size)
        ]
        
        # Procesar batches con semáforo
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_with_semaphore(batch):
            async with semaphore:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(batch)
                return processor(batch)
        
        tasks = [process_with_semaphore(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        # Aplanar resultados
        return [item for batch_results in results for item in batch_results]
    
    async def queue_request(
        self,
        request: Any,
        group_key: str,
        processor: Callable,
        timeout: float = 5.0,
    ) -> Any:
        """
        Encola un request para procesamiento en batch
        
        Args:
            request: Request a procesar
            group_key: Clave del grupo
            processor: Función procesadora
            timeout: Timeout en segundos
            
        Returns:
            Resultado del request
        """
        future = asyncio.Future()
        self._pending_requests[group_key].append(request)
        self._pending_futures[group_key].append(future)
        
        # Si el batch está lleno, procesar
        if len(self._pending_requests[group_key]) >= self.max_batch_size:
            await self._flush_group(group_key, processor)
        
        # Esperar resultado con timeout
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for group {group_key}")
            future.cancel()
            raise
    
    async def _flush_group(
        self,
        group_key: str,
        processor: Callable,
    ):
        """Procesa y vacía un grupo"""
        requests = self._pending_requests[group_key]
        futures = self._pending_futures[group_key]
        
        if not requests:
            return
        
        # Limpiar
        self._pending_requests[group_key] = []
        self._pending_futures[group_key] = []
        
        # Procesar batch
        try:
            if asyncio.iscoroutinefunction(processor):
                results = await processor(requests)
            else:
                results = processor(requests)
            
            # Resolver futures
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
                    
        except Exception as e:
            # Resolver con error
            for future in futures:
                if not future.done():
                    future.set_exception(e)
            
            logger.error(f"Error processing batch for group {group_key}: {e}")
    
    async def flush_all(self, processor: Callable):
        """Procesa todos los grupos pendientes"""
        group_keys = list(self._pending_requests.keys())
        
        tasks = [
            self._flush_group(key, processor)
            for key in group_keys
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del optimizador"""
        return {
            "pending_groups": len(self._pending_requests),
            "pending_requests": sum(len(reqs) for reqs in self._pending_requests.values()),
            "max_batch_size": self.max_batch_size,
            "max_concurrent_batches": self.max_concurrent_batches,
        }

