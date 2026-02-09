"""
Batch Processing Utilities
===========================

Utilidades para procesamiento en lotes optimizado.
"""

from typing import List, Callable, Any, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Procesador de lotes optimizado.
    
    Procesa items en lotes con diferentes estrategias.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        """
        Inicializar procesador.
        
        Args:
            batch_size: Tamaño del lote
            max_workers: Número máximo de workers
            use_processes: Si True, usar procesos en lugar de threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    def process(
        self,
        items: List[Any],
        processor: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """
        Procesar items en lotes.
        
        Args:
            items: Lista de items a procesar
            processor: Función de procesamiento
            progress_callback: Callback de progreso (current, total)
            
        Returns:
            Lista de resultados
        """
        results = []
        total = len(items)
        
        with self.executor_class(max_workers=self.max_workers) as executor:
            # Dividir en lotes
            batches = [
                items[i:i + self.batch_size]
                for i in range(0, total, self.batch_size)
            ]
            
            # Procesar lotes
            futures = []
            for batch in batches:
                future = executor.submit(self._process_batch, batch, processor)
                futures.append(future)
            
            # Recopilar resultados
            completed = 0
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed * self.batch_size, total)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        return results
    
    def _process_batch(self, batch: List[Any], processor: Callable) -> List[Any]:
        """Procesar un lote."""
        return [processor(item) for item in batch]
    
    def process_map(
        self,
        items: List[Any],
        processor: Callable[[Any], Any]
    ) -> List[Any]:
        """
        Procesar items usando map (más simple).
        
        Args:
            items: Lista de items
            processor: Función de procesamiento
            
        Returns:
            Lista de resultados
        """
        with self.executor_class(max_workers=self.max_workers) as executor:
            return list(executor.map(processor, items))


def batch_iter(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    Iterar sobre items en lotes.
    
    Args:
        items: Lista de items
        batch_size: Tamaño del lote
        
    Yields:
        Lotes de items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: int = 4,
    use_processes: bool = False
) -> List[Any]:
    """
    Aplicar función a items en paralelo.
    
    Args:
        func: Función a aplicar
        items: Lista de items
        max_workers: Número máximo de workers
        use_processes: Si True, usar procesos
        
    Returns:
        Lista de resultados
    """
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        return list(executor.map(func, items))


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Dividir lista en chunks.
    
    Args:
        items: Lista a dividir
        chunk_size: Tamaño del chunk
        
    Returns:
        Lista de chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]






