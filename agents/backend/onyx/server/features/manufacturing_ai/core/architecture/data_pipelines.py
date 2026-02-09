"""
Optimized Data Pipelines
========================

Pipelines de datos optimizados con procesamiento paralelo y streaming.
"""

import asyncio
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from queue import Queue, Empty
from collections import deque
import logging
import time

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataPipeline:
    """Pipeline de datos con transformaciones encadenadas."""
    
    def __init__(self):
        self.transformations: List[Callable] = []
        self.cache_enabled = False
        self.cache: Dict[str, Any] = {}
    
    def add_transform(self, transform: Callable) -> 'DataPipeline':
        """Agrega transformación al pipeline."""
        self.transformations.append(transform)
        return self
    
    def process(self, data: Any) -> Any:
        """Procesa datos a través del pipeline."""
        result = data
        for transform in self.transformations:
            result = transform(result)
        return result
    
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """Procesa lote de datos."""
        return [self.process(item) for item in batch]
    
    def __call__(self, data: Any) -> Any:
        """Permite usar pipeline como función."""
        return self.process(data)


class ParallelDataPipeline(DataPipeline):
    """Pipeline con procesamiento paralelo."""
    
    def __init__(self, num_workers: int = 4):
        super().__init__()
        self.num_workers = num_workers
        self.executor = None
    
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """Procesa lote en paralelo."""
        if TORCH_AVAILABLE:
            import torch.multiprocessing as mp
            with mp.Pool(self.num_workers) as pool:
                return pool.map(self.process, batch)
        else:
            import multiprocessing as mp
            with mp.Pool(self.num_workers) as pool:
                return pool.map(self.process, batch)


class StreamingPipeline:
    """Pipeline para streaming de datos en tiempo real."""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.input_queue = Queue(maxsize=buffer_size)
        self.output_queue = Queue(maxsize=buffer_size)
        self.transformations: List[Callable] = []
        self.running = False
        self.worker_thread = None
    
    def add_transform(self, transform: Callable) -> 'StreamingPipeline':
        """Agrega transformación."""
        self.transformations.append(transform)
        return self
    
    def start(self) -> None:
        """Inicia procesamiento en background."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
    
    def stop(self) -> None:
        """Detiene procesamiento."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
    
    def put(self, data: Any) -> bool:
        """Agrega dato al pipeline."""
        try:
            self.input_queue.put(data, timeout=1.0)
            return True
        except:
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Obtiene dato procesado."""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _process_loop(self) -> None:
        """Loop de procesamiento."""
        while self.running:
            try:
                data = self.input_queue.get(timeout=0.1)
                result = data
                for transform in self.transformations:
                    result = transform(result)
                self.output_queue.put(result)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")


class AsyncDataPipeline:
    """Pipeline asíncrono para procesamiento no bloqueante."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.transformations: List[Callable] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    def add_transform(self, transform: Callable) -> 'AsyncDataPipeline':
        """Agrega transformación."""
        self.transformations.append(transform)
        return self
    
    async def process(self, data: Any) -> Any:
        """Procesa dato de forma asíncrona."""
        async with self.semaphore:
            result = data
            for transform in self.transformations:
                if asyncio.iscoroutinefunction(transform):
                    result = await transform(result)
                else:
                    result = transform(result)
            return result
    
    async def process_batch(self, batch: List[Any]) -> List[Any]:
        """Procesa lote de forma asíncrona."""
        tasks = [self.process(item) for item in batch]
        return await asyncio.gather(*tasks)


class PrefetchDataLoader:
    """DataLoader con prefetching optimizado."""
    
    def __init__(
        self,
        dataset: Any,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PrefetchDataLoader")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


class BatchAggregator:
    """Agrega datos en lotes eficientes."""
    
    def __init__(self, batch_size: int, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.buffer = deque()
        self.last_batch_time = time.time()
    
    def add(self, item: Any) -> Optional[List[Any]]:
        """Agrega item y retorna lote si está completo."""
        self.buffer.append(item)
        current_time = time.time()
        
        # Retornar lote si está completo o timeout
        if (len(self.buffer) >= self.batch_size or 
            (current_time - self.last_batch_time) >= self.timeout):
            batch = list(self.buffer)
            self.buffer.clear()
            self.last_batch_time = current_time
            return batch
        return None
    
    def flush(self) -> Optional[List[Any]]:
        """Fuerza retorno de lote actual."""
        if self.buffer:
            batch = list(self.buffer)
            self.buffer.clear()
            return batch
        return None


class DataTransformer:
    """Transformador de datos reutilizable."""
    
    @staticmethod
    def normalize(data: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """Normaliza tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        return (data - mean) / std
    
    @staticmethod
    def standardize(data: torch.Tensor) -> torch.Tensor:
        """Estandariza tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        mean = data.mean()
        std = data.std()
        return (data - mean) / (std + 1e-8)
    
    @staticmethod
    def to_tensor(data: Any) -> torch.Tensor:
        """Convierte a tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(data)
    
    @staticmethod
    def pad_sequence(sequences: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        """Rellena secuencias a misma longitud."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padding = torch.full((pad_len, *seq.shape[1:]), pad_value)
                padded.append(torch.cat([seq, padding]))
            else:
                padded.append(seq)
        return torch.stack(padded)


class PipelineBuilder:
    """Builder para crear pipelines complejos."""
    
    def __init__(self):
        self.pipeline = DataPipeline()
    
    def add_normalization(self, mean: float = 0.0, std: float = 1.0) -> 'PipelineBuilder':
        """Agrega normalización."""
        def normalize(x):
            if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
                return DataTransformer.normalize(x, mean, std)
            return x
        self.pipeline.add_transform(normalize)
        return self
    
    def add_standardization(self) -> 'PipelineBuilder':
        """Agrega estandarización."""
        def standardize(x):
            if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
                return DataTransformer.standardize(x)
            return x
        self.pipeline.add_transform(standardize)
        return self
    
    def add_transform(self, transform: Callable) -> 'PipelineBuilder':
        """Agrega transformación personalizada."""
        self.pipeline.add_transform(transform)
        return self
    
    def build(self) -> DataPipeline:
        """Construye pipeline."""
        return self.pipeline


# Factory functions
def create_data_pipeline() -> DataPipeline:
    """Crea pipeline de datos básico."""
    return DataPipeline()


def create_parallel_pipeline(num_workers: int = 4) -> ParallelDataPipeline:
    """Crea pipeline paralelo."""
    return ParallelDataPipeline(num_workers=num_workers)


def create_streaming_pipeline(buffer_size: int = 100) -> StreamingPipeline:
    """Crea pipeline de streaming."""
    return StreamingPipeline(buffer_size=buffer_size)


def create_async_pipeline(max_concurrent: int = 10) -> AsyncDataPipeline:
    """Crea pipeline asíncrono."""
    return AsyncDataPipeline(max_concurrent=max_concurrent)


