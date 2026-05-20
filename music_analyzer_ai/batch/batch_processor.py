"""
Advanced Batch Processing
Efficient batch processing for large-scale music analysis
"""

from typing import List, Dict, Any, Optional, Callable, Iterator
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Advanced batch processor with:
    - Parallel processing
    - Progress tracking
    - Error handling
    - Result aggregation
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process items in batches"""
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(items), self.batch_size):
            batch = items[batch_idx:batch_idx + self.batch_size]
            
            # Process batch
            batch_results = []
            futures = []
            
            for item in batch:
                future = self.executor.submit(process_func, item)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 min timeout
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
                    batch_results.append({"error": str(e)})
            
            results.extend(batch_results)
            
            # Progress callback
            if progress_callback:
                progress = (batch_idx // self.batch_size + 1) / total_batches
                progress_callback(progress, batch_idx // self.batch_size + 1, total_batches)
        
        return results
    
    async def process_batch_async(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process items in batches asynchronously"""
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(items), self.batch_size):
            batch = items[batch_idx:batch_idx + self.batch_size]
            
            # Process batch asynchronously
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    process_func,
                    item
                )
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Async batch processing error: {str(result)}")
                    processed_results.append({"error": str(result)})
                else:
                    processed_results.append(result)
            
            results.extend(processed_results)
            
            # Progress callback
            if progress_callback:
                progress = (batch_idx // self.batch_size + 1) / total_batches
                await progress_callback(progress, batch_idx // self.batch_size + 1, total_batches)
        
        return results
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)


class StreamingProcessor:
    """
    Stream processing for real-time analysis
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer: List[Any] = []
        self.processors: List[Callable] = []
    
    def add_processor(self, processor: Callable):
        """Add processor to pipeline"""
        self.processors.append(processor)
    
    def process_stream(self, items: Iterator[Any]) -> Iterator[Any]:
        """Process stream of items"""
        for item in items:
            # Add to buffer
            self.buffer.append(item)
            
            # Process if buffer full
            if len(self.buffer) >= self.buffer_size:
                batch = self.buffer[:self.buffer_size]
                self.buffer = self.buffer[self.buffer_size:]
                
                # Process through pipeline
                result = batch
                for processor in self.processors:
                    result = processor(result)
                
                yield from result
    
    def flush(self) -> List[Any]:
        """Flush remaining items in buffer"""
        if not self.buffer:
            return []
        
        result = self.buffer
        for processor in self.processors:
            result = processor(result)
        
        self.buffer = []
        return result

