"""Parallel processing utilities"""
import asyncio
from typing import List, Callable, Any, TypeVar, Coroutine, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ParallelProcessor:
    """Process tasks in parallel"""
    
    def __init__(self, max_workers: int = 4, use_processes: bool = False):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Maximum number of workers
            use_processes: Use processes instead of threads
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor = ProcessPoolExecutor(max_workers=max_workers) if use_processes else ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_async(
        self,
        tasks: List[Coroutine[Any, Any, T]],
        timeout: Optional[float] = None
    ) -> List[T]:
        """
        Process async tasks in parallel
        
        Args:
            tasks: List of async tasks
            timeout: Optional timeout in seconds
            
        Returns:
            List of results
        """
        try:
            if timeout:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return processed_results
        except asyncio.TimeoutError:
            logger.error(f"Parallel processing timed out after {timeout} seconds")
            return [None] * len(tasks)
    
    def process_sync(
        self,
        tasks: List[Callable[[], T]],
        timeout: Optional[float] = None
    ) -> List[T]:
        """
        Process sync tasks in parallel
        
        Args:
            tasks: List of callable tasks
            timeout: Optional timeout in seconds
            
        Returns:
            List of results
        """
        try:
            futures = [self.executor.submit(task) for task in tasks]
            
            if timeout:
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=timeout)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
                        results.append(None)
                return results
            else:
                return [future.result() for future in futures]
        except Exception as e:
            logger.error(f"Parallel processing error: {e}")
            return [None] * len(tasks)
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)


# Global parallel processor
_parallel_processor: Optional[ParallelProcessor] = None


def get_parallel_processor(max_workers: int = 4) -> ParallelProcessor:
    """Get global parallel processor"""
    global _parallel_processor
    if _parallel_processor is None:
        _parallel_processor = ParallelProcessor(max_workers=max_workers)
    return _parallel_processor

