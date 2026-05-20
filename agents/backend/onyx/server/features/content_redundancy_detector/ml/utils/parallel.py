"""
Parallel Processing Utilities
Parallel and async processing utilities
"""

import torch
from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    Parallel processing utilities
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        use_processes: bool = False,
    ):
        """
        Initialize parallel processor
        
        Args:
            num_workers: Number of workers
            use_processes: Use processes instead of threads
        """
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    def process_parallel(
        self,
        items: List[Any],
        process_fn: Callable,
        show_progress: bool = True,
    ) -> List[Any]:
        """
        Process items in parallel
        
        Args:
            items: List of items to process
            process_fn: Function to process each item
            show_progress: Show progress bar
            
        Returns:
            List of processed results
        """
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        with self.executor_class(max_workers=self.num_workers) as executor:
            if show_progress and use_tqdm:
                results = list(tqdm(
                    executor.map(process_fn, items),
                    total=len(items),
                    desc="Processing"
                ))
            else:
                results = list(executor.map(process_fn, items))
        
        return results
    
    def process_batches_parallel(
        self,
        batches: List[Any],
        process_fn: Callable,
    ) -> List[Any]:
        """
        Process batches in parallel
        
        Args:
            batches: List of batches
            process_fn: Function to process each batch
            
        Returns:
            List of processed results
        """
        with self.executor_class(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_fn, batches))
        
        return results



