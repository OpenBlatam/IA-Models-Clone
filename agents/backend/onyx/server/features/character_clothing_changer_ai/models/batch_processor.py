"""
Batch Processor for Flux2 Clothing Changer
==========================================

Advanced batch processing with progress tracking and parallel execution.
"""

import torch
from typing import List, Dict, Any, Optional, Callable, Union
from PIL import Image
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from queue import Queue
import threading

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Single item in a batch."""
    image: Union[Image.Image, str, Path, np.ndarray]
    clothing_description: str
    mask: Optional[Union[Image.Image, np.ndarray]] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult:
    """Result of batch processing."""
    success: bool
    result: Optional[Image.Image] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class BatchProcessor:
    """Advanced batch processor for clothing changes."""
    
    def __init__(
        self,
        model,
        max_workers: int = 2,
        batch_size: int = 4,
        enable_progress: bool = True,
        enable_parallel: bool = True,
    ):
        """
        Initialize batch processor.
        
        Args:
            model: Flux2ClothingChangerModelV2 instance
            max_workers: Maximum parallel workers
            batch_size: Batch size for processing
            enable_progress: Enable progress tracking
            enable_parallel: Enable parallel processing
        """
        self.model = model
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_progress = enable_progress
        self.enable_parallel = enable_parallel
    
    def process_batch(
        self,
        items: List[BatchItem],
        callback: Optional[Callable[[int, BatchResult], None]] = None,
    ) -> List[BatchResult]:
        """
        Process a batch of items.
        
        Args:
            items: List of batch items
            callback: Optional callback for progress (index, result)
            
        Returns:
            List of batch results
        """
        total = len(items)
        results = []
        
        if self.enable_parallel and self.max_workers > 1:
            results = self._process_parallel(items, callback)
        else:
            results = self._process_sequential(items, callback)
        
        return results
    
    def _process_sequential(
        self,
        items: List[BatchItem],
        callback: Optional[Callable[[int, BatchResult], None]] = None,
    ) -> List[BatchResult]:
        """Process items sequentially."""
        results = []
        
        for idx, item in enumerate(items):
            start_time = time.time()
            
            try:
                result_image = self.model.change_clothing(
                    image=item.image,
                    clothing_description=item.clothing_description,
                    mask=item.mask,
                    prompt=item.prompt,
                    negative_prompt=item.negative_prompt,
                )
                
                processing_time = time.time() - start_time
                result = BatchResult(
                    success=True,
                    result=result_image,
                    processing_time=processing_time,
                    metadata=item.metadata,
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                result = BatchResult(
                    success=False,
                    error=str(e),
                    processing_time=processing_time,
                    metadata=item.metadata,
                )
                logger.error(f"Batch item {idx} failed: {e}")
            
            results.append(result)
            
            if callback:
                callback(idx, result)
            
            if self.enable_progress:
                logger.info(f"Processed {idx + 1}/{len(items)} items")
        
        return results
    
    def _process_parallel(
        self,
        items: List[BatchItem],
        callback: Optional[Callable[[int, BatchResult], None]] = None,
    ) -> List[BatchResult]:
        """Process items in parallel."""
        results = [None] * len(items)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._process_item, item): idx
                for idx, item in enumerate(items)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    completed += 1
                    
                    if callback:
                        callback(idx, result)
                    
                    if self.enable_progress:
                        logger.info(f"Processed {completed}/{len(items)} items")
                        
                except Exception as e:
                    logger.error(f"Batch item {idx} failed: {e}")
                    results[idx] = BatchResult(
                        success=False,
                        error=str(e),
                        metadata=items[idx].metadata,
                    )
        
        return results
    
    def _process_item(self, item: BatchItem) -> BatchResult:
        """Process a single item."""
        start_time = time.time()
        
        try:
            result_image = self.model.change_clothing(
                image=item.image,
                clothing_description=item.clothing_description,
                mask=item.mask,
                prompt=item.prompt,
                negative_prompt=item.negative_prompt,
            )
            
            processing_time = time.time() - start_time
            return BatchResult(
                success=True,
                result=result_image,
                processing_time=processing_time,
                metadata=item.metadata,
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return BatchResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                metadata=item.metadata,
            )
    
    def process_stream(
        self,
        items_queue: Queue,
        results_queue: Queue,
        stop_event: threading.Event,
    ) -> None:
        """
        Process items from a queue stream.
        
        Args:
            items_queue: Queue of items to process
            results_queue: Queue for results
            stop_event: Event to stop processing
        """
        while not stop_event.is_set():
            try:
                item = items_queue.get(timeout=1.0)
                result = self._process_item(item)
                results_queue.put(result)
                items_queue.task_done()
            except:
                continue


