"""
Advanced Data Processing Pipeline
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Callable, Iterator
import logging
from functools import partial

logger = logging.getLogger(__name__)


class PipelineStage:
    """Single pipeline stage"""
    
    def __init__(
        self,
        name: str,
        transform: Callable,
        enabled: bool = True
    ):
        """
        Initialize pipeline stage
        
        Args:
            name: Stage name
            transform: Transform function
            enabled: Whether stage is enabled
        """
        self.name = name
        self.transform = transform
        self.enabled = enabled
    
    def __call__(self, data: Any) -> Any:
        """Execute stage"""
        if not self.enabled:
            return data
        return self.transform(data)


class ProcessingPipeline:
    """Advanced processing pipeline"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.stages: List[PipelineStage] = []
        logger.info("ProcessingPipeline initialized")
    
    def add_stage(
        self,
        name: str,
        transform: Callable,
        enabled: bool = True,
        position: Optional[int] = None
    ):
        """
        Add stage to pipeline
        
        Args:
            name: Stage name
            transform: Transform function
            enabled: Whether stage is enabled
            position: Optional position (default: append)
        """
        stage = PipelineStage(name, transform, enabled)
        
        if position is not None:
            self.stages.insert(position, stage)
        else:
            self.stages.append(stage)
        
        logger.info(f"Stage added: {name}")
    
    def remove_stage(self, name: str):
        """Remove stage from pipeline"""
        self.stages = [s for s in self.stages if s.name != name]
        logger.info(f"Stage removed: {name}")
    
    def enable_stage(self, name: str):
        """Enable stage"""
        for stage in self.stages:
            if stage.name == name:
                stage.enabled = True
                logger.info(f"Stage enabled: {name}")
                return
    
    def disable_stage(self, name: str):
        """Disable stage"""
        for stage in self.stages:
            if stage.name == name:
                stage.enabled = False
                logger.info(f"Stage disabled: {name}")
                return
    
    def process(self, data: Any) -> Any:
        """
        Process data through pipeline
        
        Args:
            data: Input data
        
        Returns:
            Processed data
        """
        result = data
        for stage in self.stages:
            try:
                result = stage(result)
            except Exception as e:
                logger.error(f"Stage '{stage.name}' failed: {e}")
                raise
        return result
    
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """
        Process batch through pipeline
        
        Args:
            batch: Batch of data
        
        Returns:
            Processed batch
        """
        return [self.process(item) for item in batch]
    
    def get_stage_names(self) -> List[str]:
        """Get list of stage names"""
        return [stage.name for stage in self.stages]


class BatchProcessor:
    """Advanced batch processing"""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = 4
    ):
        """
        Initialize batch processor
        
        Args:
            batch_size: Batch size
            max_workers: Maximum workers
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        logger.info(f"BatchProcessor initialized: batch_size={batch_size}")
    
    def process(
        self,
        data: List[Any],
        processor: Callable
    ) -> List[Any]:
        """
        Process data in batches
        
        Args:
            data: Data to process
            processor: Processing function
        
        Returns:
            Processed data
        """
        results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_results = processor(batch)
            results.extend(batch_results)
        
        return results
    
    def process_streaming(
        self,
        data_stream: Iterator[Any],
        processor: Callable
    ) -> Iterator[Any]:
        """
        Process streaming data
        
        Args:
            data_stream: Data stream
            processor: Processing function
        
        Returns:
            Processed data stream
        """
        batch = []
        
        for item in data_stream:
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                results = processor(batch)
                for result in results:
                    yield result
                batch = []
        
        # Process remaining
        if batch:
            results = processor(batch)
            for result in results:
                yield result

