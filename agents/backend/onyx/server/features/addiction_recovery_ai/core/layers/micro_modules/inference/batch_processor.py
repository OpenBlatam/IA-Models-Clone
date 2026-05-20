"""
Batch Processor - Ultra-Specific Batch Processing
Separated into its own file for maximum modularity
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BatchProcessorBase(ABC):
    """Base class for batch processors"""
    
    def __init__(self, name: str = "BatchProcessor"):
        self.name = name
    
    @abstractmethod
    def process(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Process batch"""
        pass


class StandardBatchProcessor(BatchProcessorBase):
    """Standard batch processing"""
    
    def __init__(self):
        super().__init__("StandardBatchProcessor")
    
    def process(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Process batch with standard forward pass"""
        model.eval()
        with torch.no_grad():
            outputs = model(batch['input'])
        
        return {
            'outputs': outputs,
            'batch_size': batch['input'].size(0)
        }


class BatchedInferenceProcessor(BatchProcessorBase):
    """Process batches with automatic batching"""
    
    def __init__(self, batch_size: int = 32):
        super().__init__("BatchedInferenceProcessor")
        self.batch_size = batch_size
    
    def process(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Process batch with automatic batching"""
        model.eval()
        
        inputs = batch['input']
        total_size = inputs.size(0)
        all_outputs = []
        
        with torch.no_grad():
            for i in range(0, total_size, self.batch_size):
                end_idx = min(i + self.batch_size, total_size)
                batch_input = inputs[i:end_idx]
                batch_output = model(batch_input)
                all_outputs.append(batch_output)
        
        outputs = torch.cat(all_outputs, dim=0)
        
        return {
            'outputs': outputs,
            'batch_size': total_size,
            'num_batches': (total_size + self.batch_size - 1) // self.batch_size
        }


class StreamingBatchProcessor(BatchProcessorBase):
    """Process batches in streaming fashion"""
    
    def __init__(self, chunk_size: int = 16):
        super().__init__("StreamingBatchProcessor")
        self.chunk_size = chunk_size
    
    def process(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Process batch in streaming chunks"""
        model.eval()
        
        inputs = batch['input']
        total_size = inputs.size(0)
        all_outputs = []
        
        with torch.no_grad():
            for i in range(0, total_size, self.chunk_size):
                end_idx = min(i + self.chunk_size, total_size)
                chunk = inputs[i:end_idx]
                chunk_output = model(chunk)
                all_outputs.append(chunk_output)
        
        outputs = torch.cat(all_outputs, dim=0)
        
        return {
            'outputs': outputs,
            'batch_size': total_size,
            'chunk_size': self.chunk_size,
            'num_chunks': (total_size + self.chunk_size - 1) // self.chunk_size
        }


class ParallelBatchProcessor(BatchProcessorBase):
    """Process batches in parallel"""
    
    def __init__(self, num_workers: int = 2):
        super().__init__("ParallelBatchProcessor")
        self.num_workers = num_workers
    
    def process(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Process batch in parallel"""
        # This is a simplified version
        # Full implementation would use multiprocessing or threading
        model.eval()
        
        inputs = batch['input']
        chunk_size = inputs.size(0) // self.num_workers
        
        with torch.no_grad():
            outputs = model(inputs)
        
        return {
            'outputs': outputs,
            'batch_size': inputs.size(0),
            'num_workers': self.num_workers
        }


# Factory for batch processors
class BatchProcessorFactory:
    """Factory for creating batch processors"""
    
    _registry = {
        'standard': StandardBatchProcessor,
        'batched': BatchedInferenceProcessor,
        'streaming': StreamingBatchProcessor,
        'parallel': ParallelBatchProcessor,
    }
    
    @classmethod
    def create(cls, processor_type: str, **kwargs) -> BatchProcessorBase:
        """Create batch processor"""
        processor_type = processor_type.lower()
        if processor_type not in cls._registry:
            raise ValueError(f"Unknown batch processor type: {processor_type}")
        return cls._registry[processor_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, processor_class: type):
        """Register custom batch processor"""
        cls._registry[name.lower()] = processor_class


__all__ = [
    "BatchProcessorBase",
    "StandardBatchProcessor",
    "BatchedInferenceProcessor",
    "StreamingBatchProcessor",
    "ParallelBatchProcessor",
    "BatchProcessorFactory",
]



