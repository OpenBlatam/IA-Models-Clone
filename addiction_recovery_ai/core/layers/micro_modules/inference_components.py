"""
Inference Components - Ultra-Granular Inference Management
Re-exports from specialized modules for backward compatibility
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# Import from specialized modules
from .inference.batch_processor import (
    BatchProcessorBase,
    StandardBatchProcessor,
    BatchProcessorFactory
)

from .inference.cache_manager import (
    CacheManagerBase,
    LRUCacheManager,
    CacheManagerFactory
)

from .inference.output_formatter import (
    OutputFormatterBase,
    TensorFormatter,
    OutputFormatterFactory
)

from .inference.post_processor import (
    PostProcessorBase,
    SoftmaxPostProcessor,
    PostProcessorFactory
)


# Backward compatibility wrappers
class BatchProcessor:
    """Process batches during inference (backward compatibility)"""
    
    @staticmethod
    def process_batch(model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Process batch"""
        processor = StandardBatchProcessor()
        return processor.process(model, batch, **kwargs)


class CacheManager:
    """Manage caching during inference (backward compatibility)"""
    
    def __init__(self, cache_type: str = 'lru', max_size: int = 100):
        self.cache = CacheManagerFactory.create(cache_type, max_size=max_size)
    
    def get(self, key: Any) -> Optional[Any]:
        """Get from cache"""
        return self.cache.get(key)
    
    def set(self, key: Any, value: Any) -> None:
        """Set in cache"""
        self.cache.set(key, value)
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()


class OutputFormatter:
    """Format outputs (backward compatibility)"""
    
    @staticmethod
    def format(outputs: torch.Tensor, format_type: str = 'tensor', **kwargs) -> Any:
        """Format outputs"""
        formatter = OutputFormatterFactory.create(format_type, **kwargs)
        return formatter.format(outputs, **kwargs)


class PostProcessor:
    """Post-process outputs (backward compatibility)"""
    
    @staticmethod
    def process(outputs: torch.Tensor, processor_type: str = 'softmax', **kwargs) -> torch.Tensor:
        """Post-process outputs"""
        processor = PostProcessorFactory.create(processor_type, **kwargs)
        return processor.process(outputs, **kwargs)


# Export all components
__all__ = [
    "BatchProcessor",
    "CacheManager",
    "OutputFormatter",
    "PostProcessor",
]
