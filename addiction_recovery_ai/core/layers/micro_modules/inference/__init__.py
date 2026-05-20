"""
Inference Micro-Modules
Organized by category for maximum modularity
"""

from .batch_processor import (
    BatchProcessorBase,
    StandardBatchProcessor,
    BatchedInferenceProcessor,
    StreamingBatchProcessor,
    ParallelBatchProcessor,
    BatchProcessorFactory
)

from .cache_manager import (
    CacheManagerBase,
    LRUCacheManager,
    FIFOCacheManager,
    TTLCacheManager,
    NoCacheManager,
    CacheManagerFactory
)

from .output_formatter import (
    OutputFormatterBase,
    TensorFormatter,
    NumpyFormatter,
    ListFormatter,
    DictFormatter,
    ProbabilityFormatter,
    LogitsFormatter,
    ArgMaxFormatter,
    TopKFormatter,
    OutputFormatterFactory
)

from .post_processor import (
    PostProcessorBase,
    SoftmaxPostProcessor,
    SigmoidPostProcessor,
    NormalizePostProcessor,
    ClampPostProcessor,
    ThresholdPostProcessor,
    ArgMaxPostProcessor,
    ComposePostProcessor,
    PostProcessorFactory
)

# Backward compatibility
from ..inference_components import (
    BatchProcessor,
    CacheManager,
    OutputFormatter,
    PostProcessor
)

__all__ = [
    # Batch Processing
    "BatchProcessorBase",
    "StandardBatchProcessor",
    "BatchedInferenceProcessor",
    "StreamingBatchProcessor",
    "ParallelBatchProcessor",
    "BatchProcessorFactory",
    # Cache Management
    "CacheManagerBase",
    "LRUCacheManager",
    "FIFOCacheManager",
    "TTLCacheManager",
    "NoCacheManager",
    "CacheManagerFactory",
    # Output Formatting
    "OutputFormatterBase",
    "TensorFormatter",
    "NumpyFormatter",
    "ListFormatter",
    "DictFormatter",
    "ProbabilityFormatter",
    "LogitsFormatter",
    "ArgMaxFormatter",
    "TopKFormatter",
    "OutputFormatterFactory",
    # Post Processing
    "PostProcessorBase",
    "SoftmaxPostProcessor",
    "SigmoidPostProcessor",
    "NormalizePostProcessor",
    "ClampPostProcessor",
    "ThresholdPostProcessor",
    "ArgMaxPostProcessor",
    "ComposePostProcessor",
    "PostProcessorFactory",
    # Backward Compatibility
    "BatchProcessor",
    "CacheManager",
    "OutputFormatter",
    "PostProcessor",
]

