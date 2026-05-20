"""Optimization module."""

from .batch_processor import BatchProcessor, AsyncBatchProcessor
from .async_optimizer import AsyncOptimizer
from .speed_optimizer import SpeedOptimizer, FastInference
from .memory_optimizer import MemoryOptimizer
from .aggressive_optimizer import (
    AggressiveOptimizer,
    InferenceCache,
    BatchInferenceOptimizer,
    cached_feature_extraction
)
from .kernel_fusion import KernelFusion
from .quantization import QuantizationOptimizer, FastInferenceEngine
from .torchscript_optimizer import TorchScriptOptimizer
from .batch_optimizer import SmartBatchProcessor, ParallelBatchProcessor

__all__ = [
    "BatchProcessor",
    "AsyncBatchProcessor",
    "AsyncOptimizer",
    "SpeedOptimizer",
    "FastInference",
    "MemoryOptimizer",
    "AggressiveOptimizer",
    "InferenceCache",
    "BatchInferenceOptimizer",
    "cached_feature_extraction",
    "KernelFusion",
    "QuantizationOptimizer",
    "FastInferenceEngine",
    "TorchScriptOptimizer",
    "SmartBatchProcessor",
    "ParallelBatchProcessor",
]

