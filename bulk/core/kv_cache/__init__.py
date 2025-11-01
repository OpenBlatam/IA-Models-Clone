"""
Ultra-Adaptive Key-Value Cache Engine - Modular Package

This package provides a modular, extensible, and production-ready KV cache
implementation following best practices for PyTorch, Transformers, and LLMs.

Structure:
- config: Configuration classes (CacheStrategy, CacheMode, KVCacheConfig)
- quantization: Quantization module (Quantizer)
- compression: Compression module (Compressor)
- memory_manager: Memory management (MemoryManager)
- strategies: Eviction strategies (LRU, LFU, Adaptive)
"""

from kv_cache.config import (
    CacheStrategy,
    CacheMode,
    KVCacheConfig,
)
from kv_cache.quantization import Quantizer
from kv_cache.compression import Compressor
from kv_cache.memory_manager import MemoryManager
from kv_cache.strategies import (
    BaseEvictionStrategy,
    LRUEvictionStrategy,
    LFUEvictionStrategy,
    AdaptiveEvictionStrategy,
    create_eviction_strategy,
)

# Import base classes
from kv_cache.base import BaseKVCache
from kv_cache.stats import CacheStatsTracker
from kv_cache.device_manager import DeviceManager
from kv_cache.cache_storage import CacheStorage
from kv_cache.validators import CacheValidator
from kv_cache.utils import (
    get_device_info,
    validate_tensor_shapes,
    format_memory_size,
    safe_device_transfer,
    calculate_tensor_memory_mb,
    get_tensor_info,
)

# Import adapters
from kv_cache.adapters.adaptive_cache import AdaptiveKVCache
from kv_cache.adapters.paged_cache import PagedKVCache

# Import error handling and profiling
from kv_cache.error_handler import ErrorHandler, CacheError, CacheMemoryError, CacheValidationError, CacheDeviceError
from kv_cache.profiler import CacheProfiler

# Import optimizations
from kv_cache.optimizations import (
    FastQuantizer, FastCompressor, enable_torch_optimizations
)
from kv_cache.batch_operations import BatchCacheOperations

# Import additional features
from kv_cache.transformers_integration import TransformersKVCache, ModelCacheWrapper
from kv_cache.monitoring import CacheMonitor, CacheMetrics, MetricsExporter
from kv_cache.persistence import CachePersistence, save_cache_checkpoint, load_cache_checkpoint

# Try to import engine if available (may not exist yet)
try:
    from kv_cache.engine import UltraAdaptiveKVCacheEngine
except ImportError:
    UltraAdaptiveKVCacheEngine = None

__version__ = "2.5.0"  # Production Ready - Advanced Features Added
__all__ = [
    # Config
    "CacheStrategy",
    "CacheMode",
    "KVCacheConfig",
    # Components
    "Quantizer",
    "Compressor",
    "MemoryManager",
    # Strategies
    "BaseEvictionStrategy",
    "LRUEvictionStrategy",
    "LFUEvictionStrategy",
    "AdaptiveEvictionStrategy",
    "create_eviction_strategy",
    # Statistics
    "CacheStatsTracker",
    # Storage and Device
    "CacheStorage",
    "DeviceManager",
    # Validators
    "CacheValidator",
    # Utilities
    "get_device_info",
    "validate_tensor_shapes",
    "format_memory_size",
    "safe_device_transfer",
    "calculate_tensor_memory_mb",
    "get_tensor_info",
    # Main classes
    "BaseKVCache",
    "AdaptiveKVCache",
    "PagedKVCache",
    "UltraAdaptiveKVCacheEngine",
    # Error handling
    "ErrorHandler",
    "CacheError",
    "CacheMemoryError",
    "CacheValidationError",
    "CacheDeviceError",
    # Profiling
    "CacheProfiler",
    # Optimizations
    "FastQuantizer",
    "FastCompressor",
    "BatchCacheOperations",
    "enable_torch_optimizations",
    # Transformers Integration
    "TransformersKVCache",
    "ModelCacheWrapper",
    # Monitoring
    "CacheMonitor",
    "CacheMetrics",
    "MetricsExporter",
    # Persistence
    "CachePersistence",
    "save_cache_checkpoint",
    "load_cache_checkpoint",
]

