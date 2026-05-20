"""
Configuration module for KV Cache Engine.

Separates configuration from implementation for better modularity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import torch

from kv_cache.constants import (
    DEFAULT_MAX_TOKENS, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM,
    DEFAULT_BLOCK_SIZE, DEFAULT_COMPRESSION_RATIO, DEFAULT_GC_THRESHOLD,
    COMPRESSION_RATIO_MIN, COMPRESSION_RATIO_MAX,
    QUANTIZATION_BITS_SUPPORTED
)


class CacheStrategy(Enum):
    """KV Cache strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns
    PAGED = "paged"  # Paged memory allocation
    COMPRESSED = "compressed"  # With compression
    QUANTIZED = "quantized"  # With quantization


class CacheMode(Enum):
    """Cache operation modes."""
    TRAINING = "training"
    INFERENCE = "inference"
    BULK = "bulk"
    STREAMING = "streaming"
    INTERACTIVE = "interactive"


@dataclass
class KVCacheConfig:
    """
    Configuration for KV cache engine.
    
    Follows best practices with clear separation of concerns:
    - Core settings: Model architecture parameters
    - Strategy: Cache replacement strategy
    - Optimization: Compression and quantization
    - Memory: Memory management settings
    - Performance: GPU and performance optimizations
    """
    # Core settings
    num_heads: int = DEFAULT_NUM_HEADS
    head_dim: int = DEFAULT_HEAD_DIM
    max_tokens: int = DEFAULT_MAX_TOKENS
    block_size: int = DEFAULT_BLOCK_SIZE
    
    # Strategy
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    cache_mode: CacheMode = CacheMode.INFERENCE
    
    # Optimization
    use_compression: bool = True
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO
    use_quantization: bool = False
    quantization_bits: int = 8
    compression_method: str = "svd"  # svd|lowrank|sparse
    
    # Memory
    max_memory_mb: int | None = None
    enable_gc: bool = True
    gc_threshold: float = DEFAULT_GC_THRESHOLD
    
    # Performance
    pin_memory: bool = True
    non_blocking: bool = True
    dtype: torch.dtype | None = None  # Will be set to torch.float16 in __post_init__
    
    # Adaptive settings
    adaptive_compression: bool = True
    adaptive_quantization: bool = True
    monitor_memory: bool = True
    
    # Advanced features
    enable_persistence: bool = False
    persistence_path: str | None = None
    enable_prefetch: bool = True
    prefetch_size: int = 4
    enable_profiling: bool = False
    enable_distributed: bool = False
    distributed_backend: str = "nccl"  # nccl|gloo|mpi
    multi_tenant: bool = False
    tenant_isolation: bool = True
    enable_async: bool = True
    async_threads: int = 2
    enable_warmup: bool = False
    warmup_samples: int = 100
    
    def __post_init__(self) -> None:
        """Set default dtype if not provided."""
        if self.dtype is None:
            self.dtype = torch.float16
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if not COMPRESSION_RATIO_MIN < self.compression_ratio <= COMPRESSION_RATIO_MAX:
            raise ValueError(
                f"compression_ratio must be in ({COMPRESSION_RATIO_MIN}, {COMPRESSION_RATIO_MAX}], "
                f"got {self.compression_ratio}"
            )
        if not 0.0 < self.gc_threshold <= 1.0:
            raise ValueError(
                f"gc_threshold must be in (0, 1], got {self.gc_threshold}"
            )
        if self.quantization_bits not in QUANTIZATION_BITS_SUPPORTED:
            raise ValueError(
                f"quantization_bits must be one of {QUANTIZATION_BITS_SUPPORTED}, "
                f"got {self.quantization_bits}"
            )
    
    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        """Convert config to dictionary."""
        result: dict[str, str | int | float | bool | None] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, torch.dtype):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, str | int | float | bool | None]) -> KVCacheConfig:
        """
        Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            KVCacheConfig instance
        """
        # Create a copy to avoid modifying the original
        config_dict = config_dict.copy()
        
        # Handle dtype conversion
        if "dtype" in config_dict:
            dtype_value = config_dict["dtype"]
            if isinstance(dtype_value, str):
                dtype_map: dict[str, torch.dtype] = {
                    "float16": torch.float16,
                    "float32": torch.float32,
                    "bfloat16": torch.bfloat16,
                }
                config_dict["dtype"] = dtype_map.get(dtype_value, torch.float16)
        
        # Handle enum conversions
        if "cache_strategy" in config_dict:
            strategy_value = config_dict["cache_strategy"]
            if isinstance(strategy_value, str):
                config_dict["cache_strategy"] = CacheStrategy(strategy_value)
        
        if "cache_mode" in config_dict:
            mode_value = config_dict["cache_mode"]
            if isinstance(mode_value, str):
                config_dict["cache_mode"] = CacheMode(mode_value)
        
        return cls(**config_dict)

