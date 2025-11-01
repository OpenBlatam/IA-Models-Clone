"""
Configuration module for KV Cache Engine.

Separates configuration from implementation for better modularity.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
    num_heads: int = 8
    head_dim: int = 64
    max_tokens: int = 4096
    block_size: int = 128
    
    # Strategy
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    cache_mode: CacheMode = CacheMode.INFERENCE
    
    # Optimization
    use_compression: bool = True
    compression_ratio: float = 0.3
    use_quantization: bool = False
    quantization_bits: int = 8
    compression_method: str = "svd"  # svd|lowrank|sparse
    
    # Memory
    max_memory_mb: Optional[int] = None
    enable_gc: bool = True
    gc_threshold: float = 0.8
    
    # Performance
    pin_memory: bool = True
    non_blocking: bool = True
    dtype: Optional[object] = None  # Will be set to torch.float16 in __post_init__
    
    # Adaptive settings
    adaptive_compression: bool = True
    adaptive_quantization: bool = True
    monitor_memory: bool = True
    
    # Advanced features
    enable_persistence: bool = False
    persistence_path: Optional[str] = None
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
    
    def __post_init__(self):
        """Set default dtype if not provided."""
        import torch
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
        if not 0.0 < self.compression_ratio <= 1.0:
            raise ValueError(
                f"compression_ratio must be in (0, 1], got {self.compression_ratio}"
            )
        if not 0.0 < self.gc_threshold <= 1.0:
            raise ValueError(
                f"gc_threshold must be in (0, 1], got {self.gc_threshold}"
            )
        if self.quantization_bits not in [4, 8, 16]:
            raise ValueError(
                f"quantization_bits must be 4, 8, or 16, got {self.quantization_bits}"
            )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        import torch
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, torch.dtype):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "KVCacheConfig":
        """Create config from dictionary."""
        import torch
        
        # Handle dtype conversion
        if "dtype" in config_dict:
            dtype_str = config_dict["dtype"]
            if isinstance(dtype_str, str):
                if dtype_str == "float16":
                    config_dict["dtype"] = torch.float16
                elif dtype_str == "float32":
                    config_dict["dtype"] = torch.float32
                elif dtype_str == "bfloat16":
                    config_dict["dtype"] = torch.bfloat16
        
        # Handle enum conversions
        if "cache_strategy" in config_dict:
            if isinstance(config_dict["cache_strategy"], str):
                config_dict["cache_strategy"] = CacheStrategy(config_dict["cache_strategy"])
        
        if "cache_mode" in config_dict:
            if isinstance(config_dict["cache_mode"], str):
                config_dict["cache_mode"] = CacheMode(config_dict["cache_mode"])
        
        return cls(**config_dict)

