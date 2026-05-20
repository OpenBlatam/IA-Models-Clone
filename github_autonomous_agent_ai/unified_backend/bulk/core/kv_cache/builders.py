"""
Builder utilities for KV Cache.

Provides fluent builders for creating cache configurations.
"""
from __future__ import annotations

from typing import Any

from kv_cache.config import KVCacheConfig, CacheStrategy, CacheMode
from kv_cache.constants import (
    DEFAULT_MAX_TOKENS, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM,
    DEFAULT_COMPRESSION_RATIO
)


class CacheConfigBuilder:
    """
    Fluent builder for KVCacheConfig.
    
    Example:
        config = (CacheConfigBuilder()
                 .with_max_tokens(4096)
                 .with_strategy(CacheStrategy.ADAPTIVE)
                 .with_compression(ratio=0.3)
                 .build())
    """
    
    def __init__(self) -> None:
        """Initialize builder with defaults."""
        self._config_dict: dict[str, Any] = {}
    
    def with_max_tokens(self, max_tokens: int) -> CacheConfigBuilder:
        """Set maximum number of tokens."""
        self._config_dict["max_tokens"] = max_tokens
        return self
    
    def with_heads(self, num_heads: int) -> CacheConfigBuilder:
        """Set number of attention heads."""
        self._config_dict["num_heads"] = num_heads
        return self
    
    def with_head_dim(self, head_dim: int) -> CacheConfigBuilder:
        """Set head dimension."""
        self._config_dict["head_dim"] = head_dim
        return self
    
    def with_strategy(self, strategy: CacheStrategy) -> CacheConfigBuilder:
        """Set cache strategy."""
        self._config_dict["cache_strategy"] = strategy
        return self
    
    def with_mode(self, mode: CacheMode) -> CacheConfigBuilder:
        """Set cache mode."""
        self._config_dict["cache_mode"] = mode
        return self
    
    def with_compression(
        self,
        enabled: bool = True,
        ratio: float = DEFAULT_COMPRESSION_RATIO,
        method: str = "svd"
    ) -> CacheConfigBuilder:
        """Configure compression."""
        self._config_dict["use_compression"] = enabled
        self._config_dict["compression_ratio"] = ratio
        self._config_dict["compression_method"] = method
        return self
    
    def with_quantization(
        self,
        enabled: bool = True,
        bits: int = 8
    ) -> CacheConfigBuilder:
        """Configure quantization."""
        self._config_dict["use_quantization"] = enabled
        self._config_dict["quantization_bits"] = bits
        return self
    
    def with_memory_limit(self, max_memory_mb: int) -> CacheConfigBuilder:
        """Set memory limit in MB."""
        self._config_dict["max_memory_mb"] = max_memory_mb
        return self
    
    def with_profiling(self, enabled: bool = True) -> CacheConfigBuilder:
        """Enable/disable profiling."""
        self._config_dict["enable_profiling"] = enabled
        return self
    
    def with_adaptive(self, enabled: bool = True) -> CacheConfigBuilder:
        """Enable/disable adaptive features."""
        self._config_dict["adaptive_compression"] = enabled
        self._config_dict["adaptive_quantization"] = enabled
        return self
    
    def build(self) -> KVCacheConfig:
        """Build KVCacheConfig from builder state."""
        return KVCacheConfig.from_dict(self._config_dict)


def create_default_config() -> KVCacheConfig:
    """
    Create default configuration.
    
    Returns:
        Default KVCacheConfig instance
    """
    return KVCacheConfig(
        max_tokens=DEFAULT_MAX_TOKENS,
        num_heads=DEFAULT_NUM_HEADS,
        head_dim=DEFAULT_HEAD_DIM,
    )


def create_inference_config(max_tokens: int = DEFAULT_MAX_TOKENS) -> KVCacheConfig:
    """
    Create configuration optimized for inference.
    
    Args:
        max_tokens: Maximum cache tokens
        
    Returns:
        Inference-optimized KVCacheConfig
    """
    return KVCacheConfig(
        max_tokens=max_tokens,
        cache_mode=CacheMode.INFERENCE,
        cache_strategy=CacheStrategy.ADAPTIVE,
        use_compression=True,
        compression_ratio=DEFAULT_COMPRESSION_RATIO,
        adaptive_compression=True,
    )


def create_training_config(max_tokens: int = DEFAULT_MAX_TOKENS) -> KVCacheConfig:
    """
    Create configuration optimized for training.
    
    Args:
        max_tokens: Maximum cache tokens
        
    Returns:
        Training-optimized KVCacheConfig
    """
    return KVCacheConfig(
        max_tokens=max_tokens,
        cache_mode=CacheMode.TRAINING,
        cache_strategy=CacheStrategy.LRU,
        use_compression=False,  # Disable for training accuracy
        use_quantization=False,  # Disable for training accuracy
    )


def create_memory_efficient_config(max_tokens: int = DEFAULT_MAX_TOKENS) -> KVCacheConfig:
    """
    Create configuration optimized for memory efficiency.
    
    Args:
        max_tokens: Maximum cache tokens
        
    Returns:
        Memory-efficient KVCacheConfig
    """
    return KVCacheConfig(
        max_tokens=max_tokens,
        cache_strategy=CacheStrategy.ADAPTIVE,
        use_compression=True,
        compression_ratio=0.2,  # Higher compression
        use_quantization=True,
        quantization_bits=8,
        adaptive_compression=True,
        adaptive_quantization=True,
    )


def create_high_performance_config(max_tokens: int = DEFAULT_MAX_TOKENS) -> KVCacheConfig:
    """
    Create configuration optimized for high performance.
    
    Args:
        max_tokens: Maximum cache tokens
        
    Returns:
        High-performance KVCacheConfig
    """
    return KVCacheConfig(
        max_tokens=max_tokens,
        cache_strategy=CacheStrategy.ADAPTIVE,
        use_compression=False,  # Disable for speed
        use_quantization=False,  # Disable for speed
        enable_profiling=True,
        pin_memory=True,
        non_blocking=True,
    )

