"""
Base KV Cache implementation using modular components.

Follows best practices for PyTorch, Transformers, and LLM development.
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn

from kv_cache.types import TensorPair, StatsDict

from kv_cache.config import KVCacheConfig, CacheMode, CacheStrategy
from kv_cache.quantization import Quantizer
from kv_cache.compression import Compressor
from kv_cache.memory_manager import MemoryManager
from kv_cache.stats import CacheStatsTracker
from kv_cache.strategies import create_eviction_strategy
from kv_cache.device_manager import DeviceManager
from kv_cache.cache_storage import CacheStorage
from kv_cache.validators import CacheValidator
from kv_cache.utils import safe_device_transfer, validate_tensor_shapes
from kv_cache.optimizations import (
    FastQuantizer, FastCompressor, FastStorage,
    optimize_tensor_transfer, fast_tensor_validation,
    enable_torch_optimizations
)
from kv_cache.error_handler import ErrorHandler, CacheError, CacheMemoryError
from kv_cache.profiler import CacheProfiler

logger = logging.getLogger(__name__)

# Enable optimizations on import
enable_torch_optimizations()


class BaseKVCache(nn.Module):
    """
    Base class for KV cache implementations.
    
    Provides core caching functionality with error handling, GPU optimization,
    and mixed precision support following PyTorch best practices.
    
    Uses composition of modular components:
    - Quantizer for quantization
    - Compressor for compression
    - MemoryManager for memory management
    - EvictionStrategy for cache eviction
    """
    
    def __init__(self, config: KVCacheConfig):
        """
        Initialize base KV cache.
        
        Args:
            config: KV cache configuration
            
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If CUDA is requested but unavailable
        """
        super().__init__()
        
        # Validate configuration
        config.validate()
        self.config = config
        
        # Initialize modular components
        self.device_manager = DeviceManager(
            cache_mode=config.cache_mode,
            preferred_device=None  # Will auto-resolve
        )
        self.device = self.device_manager.get_device()
        
        # Initialize cache storage (modular)
        self.storage = CacheStorage()
        
        # Statistics tracking (using modular tracker)
        self.stats_tracker = CacheStatsTracker(history_size=1000)
        
        # Validator
        self.validator = CacheValidator()
        
        # Error handler
        self.error_handler = ErrorHandler(max_retries=3)
        
        # Profiler (disabled by default, enable via config)
        self.profiler = CacheProfiler(enabled=config.enable_profiling)
        
        # Mixed precision support
        self._use_amp = (
            self.device_manager.supports_mixed_precision(config.dtype) and
            config.dtype in (torch.float16, torch.bfloat16)
        )
        
        # Initialize other modular components
        self._init_components()
        
        logger.info(
            f"Initialized BaseKVCache on {self.device} with "
            f"max_tokens={config.max_tokens}, dtype={config.dtype}, "
            f"mixed_precision={self._use_amp}"
        )
    
    def _init_components(self) -> None:
        """Initialize modular components with fast implementations."""
        # Use fast quantizer for better performance
        if self.config.use_quantization:
            try:
                self.quantizer = FastQuantizer(
                    bits=self.config.quantization_bits,
                    use_amp=self._use_amp
                )
            except Exception:
                # Fallback to regular quantizer
                self.quantizer = Quantizer(
                    bits=self.config.quantization_bits,
                    use_amp=self._use_amp
                )
        else:
            self.quantizer = None
        
        # Use fast compressor for better performance
        if self.config.use_compression:
            try:
                self.compressor = FastCompressor(
                    compression_ratio=self.config.compression_ratio,
                    method=self.config.compression_method,
                    use_amp=self._use_amp
                )
            except Exception:
                # Fallback to regular compressor
                self.compressor = Compressor(
                    compression_ratio=self.config.compression_ratio,
                    method=self.config.compression_method,
                    use_amp=self._use_amp
                )
        else:
            self.compressor = None
        
        # Memory manager
        self.memory_manager = MemoryManager(self.config, self.device)
        
        # Eviction strategy
        self.eviction_strategy = create_eviction_strategy(self.config.cache_strategy)
    
    
    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_position: Optional[int] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process key-value pairs with caching.
        
        Args:
            key: Key tensor [batch, heads, seq_len, head_dim]
            value: Value tensor [batch, heads, seq_len, head_dim]
            cache_position: Optional cache position
            use_cache: Whether to use cache
        
        Returns:
            Tuple of (key, value, cache_info) where cache_info contains:
            - cached: bool indicating if cache was used
            - position: cache position if cached
            - error: error message if operation failed
        
        Raises:
            ValueError: If tensor shapes are invalid
            RuntimeError: If GPU operations fail
        """
        if not use_cache:
            return key, value, {"cached": False}
        
        # Fast tensor validation using JIT-compiled function
        try:
            if not fast_tensor_validation(key, value):
                # Fallback to full validation for error message
                is_valid, error_msg = self.validator.validate_tensors(key, value)
                if not is_valid:
                    logger.error(f"Tensor validation failed: {error_msg}")
                    return key, value, {"cached": False, "error": error_msg}
        except Exception:
            # Fallback to regular validation
            is_valid, error_msg = self.validator.validate_tensors(key, value)
            if not is_valid:
                logger.error(f"Tensor validation failed: {error_msg}")
                return key, value, {"cached": False, "error": error_msg}
        
        # Validate position if provided
        if cache_position is not None:
            is_valid, error_msg = self.validator.validate_position(cache_position)
            if not is_valid:
                logger.error(f"Position validation failed: {error_msg}")
                return key, value, {"cached": False, "error": error_msg}
        
        # Optimized tensor transfer with pinning
        try:
            key = optimize_tensor_transfer(
                key, self.device, pin_memory=self.config.pin_memory
            )
            value = optimize_tensor_transfer(
                value, self.device, pin_memory=self.config.pin_memory
            )
        except Exception as e:
            logger.error(f"Device transfer failed: {e}", exc_info=True)
            return key, value, {"cached": False, "error": str(e)}
        
        # Try to get from cache
        if cache_position is not None:
            try:
                cached = self.get(cache_position)
                if cached is not None:
                    self.stats_tracker.record_hit()
                    return cached[0], cached[1], {
                        "cached": True,
                        "position": cache_position
                    }
            except Exception as e:
                logger.warning(f"Cache retrieval failed for position {cache_position}: {e}")
        
        # Cache miss - store and return
        try:
            self.stats_tracker.record_miss()
            
            position = cache_position if cache_position is not None else self.storage.size()
            self.put(position, key, value)
            
            return key, value, {"cached": False, "position": position}
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}", exc_info=True)
            return key, value, {"cached": False, "error": str(e)}
    
    def get(self, position: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached KV at position.
        
        Args:
            position: Cache position
            
        Returns:
            Tuple of (key, value) if found, None otherwise
        """
        # Validate position
        is_valid, error_msg = self.validator.validate_position(position)
        if not is_valid:
            logger.warning(f"Invalid position {position}: {error_msg}")
            return None
        
        # Get from storage
        return self.storage.get(position)
    
    def put(
        self,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """
        Put KV into cache at position with optimizations.
        
        Args:
            position: Cache position
            key: Key tensor
            value: Value tensor
            
        Raises:
            RuntimeError: If memory allocation fails
            CacheMemoryError: If OOM persists after retries
        """
        with self.profiler.profile_operation("put"):
            try:
                # Use retry logic for put operations
                self._put_with_retry(position, key, value)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"GPU OOM during cache put at position {position}")
                    # Use error handler for retry logic
                    try:
                        return self.error_handler.handle_oom(
                            self._put_with_retry, position, key, value
                        )
                    except CacheMemoryError:
                        # Try to free memory one more time
                        self._evict_entries()
                        self.memory_manager.collect_garbage()
                        raise CacheMemoryError(f"GPU out of memory: {e}") from e
                raise
            except Exception as e:
                logger.error(f"Unexpected error in put: {e}", exc_info=True)
                self.error_handler.record_error(type(e).__name__)
                raise
    
    def _put_with_retry(self, position: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Internal put with retry logic."""
        # Optimized device transfer with pinning
        key = optimize_tensor_transfer(
            key, self.device, pin_memory=self.config.pin_memory
        )
        value = optimize_tensor_transfer(
            value, self.device, pin_memory=self.config.pin_memory
        )
        
        # Ensure correct dtype
        if key.dtype != self.config.dtype:
            key = key.to(dtype=self.config.dtype)
        if value.dtype != self.config.dtype:
            value = value.to(dtype=self.config.dtype)
        
        # Quantize if enabled
        if self.quantizer is not None:
            try:
                key, value = self.quantizer.quantize(key, value, dtype=self.config.dtype)
            except Exception as e:
                logger.warning(f"Quantization failed: {e}, continuing without quantization")
        
        # Compress if enabled
        if self.compressor is not None:
            try:
                key, value = self.compressor.compress(key, value, dtype=self.config.dtype)
            except Exception as e:
                logger.warning(f"Compression failed: {e}, continuing without compression")
        
        # Check memory limits and evict if needed
        if self.memory_manager.should_evict(self.storage.size()):
            self._evict_entries()
        
        # Store in cache
        self.storage.put(position, key, value)
        self._update_stats()
    
    def _evict_entries(self) -> None:
        """
        Evict entries based on configured strategy using modular eviction strategy.
        
        Uses thread-safe eviction following the cache strategy.
        Automatically triggers garbage collection if enabled.
        """
        cache_size = self.storage.size()
        if cache_size == 0:
            return
        
        # Calculate number of entries to evict (evict 25% or at least 1)
        num_to_evict = max(1, cache_size // 4)
        
        try:
            # Use modular eviction strategy with storage access data
            positions_to_evict = self.eviction_strategy.select_eviction_candidates(
                cache={},  # Strategy doesn't need full cache, just metadata
                access_times=self.storage.get_access_times(),
                access_counts=self.storage.get_access_counts(),
                num_to_evict=num_to_evict
            )
            
            # Evict entries using storage
            evicted = self.storage.remove(positions_to_evict)
            
            # Update eviction stats
            if evicted > 0:
                self.stats_tracker.record_eviction(count=evicted)
                logger.debug(
                    f"Evicted {evicted} entries using "
                    f"{self.config.cache_strategy.value} strategy"
                )
                    
        except Exception as e:
            logger.error(f"Error during eviction: {e}", exc_info=True)
            # Fallback: clear cache if eviction fails
            if self.storage.size() > self.config.max_tokens * 2:
                logger.warning("Cache overflow detected, clearing cache")
                self.clear()
        
        # Trigger garbage collection if enabled
        stats = self.stats_tracker.get_stats()
        if self.config.enable_gc and stats.get("evictions", 0) % 10 == 0:
            self.memory_manager.collect_garbage()
    
    def _update_stats(self) -> None:
        """Update cache statistics."""
        total_memory_mb = self.storage.get_total_memory_mb()
        # Convert to approximate element count (rough estimate)
        total_elements = int(total_memory_mb * 1024**2 / 4)  # Assuming 4 bytes per element
        self.stats_tracker.update_size(total_elements)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.storage.clear()
        self.stats_tracker.reset()
        self.memory_manager.collect_garbage()
    
    def get_stats(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            include_history: Whether to include historical data
            
        Returns:
            Dictionary with cache statistics including:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - evictions: Number of evicted entries
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - num_entries: Current number of cache entries
            - total_size: Total size in elements
            - max_tokens: Maximum tokens configured
            - memory_stats: Memory statistics
            - trend: Hit rate trend (if history included)
        """
        stats = self.stats_tracker.get_stats(include_history=include_history)
        stats.update({
            "num_entries": self.storage.size(),
            "max_tokens": self.config.max_tokens,
            "max_tokens_config": self.config.max_tokens,
            "memory_stats": self.memory_manager.get_memory_stats(),
            "storage_memory_mb": self.storage.get_total_memory_mb(),
            "device_info": self.device_manager.get_device_info(),
            "error_stats": self.error_handler.get_error_stats(),
        })
        
        if include_history:
            stats["trend"] = self.stats_tracker.get_trend()
            stats["average_hit_rate"] = self.stats_tracker.get_average_hit_rate()
        
        # Add profiling stats if enabled
        if self.profiler.enabled:
            stats["profiling"] = self.profiler.get_