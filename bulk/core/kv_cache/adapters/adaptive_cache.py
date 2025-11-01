"""
Adaptive cache adapter.

Extends BaseKVCache with adaptive behavior.
"""
import logging
from typing import Dict, Any, Optional, Tuple
import torch

from kv_cache.base import BaseKVCache
from kv_cache.config import KVCacheConfig, CacheStrategy

logger = logging.getLogger(__name__)


class AdaptiveKVCache(BaseKVCache):
    """
    Adaptive KV cache that adjusts strategy based on usage patterns.
    
    Extends BaseKVCache with:
    - Automatic strategy adaptation
    - Dynamic compression adjustment
    - Performance-based optimization
    """
    
    def __init__(self, config: KVCacheConfig):
        """
        Initialize adaptive cache.
        
        Args:
            config: KV cache configuration
        """
        # Ensure adaptive strategy
        if config.cache_strategy != CacheStrategy.ADAPTIVE:
            logger.warning(
                f"AdaptiveKVCache requires ADAPTIVE strategy, "
                f"got {config.cache_strategy}. Overriding."
            )
            config.cache_strategy = CacheStrategy.ADAPTIVE
        
        super().__init__(config)
        self._adaptation_interval = 100
        self._adaptation_counter = 0
        
        logger.info("Initialized AdaptiveKVCache with adaptive behavior")
    
    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_position: Optional[int] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward with adaptive behavior."""
        # Increment adaptation counter
        self._adaptation_counter += 1
        
        # Run base forward
        result = super().forward(key, value, cache_position, use_cache)
        
        # Adapt periodically
        if self._adaptation_counter % self._adaptation_interval == 0:
            self._adapt_cache_strategy()
        
        return result
    
    def _adapt_cache_strategy(self) -> None:
        """Adapt cache strategy based on statistics."""
        stats = self.get_stats()
        hit_rate = stats.get("hit_rate", 0.0)
        
        # Adapt compression based on hit rate
        if self.config.adaptive_compression:
            if hit_rate < 0.3:
                # Low hit rate, increase compression to save memory
                old_ratio = self.config.compression_ratio
                self.config.compression_ratio = max(0.1, self.config.compression_ratio * 0.95)
                logger.debug(
                    f"Adapting compression: {old_ratio:.3f} -> {self.config.compression_ratio:.3f} "
                    f"(low hit rate: {hit_rate:.2f})"
                )
                # Update compressor if it exists
                if self.compressor is not None:
                    self.compressor.compression_ratio = self.config.compression_ratio
            elif hit_rate > 0.8:
                # High hit rate, can relax compression
                old_ratio = self.config.compression_ratio
                self.config.compression_ratio = min(0.5, self.config.compression_ratio * 1.05)
                logger.debug(
                    f"Relaxing compression: {old_ratio:.3f} -> {self.config.compression_ratio:.3f} "
                    f"(high hit rate: {hit_rate:.2f})"
                )
                if self.compressor is not None:
                    self.compressor.compression_ratio = self.config.compression_ratio
    
    def adapt(self, performance_metrics: Dict[str, float]) -> None:
        """
        Adapt cache based on performance metrics.
        
        Args:
            performance_metrics: Dictionary with performance data
                - hit_rate: Cache hit rate
                - memory_usage: Memory usage ratio
                - latency: Average latency
        """
        hit_rate = performance_metrics.get("hit_rate", 0.0)
        memory_usage = performance_metrics.get("memory_usage", 0.0)
        
        # Adapt compression based on memory
        if memory_usage > 0.8 and self.config.adaptive_compression:
            old_ratio = self.config.compression_ratio
            self.config.compression_ratio = max(0.1, self.config.compression_ratio * 0.95)
            logger.info(
                f"Adapted compression ratio: {old_ratio:.3f} -> {self.config.compression_ratio:.3f} "
                f"(memory usage: {memory_usage:.2f})"
            )
            if self.compressor is not None:
                self.compressor.compression_ratio = self.config.compression_ratio
        
        # Adapt quantization based on performance
        if hit_rate < 0.5 and self.config.adaptive_quantization:
            if not self.config.use_quantization:
                self.config.use_quantization = True
                from kv_cache.quantization import Quantizer
                self.quantizer = Quantizer(
                    bits=self.config.quantization_bits,
                    use_amp=self._use_amp
                )
                logger.info("Enabled quantization for better memory efficiency")
            elif self.config.quantization_bits > 4:
                self.config.quantization_bits = max(4, self.config.quantization_bits - 2)
                if self.quantizer is not None:
                    self.quantizer.quantization_bits = self.config.quantization_bits
                logger.info(f"Reduced quantization to {self.config.quantization_bits} bits")

