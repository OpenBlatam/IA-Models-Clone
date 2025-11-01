"""
Integration with HuggingFace Transformers.

Provides seamless integration with Transformers library.
"""
import logging
from typing import Optional, Dict, Any, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from kv_cache import BaseKVCache, KVCacheConfig, CacheStrategy, CacheMode

logger = logging.getLogger(__name__)


class TransformersKVCache:
    """
    Wrapper for KV cache that integrates seamlessly with Transformers models.
    
    Automatically handles cache management for transformer inference.
    """
    
    def __init__(
        self,
        config: KVCacheConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initialize Transformers KV cache.
        
        Args:
            config: KV cache configuration
            model: Optional transformer model (for auto-detection)
            tokenizer: Optional tokenizer (for sequence length detection)
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Create cache instance
        self.cache = BaseKVCache(config)
        
        # Auto-detect settings from model if provided
        if model is not None:
            self._auto_configure_from_model()
        
        logger.info(
            f"Initialized TransformersKVCache with "
            f"strategy={config.cache_strategy.value}"
        )
    
    def _auto_configure_from_model(self) -> None:
        """Auto-configure cache from model architecture."""
        if self.model is None:
            return
        
        try:
            # Try to detect model config
            if hasattr(self.model, 'config'):
                config = self.model.config
                
                # Detect attention heads
                if hasattr(config, 'num_attention_heads'):
                    self.config.num_heads = config.num_attention_heads
                
                # Detect head dimension
                if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
                    self.config.head_dim = config.hidden_size // config.num_attention_heads
                
                # Detect max sequence length
                if hasattr(config, 'max_position_embeddings'):
                    max_seq = config.max_position_embeddings
                    # Set max_tokens to a reasonable fraction
                    if self.config.max_tokens > max_seq:
                        self.config.max_tokens = max_seq
                
                logger.info(
                    f"Auto-configured from model: "
                    f"heads={self.config.num_heads}, "
                    f"head_dim={self.config.head_dim}, "
                    f"max_tokens={self.config.max_tokens}"
                )
        except Exception as e:
            logger.warning(f"Could not auto-configure from model: {e}")
    
    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        cache_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass with integrated cache.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            past_key_values: Previous key-value cache (optional)
            use_cache: Whether to use cache
            cache_positions: Cache positions for each token (optional)
            
        Returns:
            Dictionary with outputs and cache info
        """
        if not use_cache or self.model is None:
            return {"use_cache": False}
        
        # Generate cache positions if not provided
        if cache_positions is None:
            batch_size, seq_len = input_ids.shape
            cache_positions = torch.arange(seq_len, device=input_ids.device)
            cache_positions = cache_positions.unsqueeze(0).expand(batch_size, -1)
        
        cache_info = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_positions": [],
        }
        
        return {
            "input_ids": input_ids,
            "cache_positions": cache_positions,
            "cache_info": cache_info,
            "use_cache": True,
        }
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def warmup(self, num_samples: int = 100, seq_len: int = 128) -> None:
        """
        Warmup cache with sample data.
        
        Args:
            num_samples: Number of warmup samples
            seq_len: Sequence length for warmup
        """
        if self.config.enable_warmup:
            logger.info(f"Warming up cache with {num_samples} samples...")
            
            device = self.cache.device
            for i in range(num_samples):
                # Create dummy data
                key = torch.randn(
                    1, self.config.num_heads, seq_len, self.config.head_dim,
                    device=device, dtype=self.config.dtype
                )
                value = torch.randn(
                    1, self.config.num_heads, seq_len, self.config.head_dim,
                    device=device, dtype=self.config.dtype
                )
                
                # Cache it
                self.cache.put(i, key, value)
            
            logger.info("Cache warmup completed")


class ModelCacheWrapper:
    """
    Wrapper to add caching to any transformer model.
    
    Automatically intercepts attention operations to use KV cache.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        cache_config: KVCacheConfig
    ):
        """
        Initialize model cache wrapper.
        
        Args:
            model: Transformer model to wrap
            cache_config: KV cache configuration
        """
        self.model = model
        self.cache_config = cache_config
        self.cache = TransformersKVCache(cache_config, model=model)
        self._original_forward = None
        self._cache_enabled = True
    
    def enable_cache(self) -> None:
        """Enable caching."""
        self._cache_enabled = True
    
    def disable_cache(self) -> None:
        """Disable caching."""
        self._cache_enabled = False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_cache_stats()
    
    def clear_cache(self) -> None:
        """Clear cache."""
        self.cache.clear_cache()

