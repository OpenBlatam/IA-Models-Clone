"""
Cache transformation pipeline.

Provides transformation capabilities for cache data.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Callable, List
import torch

logger = logging.getLogger(__name__)


class CacheTransformer:
    """
    Base cache transformer.
    
    Provides base class for transformations.
    """
    
    def transform_key(self, key: Any) -> Any:
        """
        Transform cache key.
        
        Args:
            key: Original key
            
        Returns:
            Transformed key
        """
        return key
    
    def transform_value(self, value: Any) -> Any:
        """
        Transform cache value.
        
        Args:
            value: Original value
            
        Returns:
            Transformed value
        """
        return value
    
    def transform_key_value(self, key: Any, value: Any) -> tuple:
        """
        Transform key-value pair.
        
        Args:
            key: Original key
            value: Original value
            
        Returns:
            Transformed (key, value) tuple
        """
        return self.transform_key(key), self.transform_value(value)


class TransformationPipeline:
    """
    Transformation pipeline.
    
    Manages chain of transformations.
    """
    
    def __init__(self):
        """Initialize pipeline."""
        self.transformers: List[CacheTransformer] = []
    
    def add_transformer(self, transformer: CacheTransformer) -> None:
        """
        Add transformer to pipeline.
        
        Args:
            transformer: Transformer instance
        """
        self.transformers.append(transformer)
        logger.info(f"Added transformer: {transformer.__class__.__name__}")
    
    def transform(self, key: Any, value: Any) -> tuple:
        """
        Transform key-value pair.
        
        Args:
            key: Original key
            value: Original value
            
        Returns:
            Transformed (key, value) tuple
        """
        for transformer in self.transformers:
            key, value = transformer.transform_key_value(key, value)
        return key, value
    
    def transform_key(self, key: Any) -> Any:
        """
        Transform key.
        
        Args:
            key: Original key
            
        Returns:
            Transformed key
        """
        for transformer in self.transformers:
            key = transformer.transform_key(key)
        return key
    
    def transform_value(self, value: Any) -> Any:
        """
        Transform value.
        
        Args:
            value: Original value
            
        Returns:
            Transformed value
        """
        for transformer in self.transformers:
            value = transformer.transform_value(value)
        return value


class NormalizationTransformer(CacheTransformer):
    """Normalization transformer."""
    
    def __init__(self, norm: float = 1.0):
        """
        Initialize normalization transformer.
        
        Args:
            norm: Normalization factor
        """
        self.norm = norm
    
    def transform_value(self, value: Any) -> Any:
        """Normalize tensor value."""
        if isinstance(value, torch.Tensor):
            return value / self.norm
        if isinstance(value, tuple):
            key, val = value
            if isinstance(key, torch.Tensor) and isinstance(val, torch.Tensor):
                return (key / self.norm, val / self.norm)
        return value


class QuantizationTransformer(CacheTransformer):
    """Quantization transformer."""
    
    def __init__(self, bits: int = 8):
        """
        Initialize quantization transformer.
        
        Args:
            bits: Number of bits
        """
        self.bits = bits
    
    def transform_value(self, value: Any) -> Any:
        """Quantize tensor value."""
        if isinstance(value, torch.Tensor):
            scale = (2 ** (self.bits - 1)) - 1
            return (value * scale).round().clamp(-scale, scale) / scale
        if isinstance(value, tuple):
            key, val = value
            if isinstance(key, torch.Tensor) and isinstance(val, torch.Tensor):
                scale = (2 ** (self.bits - 1)) - 1
                return (
                    (key * scale).round().clamp(-scale, scale) / scale,
                    (val * scale).round().clamp(-scale, scale) / scale
                )
        return value


class CompressionTransformer(CacheTransformer):
    """Compression transformer."""
    
    def __init__(self, compression_ratio: float = 0.5):
        """
        Initialize compression transformer.
        
        Args:
            compression_ratio: Compression ratio
        """
        self.compression_ratio = compression_ratio
    
    def transform_value(self, value: Any) -> Any:
        """Compress tensor value."""
        if isinstance(value, torch.Tensor):
            # Simple compression: take subset
            keep = int(value.numel() * self.compression_ratio)
            return value.flatten()[:keep].reshape(-1)
        if isinstance(value, tuple):
            key, val = value
            if isinstance(key, torch.Tensor) and isinstance(val, torch.Tensor):
                key_keep = int(key.numel() * self.compression_ratio)
                val_keep = int(val.numel() * self.compression_ratio)
                return (
                    key.flatten()[:key_keep].reshape(-1),
                    val.flatten()[:val_keep].reshape(-1)
                )
        return value


class CustomTransformer(CacheTransformer):
    """Custom transformer using function."""
    
    def __init__(self, transform_fn: Callable[[Any], Any]):
        """
        Initialize custom transformer.
        
        Args:
            transform_fn: Transformation function
        """
        self.transform_fn = transform_fn
    
    def transform_value(self, value: Any) -> Any:
        """Apply custom transformation."""
        return self.transform_fn(value)

