"""
Advanced compression techniques.

Provides advanced compression algorithms for cache.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple, Callable
import torch

logger = logging.getLogger(__name__)


class DeltaCompression:
    """
    Delta compression.
    
    Compresses based on differences from previous values.
    """
    
    def __init__(self):
        """Initialize delta compression."""
        self.previous_values: Dict[int, Any] = {}
    
    def compress(self, position: int, value: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Compress using delta encoding.
        
        Args:
            position: Cache position
            value: Value to compress
            
        Returns:
            Compressed value and metadata
        """
        if position in self.previous_values:
            previous = self.previous_values[position]
            
            if isinstance(value, torch.Tensor) and isinstance(previous, torch.Tensor):
                # Delta encoding
                delta = value - previous
                
                # Store delta if smaller
                if delta.numel() < value.numel():
                    metadata = {"compression": "delta", "original_shape": value.shape}
                    self.previous_values[position] = value
                    return delta, metadata
        
        # Store original
        self.previous_values[position] = value
        metadata = {"compression": "none"}
        return value, metadata
    
    def decompress(
        self,
        position: int,
        compressed: Any,
        metadata: Dict[str, Any]
    ) -> Any:
        """
        Decompress delta encoded value.
        
        Args:
            position: Cache position
            compressed: Compressed value
            metadata: Compression metadata
            
        Returns:
            Decompressed value
        """
        if metadata.get("compression") == "delta":
            if position in self.previous_values:
                previous = self.previous_values[position]
                if isinstance(compressed, torch.Tensor) and isinstance(previous, torch.Tensor):
                    return previous + compressed
        
        return compressed


class DictionaryCompression:
    """
    Dictionary compression.
    
    Uses dictionary-based compression.
    """
    
    def __init__(self, dict_size: int = 256):
        """
        Initialize dictionary compression.
        
        Args:
            dict_size: Dictionary size
        """
        self.dict_size = dict_size
        self.dictionary: Dict[int, torch.Tensor] = {}
        self.next_code = 0
    
    def compress(self, value: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Compress using dictionary.
        
        Args:
            value: Value to compress
            
        Returns:
            Compressed value and metadata
        """
        if not isinstance(value, torch.Tensor):
            return value, {"compression": "none"}
        
        # Find similar entry in dictionary
        for code, dict_value in self.dictionary.items():
            if torch.allclose(value, dict_value, atol=1e-5):
                metadata = {"compression": "dictionary", "code": code}
                return code, metadata
        
        # Add to dictionary if space
        if len(self.dictionary) < self.dict_size:
            code = self.next_code
            self.dictionary[code] = value.clone()
            self.next_code = (self.next_code + 1) % self.dict_size
            metadata = {"compression": "dictionary", "code": code, "new": True}
            return code, metadata
        
        # No compression possible
        return value, {"compression": "none"}
    
    def decompress(
        self,
        compressed: Any,
        metadata: Dict[str, Any]
    ) -> Any:
        """
        Decompress from dictionary.
        
        Args:
            compressed: Compressed value
            metadata: Compression metadata
            
        Returns:
            Decompressed value
        """
        if metadata.get("compression") == "dictionary":
            code = metadata.get("code")
            if code in self.dictionary:
                return self.dictionary[code].clone()
        
        return compressed


class PredictiveCompression:
    """
    Predictive compression.
    
    Uses prediction for compression.
    """
    
    def __init__(self, predictor: Optional[Callable] = None):
        """
        Initialize predictive compression.
        
        Args:
            predictor: Optional prediction function
        """
        self.predictor = predictor
        self.history: Dict[int, list[Any]] = {}
    
    def compress(self, position: int, value: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Compress using prediction.
        
        Args:
            position: Cache position
            value: Value to compress
            
        Returns:
            Compressed value and metadata
        """
        if not isinstance(value, torch.Tensor):
            return value, {"compression": "none"}
        
        # Predict value
        if self.predictor and position in self.history:
            predicted = self.predictor(self.history[position])
            if isinstance(predicted, torch.Tensor):
                # Store residual
                residual = value - predicted
                
                # Store if residual is smaller
                if residual.numel() == value.numel():
                    metadata = {"compression": "predictive", "residual": True}
                    if position not in self.history:
                        self.history[position] = []
                    self.history[position].append(value)
                    return residual, metadata
        
        # No compression
        metadata = {"compression": "none"}
        if position not in self.history:
            self.history[position] = []
        self.history[position].append(value)
        return value, metadata
    
    def decompress(
        self,
        position: int,
        compressed: Any,
        metadata: Dict[str, Any]
    ) -> Any:
        """
        Decompress using prediction.
        
        Args:
            position: Cache position
            compressed: Compressed value
            metadata: Compression metadata
            
        Returns:
            Decompressed value
        """
        if metadata.get("compression") == "predictive" and metadata.get("residual"):
            if self.predictor and position in self.history:
                predicted = self.predictor(self.history[position])
                if isinstance(predicted, torch.Tensor) and isinstance(compressed, torch.Tensor):
                    return predicted + compressed
        
        return compressed

