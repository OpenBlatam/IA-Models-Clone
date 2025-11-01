"""
Interfaces and protocols for KV Cache.

Defines clear interfaces for components to ensure consistency.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Protocol
import torch

from kv_cache.types import TensorPair


class IQuantizer(ABC):
    """Interface for quantization components."""
    
    @abstractmethod
    def quantize(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype: torch.dtype = torch.float16
    ) -> TensorPair:
        """
        Quantize key and value tensors.
        
        Args:
            key: Key tensor
            value: Value tensor
            dtype: Target dtype
            
        Returns:
            Tuple of quantized (key, value)
        """
        pass


class ICompressor(ABC):
    """Interface for compression components."""
    
    @abstractmethod
    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype: torch.dtype = torch.float16
    ) -> TensorPair:
        """
        Compress key and value tensors.
        
        Args:
            key: Key tensor
            value: Value tensor
            dtype: Target dtype
            
        Returns:
            Tuple of compressed (key, value)
        """
        pass


class IStorage(ABC):
    """Interface for storage components."""
    
    @abstractmethod
    def get(self, position: int) -> Optional[TensorPair]:
        """Get cached entry at position."""
        pass
    
    @abstractmethod
    def put(
        self,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """Store entry in cache."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get number of entries."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass


class IMemoryManager(ABC):
    """Interface for memory management."""
    
    @abstractmethod
    def should_evict(self, cache_size: int) -> bool:
        """Check if eviction is needed."""
        pass
    
    @abstractmethod
    def collect_garbage(self) -> None:
        """Trigger garbage collection."""
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        pass

