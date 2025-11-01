"""
Type definitions for KV Cache.

Centralizes type hints and type aliases for better maintainability.
"""
from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any, Union, Protocol
import torch

# Type aliases for clarity
TensorPair = Tuple[torch.Tensor, torch.Tensor]
CacheEntry = Tuple[int, torch.Tensor, torch.Tensor]
CacheDict = Dict[int, TensorPair]
AccessTimesDict = Dict[int, float]
AccessCountsDict = Dict[int, int]

# Protocol for cache-like objects
class CacheLike(Protocol):
    """Protocol for cache-like objects."""
    def get(self, position: int) -> Optional[TensorPair]: ...
    def put(self, position: int, key: torch.Tensor, value: torch.Tensor) -> None: ...
    def clear(self) -> None: ...
    def get_stats(self) -> Dict[str, Any]: ...

# Statistics types
StatsDict = Dict[str, Any]
MetricsDict = Dict[str, Any]
ConfigDict = Dict[str, Any]

# Device types
DeviceType = Union[str, torch.device]

