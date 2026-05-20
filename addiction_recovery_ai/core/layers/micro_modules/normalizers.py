"""
Normalizers - Ultra-Specific Normalization Components
Each normalizer in its own focused implementation
"""

from typing import Optional
import torch
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class NormalizerBase(ABC):
    """Base class for all normalizers"""
    
    def __init__(self, name: str = "Normalizer"):
        self.name = name
        self._fitted = False
    
    @abstractmethod
    def _compute_stats(self, data: torch.Tensor) -> dict:
        """Compute normalization statistics"""
        pass
    
    @abstractmethod
    def _normalize(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        """Apply normalization using statistics"""
        pass
    
    def fit(self, data: torch.Tensor):
        """Fit normalizer to data"""
        self.stats = self._compute_stats(data)
        self._fitted = True
        logger.debug(f"{self.name} fitted")
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data"""
        if not self._fitted:
            self.fit(data)
        return self._normalize(data, self.stats)
    
    def inverse_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse normalization (if applicable)"""
        raise NotImplementedError(f"{self.name} does not support inverse normalization")


class StandardNormalizer(NormalizerBase):
    """Standard normalization (z-score): (x - mean) / std"""
    
    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        super().__init__("StandardNormalizer")
        self.mean = mean
        self.std = std
    
    def _compute_stats(self, data: torch.Tensor) -> dict:
        """Compute mean and std"""
        return {
            'mean': data.mean().item() if self.mean is None else self.mean,
            'std': data.std().item() + 1e-8 if self.std is None else self.std
        }
    
    def _normalize(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        """Apply standard normalization"""
        return (data - stats['mean']) / stats['std']
    
    def inverse_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse standard normalization"""
        if not self._fitted:
            raise ValueError("Normalizer must be fitted before inverse normalization")
        return data * self.stats['std'] + self.stats['mean']


class MinMaxNormalizer(NormalizerBase):
    """Min-Max normalization: (x - min) / (max - min)"""
    
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        super().__init__("MinMaxNormalizer")
        self.min_val = min_val
        self.max_val = max_val
    
    def _compute_stats(self, data: torch.Tensor) -> dict:
        """Compute min and max"""
        return {
            'min': data.min().item() if self.min_val is None else self.min_val,
            'max': data.max().item() if self.max_val is None else self.max_val
        }
    
    def _normalize(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        """Apply min-max normalization"""
        range_val = stats['max'] - stats['min'] + 1e-8
        return (data - stats['min']) / range_val
    
    def inverse_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse min-max normalization"""
        if not self._fitted:
            raise ValueError("Normalizer must be fitted before inverse normalization")
        range_val = self.stats['max'] - self.stats['min']
        return data * range_val + self.stats['min']


class RobustNormalizer(NormalizerBase):
    """Robust normalization using median and IQR"""
    
    def __init__(self):
        super().__init__("RobustNormalizer")
    
    def _compute_stats(self, data: torch.Tensor) -> dict:
        """Compute median and IQR"""
        median = data.median().item()
        q75, q25 = torch.quantile(data, torch.tensor([0.75, 0.25])).tolist()
        iqr = q75 - q25 + 1e-8
        return {'median': median, 'iqr': iqr}
    
    def _normalize(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        """Apply robust normalization"""
        return (data - stats['median']) / stats['iqr']


class UnitVectorNormalizer(NormalizerBase):
    """Unit vector normalization (L2 norm)"""
    
    def __init__(self):
        super().__init__("UnitVectorNormalizer")
    
    def _compute_stats(self, data: torch.Tensor) -> dict:
        """Compute L2 norm"""
        return {'norm': data.norm(p=2).item() + 1e-8}
    
    def _normalize(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        """Apply unit vector normalization"""
        return data / stats['norm']


# Factory for normalizers
class NormalizerFactory:
    """Factory for creating normalizers"""
    
    _registry = {
        'standard': StandardNormalizer,
        'minmax': MinMaxNormalizer,
        'robust': RobustNormalizer,
        'unit': UnitVectorNormalizer,
    }
    
    @classmethod
    def create(cls, normalizer_type: str, **kwargs) -> NormalizerBase:
        """Create normalizer"""
        normalizer_type = normalizer_type.lower()
        if normalizer_type not in cls._registry:
            raise ValueError(f"Unknown normalizer type: {normalizer_type}")
        return cls._registry[normalizer_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, normalizer_class: type):
        """Register custom normalizer"""
        cls._registry[name.lower()] = normalizer_class


__all__ = [
    "NormalizerBase",
    "StandardNormalizer",
    "MinMaxNormalizer",
    "RobustNormalizer",
    "UnitVectorNormalizer",
    "NormalizerFactory",
]



