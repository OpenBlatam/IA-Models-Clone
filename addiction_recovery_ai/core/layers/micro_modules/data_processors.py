"""
Data Processors - Ultra-Granular Data Processing Components
Re-exports from specialized modules for backward compatibility
"""

# Import from specialized modules
from .normalizers import (
    NormalizerBase as Normalizer,
    StandardNormalizer,
    MinMaxNormalizer,
    RobustNormalizer,
    UnitVectorNormalizer,
    NormalizerFactory
)

from .tokenizers import (
    TokenizerBase as Tokenizer,
    SimpleTokenizer,
    CharacterTokenizer,
    HuggingFaceTokenizer,
    BPETokenizer,
    TokenizerFactory
)

from .padders import (
    PadderBase as Padder,
    ZeroPadder,
    RepeatPadder,
    ReflectPadder,
    CircularPadder,
    CustomPadder,
    PadderFactory
)

from .augmenters import (
    AugmenterBase as Augmenter,
    NoiseAugmenter,
    DropoutAugmenter,
    ScaleAugmenter,
    ShiftAugmenter,
    FlipAugmenter,
    MixupAugmenter,
    CutoutAugmenter,
    ComposeAugmenter,
    AugmenterFactory
)

# Validators (keep here for now)
from typing import Any
import torch
from abc import ABC, abstractmethod

class Validator(ABC):
    """Base validator interface"""
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data"""
        pass


class TensorValidator(Validator):
    """Validate tensor data"""
    
    def __init__(self, check_nan: bool = True, check_inf: bool = True):
        self.check_nan = check_nan
        self.check_inf = check_inf
    
    def validate(self, data: torch.Tensor) -> bool:
        """Validate tensor"""
        if not isinstance(data, torch.Tensor):
            return False
        
        if self.check_nan and torch.isnan(data).any():
            return False
        
        if self.check_inf and torch.isinf(data).any():
            return False
        
        return True


class ShapeValidator(Validator):
    """Validate tensor shape"""
    
    def __init__(self, expected_shape: tuple):
        self.expected_shape = expected_shape
    
    def validate(self, data: torch.Tensor) -> bool:
        """Validate shape"""
        if not isinstance(data, torch.Tensor):
            return False
        return data.shape == self.expected_shape


class RangeValidator(Validator):
    """Validate value range"""
    
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, data: torch.Tensor) -> bool:
        """Validate range"""
        if not isinstance(data, torch.Tensor):
            return False
        return (data >= self.min_val).all() and (data <= self.max_val).all()


# Export all processors
__all__ = [
    # Normalizers
    "Normalizer",
    "StandardNormalizer",
    "MinMaxNormalizer",
    "RobustNormalizer",
    "UnitVectorNormalizer",
    "NormalizerFactory",
    # Tokenizers
    "Tokenizer",
    "SimpleTokenizer",
    "CharacterTokenizer",
    "HuggingFaceTokenizer",
    "BPETokenizer",
    "TokenizerFactory",
    # Padders
    "Padder",
    "ZeroPadder",
    "RepeatPadder",
    "ReflectPadder",
    "CircularPadder",
    "CustomPadder",
    "PadderFactory",
    # Augmenters
    "Augmenter",
    "NoiseAugmenter",
    "DropoutAugmenter",
    "ScaleAugmenter",
    "ShiftAugmenter",
    "FlipAugmenter",
    "MixupAugmenter",
    "CutoutAugmenter",
    "ComposeAugmenter",
    "AugmenterFactory",
    # Validators
    "Validator",
    "TensorValidator",
    "ShapeValidator",
    "RangeValidator",
]

