"""
Unified Tokenization utilities for the ads training system.

This module adapts and re-exports the legacy tokenization utilities under the new
training package namespace to support gradual migration while maintaining backward compatibility.
"""

# Re-export legacy implementation for now to preserve behavior
from ..tokenization_service import (  # type: ignore
    TextPreprocessor as _LegacyTextPreprocessor,
    AdvancedTokenizer as _LegacyAdvancedTokenizer,
    SequenceManager as _LegacySequenceManager,
    OptimizedAdsDataset as _LegacyOptimizedAdsDataset,
    TokenizationService as _LegacyTokenizationService,
)

TextPreprocessor = _LegacyTextPreprocessor
AdvancedTokenizer = _LegacyAdvancedTokenizer
SequenceManager = _LegacySequenceManager
OptimizedAdsDataset = _LegacyOptimizedAdsDataset
TokenizationService = _LegacyTokenizationService

__all__ = [
    "TextPreprocessor",
    "AdvancedTokenizer",
    "SequenceManager",
    "OptimizedAdsDataset",
    "TokenizationService",
]






