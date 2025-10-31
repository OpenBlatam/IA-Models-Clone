"""
Unified Fine-tuning utilities for the ads training system.

This module adapts and re-exports the legacy fine-tuning utilities under the new
training package namespace to support gradual migration while maintaining backward compatibility.
"""

# Re-export legacy implementation for now to preserve behavior
from ..optimized_finetuning import (  # type: ignore
    OptimizedFineTuningService as _LegacyOptimizedFineTuningService,
)

OptimizedFineTuningService = _LegacyOptimizedFineTuningService

__all__ = [
    "OptimizedFineTuningService",
]






