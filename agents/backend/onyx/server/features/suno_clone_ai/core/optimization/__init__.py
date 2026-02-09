"""
Optimization Module

Provides:
- Model optimization techniques
- Quantization
- Pruning
- Knowledge distillation
- Optimization utilities
"""

from .quantization import (
    Quantizer,
    quantize_model,
    dynamic_quantization,
    static_quantization
)

from .pruning import (
    Pruner,
    prune_model,
    magnitude_pruning,
    structured_pruning
)

from .knowledge_distillation import (
    KnowledgeDistiller,
    distill_model,
    create_distillation_loss
)

__all__ = [
    # Quantization
    "Quantizer",
    "quantize_model",
    "dynamic_quantization",
    "static_quantization",
    # Pruning
    "Pruner",
    "prune_model",
    "magnitude_pruning",
    "structured_pruning",
    # Knowledge distillation
    "KnowledgeDistiller",
    "distill_model",
    "create_distillation_loss"
]



