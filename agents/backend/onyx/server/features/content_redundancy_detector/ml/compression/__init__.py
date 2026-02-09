"""
Model Compression Module
Pruning, distillation, and other compression techniques
"""

from .pruning import ModelPruner, PruningStrategy
from .distillation import KnowledgeDistiller, DistillationConfig

__all__ = [
    "ModelPruner",
    "PruningStrategy",
    "KnowledgeDistiller",
    "DistillationConfig",
]



