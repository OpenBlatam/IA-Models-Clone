"""
Optimization Module - Advanced Optimization Techniques
======================================================

Provides advanced optimization techniques:
- Model quantization
- Pruning
- Knowledge distillation
- Neural architecture search (basic)
"""

from typing import Optional, Dict, Any
import torch

from .quantization import quantize_model, quantize_model_for_mobile
from .pruning import prune_model, get_pruning_sparsity, iterative_pruning
from .knowledge_distillation import KnowledgeDistillation, DistillationTrainer

__all__ = [
    "quantize_model",
    "quantize_model_for_mobile",
    "prune_model",
    "get_pruning_sparsity",
    "iterative_pruning",
    "KnowledgeDistillation",
    "DistillationTrainer",
]

