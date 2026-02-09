"""
Model Compression Module
=========================

Módulo de compresión de modelos.
"""

from .model_compressor import (
    ModelCompressor,
    KnowledgeDistillationCompressor,
    PruningCompressor,
    QuantizationCompressor,
    CompressionPipeline,
    CompressionFactory
)

__all__ = [
    'ModelCompressor',
    'KnowledgeDistillationCompressor',
    'PruningCompressor',
    'QuantizationCompressor',
    'CompressionPipeline',
    'CompressionFactory'
]








