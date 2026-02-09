"""
Model Optimization Module
=========================

Módulo de optimización de modelos.
"""

from .quantization import (
    Quantizer,
    DynamicQuantizer,
    StaticQuantizer,
    BitsAndBytesQuantizer,
    QuantizationFactory,
    quantize_model
)
from .pruning import (
    Pruner,
    MagnitudePruner,
    LotteryTicketPruner,
    PruningFactory,
    prune_model
)

__all__ = [
    'Quantizer',
    'DynamicQuantizer',
    'StaticQuantizer',
    'BitsAndBytesQuantizer',
    'QuantizationFactory',
    'quantize_model',
    'Pruner',
    'MagnitudePruner',
    'LotteryTicketPruner',
    'PruningFactory',
    'prune_model'
]








