"""
TruthGPT Optimizers Module
=========================

TensorFlow-like optimizer implementations for TruthGPT.
Now integrated with optimization_core for enhanced performance.
"""

from .adam import Adam
from .sgd import SGD
from .rmsprop import RMSprop
from .adagrad import Adagrad
from .adamw import AdamW

# Export adapters for advanced usage
try:
    from .adapters import (
        OptimizationCoreAdapter,
        is_optimization_core_available,
        get_optimization_core_path,
        create_optimizer_from_core
    )
    from .core_detector import (
        is_module_available,
        get_available_modules
    )
    __all__ = [
        'Adam', 'SGD', 'RMSprop', 'Adagrad', 'AdamW',
        'OptimizationCoreAdapter',
        'is_optimization_core_available',
        'get_optimization_core_path',
        'create_optimizer_from_core',
        'is_module_available',
        'get_available_modules'
    ]
except ImportError:
    __all__ = ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'AdamW']





