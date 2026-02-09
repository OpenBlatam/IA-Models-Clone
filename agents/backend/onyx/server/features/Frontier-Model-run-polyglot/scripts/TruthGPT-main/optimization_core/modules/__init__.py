"""
Modules Package for TruthGPT Optimization Core
Modular system following deep learning best practices

This module provides organized access to module components:
- optimizers: Module optimizers (CUDA, GPU, Memory)
- advanced: Advanced optimization modules
- attention: Attention mechanisms
- embeddings: Embedding components
- feed_forward: Feed-forward networks
- model: Model components
- optimization: Optimization strategies
- training: Training components
- transformer: Transformer components
"""

from __future__ import annotations

# Direct imports for backward compatibility
try:
    from .advanced_libraries import (
        BaseModule,
        ModelModule,
        DataModule,
        TrainingModule,
        TransformerModule,
        DiffusionModule,
        LoRAModule,
        QuantizedModule,
        TextDataModule,
        ImageDataModule,
        AudioDataModule,
        SupervisedTrainingModule,
        UnsupervisedTrainingModule,
        OptimizationModule,
        AdamWOptimizationModule,
        LoRAOptimizationModule,
    )
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Could not import advanced_libraries: {e}")

# Lazy imports for organized submodules
_LAZY_IMPORTS = {
    'optimizers': '.optimizers',
    'advanced': '.advanced',
    'attention': '.attention',
    'embeddings': '.embeddings',
    'feed_forward': '.feed_forward',
    'model': '.model',
    'optimization': '.optimization',
    'training': '.training',
    'transformer': '.transformer',
    'base': '.base',
    'interface': '.interface',
    'memory': '.memory',
    'monitoring': '.monitoring',
}

_import_cache = {}

import importlib
import threading
_cache_lock = threading.RLock()


def __getattr__(name: str):
    """Lazy import system for module submodules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    with _cache_lock:
        if name in _import_cache:
            return _import_cache[name]
        
        module_path = _LAZY_IMPORTS[name]
        try:
            package = __package__ or 'modules'
            module = importlib.import_module(module_path, package=package)
            
            # Return the attribute if it exists, otherwise the module itself
            if hasattr(module, name):
                obj = getattr(module, name)
            else:
                obj = module
            
            _import_cache[name] = obj
            return obj
        except Exception as e:
            raise AttributeError(
                f"module '{__name__}' has no attribute '{name}'. "
                f"Failed to import: {e}"
            ) from e


def list_available_module_submodules() -> list[str]:
    """List all available module submodules."""
    return list(_LAZY_IMPORTS.keys())


# Backward compatible imports from optimizers (via lazy import)
try:
    from .cuda_optimizer import (
        CudaKernelConfig,
        CudaKernelType,
        CudaKernelOptimizer,
        CudaKernelManager,
        create_cuda_optimizer,
        create_cuda_kernel_manager,
        create_cuda_kernel_config
    )
except ImportError:
    pass

try:
    from .gpu_optimizer import (
        GPUOptimizationConfig,
        GPUOptimizationLevel,
        GPUOptimizer,
        GPUMemoryManager,
        create_gpu_optimizer,
        create_gpu_optimization_config,
        create_gpu_memory_manager
    )
except ImportError:
    pass

try:
    from .memory_optimizer import (
        MemoryOptimizationConfig,
        MemoryOptimizationLevel,
        MemoryOptimizer,
        MemoryProfiler,
        create_memory_optimizer,
        create_memory_optimization_config,
        create_memory_profiler
    )
except ImportError:
    pass

__all__ = [
    # Advanced Libraries
    'BaseModule',
    'ModelModule',
    'DataModule',
    'TrainingModule',
    'TransformerModule',
    'DiffusionModule',
    'LoRAModule',
    'QuantizedModule',
    
    # Submodules
    'optimizers',
    'advanced',
    'attention',
    'embeddings',
    'feed_forward',
    'model',
    'optimization',
    'training',
    'transformer',
    'base',
    'interface',
    'memory',
    'monitoring',
    'list_available_module_submodules',
]

# Version information
__version__ = "1.0.0"
__author__ = "TruthGPT Optimization Core Team"
__description__ = "Modular optimization system for TruthGPT following deep learning best practices"