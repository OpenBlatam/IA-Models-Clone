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
- learning: Learning strategies and optimization
"""

from __future__ import annotations

# Direct imports for backward compatibility
from .advanced_libraries import (
    OptimizationConfig,
    BaseOptimizer,
    PerformanceMonitor,
    ModelRegistry,
    ConfigManager,
    ExperimentTracker,
    create_optimization_config,
    create_performance_monitor,
    create_model_registry,
    create_config_manager,
    create_experiment_tracker
)

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
    'learning': '.learning',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for module submodules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = __import__(module_path, fromlist=[name], level=1)
        _import_cache[name] = module
        return module
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


def list_available_module_submodules() -> list[str]:
    """List all available module submodules."""
    return list(_LAZY_IMPORTS.keys())


# Backward compatible imports from optimizers (via lazy import)
from .cuda_optimizer import (
    CudaKernelConfig,
    CudaKernelType,
    CudaKernelOptimizer,
    CudaKernelManager,
    create_cuda_optimizer,
    create_cuda_kernel_manager,
    create_cuda_kernel_config
)

from .gpu_optimizer import (
    GPUOptimizationConfig,
    GPUOptimizationLevel,
    GPUOptimizer,
    GPUMemoryManager,
    create_gpu_optimizer,
    create_gpu_optimization_config,
    create_gpu_memory_manager
)

from .memory_optimizer import (
    MemoryOptimizationConfig,
    MemoryOptimizationLevel,
    MemoryOptimizer,
    MemoryProfiler,
    create_memory_optimizer,
    create_memory_optimization_config,
    create_memory_profiler
)

__all__ = [
    # Advanced Libraries
    'OptimizationConfig',
    'BaseOptimizer',
    'PerformanceMonitor',
    'ModelRegistry',
    'ConfigManager',
    'ExperimentTracker',
    'create_optimization_config',
    'create_performance_monitor',
    'create_model_registry',
    'create_config_manager',
    'create_experiment_tracker',
    
    # CUDA Optimizer (backward compatible)
    'CudaKernelConfig',
    'CudaKernelType',
    'CudaKernelOptimizer',
    'CudaKernelManager',
    'create_cuda_optimizer',
    'create_cuda_kernel_manager',
    'create_cuda_kernel_config',
    
    # GPU Optimizer (backward compatible)
    'GPUOptimizationConfig',
    'GPUOptimizationLevel',
    'GPUOptimizer',
    'GPUMemoryManager',
    'create_gpu_optimizer',
    'create_gpu_optimization_config',
    'create_gpu_memory_manager',
    
    # Memory Optimizer (backward compatible)
    'MemoryOptimizationConfig',
    'MemoryOptimizationLevel',
    'MemoryOptimizer',
    'MemoryProfiler',
    'create_memory_optimizer',
    'create_memory_optimization_config',
    'create_memory_profiler',
    
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
    'learning',
    'list_available_module_submodules',
]

# Version information
__version__ = "1.0.0"
__author__ = "TruthGPT Optimization Core Team"
__description__ = "Modular optimization system for TruthGPT following deep learning best practices"