"""
Modules Package for TruthGPT Optimization Core
Modular system following deep learning best practices
"""

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
    
    # CUDA Optimizer
    'CudaKernelConfig',
    'CudaKernelType',
    'CudaKernelOptimizer',
    'CudaKernelManager',
    'create_cuda_optimizer',
    'create_cuda_kernel_manager',
    'create_cuda_kernel_config',
    
    # GPU Optimizer
    'GPUOptimizationConfig',
    'GPUOptimizationLevel',
    'GPUOptimizer',
    'GPUMemoryManager',
    'create_gpu_optimizer',
    'create_gpu_optimization_config',
    'create_gpu_memory_manager',
    
    # Memory Optimizer
    'MemoryOptimizationConfig',
    'MemoryOptimizationLevel',
    'MemoryOptimizer',
    'MemoryProfiler',
    'create_memory_optimizer',
    'create_memory_optimization_config',
    'create_memory_profiler'
]

# Version information
__version__ = "1.0.0"
__author__ = "TruthGPT Optimization Core Team"
__description__ = "Modular optimization system for TruthGPT following deep learning best practices"