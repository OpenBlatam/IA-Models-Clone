"""
TruthGPT Compiler Infrastructure
Advanced compilation and optimization system for TruthGPT models
TensorFlow-style architecture with comprehensive compiler support
"""

# Core compiler infrastructure
from .core.compiler_core import (
    CompilerCore, CompilationTarget, OptimizationLevel, CompilationResult,
    create_compiler_core, compilation_context
)

# AOT (Ahead-of-Time) Compilation
from .aot.aot_compiler import (
    AOTCompiler, AOTCompilationConfig, AOTOptimizationStrategy,
    create_aot_compiler, aot_compilation_context
)

# JIT (Just-in-Time) Compilation
from .jit.jit_compiler import (
    JITCompiler, JITCompilationConfig, JITOptimizationStrategy,
    create_jit_compiler, jit_compilation_context
)

# MLIR Compilation Infrastructure
from .mlir.mlir_compiler import (
    MLIRCompiler, MLIRDialect, MLIROptimizationPass, MLIRCompilationResult,
    create_mlir_compiler, mlir_compilation_context
)

# Plugin System
from .plugin.plugin_system import (
    CompilerPlugin, PluginManager, PluginRegistry, PluginInterface,
    create_plugin_manager, plugin_compilation_context
)

# TensorFlow to TensorRT Compilation
from .tf2tensorrt.tf2tensorrt_compiler import (
    TF2TensorRTCompiler, TensorRTConfig, TensorRTOptimizationLevel,
    create_tf2tensorrt_compiler, tf2tensorrt_compilation_context
)

# TensorFlow to XLA Compilation
from .tf2xla.tf2xla_compiler import (
    TF2XLACompiler, XLAConfig, XLAOptimizationLevel,
    create_tf2xla_compiler, tf2xla_compilation_context
)

# Compiler Utilities
from .utils.compiler_utils import (
    CompilerUtils, CodeGenerator, OptimizationAnalyzer,
    create_compiler_utils, compiler_utils_context
)

# Runtime Compilation
from .runtime.runtime_compiler import (
    RuntimeCompiler, RuntimeCompilationConfig, RuntimeOptimizationStrategy,
    RuntimeCompilationResult, RuntimeTarget, RuntimeOptimizationLevel,
    create_runtime_compiler, runtime_compilation_context
)

# Kernel Compilation
from .kernels.kernel_compiler import (
    KernelCompiler, KernelOptimizationLevel, KernelCompilationResult,
    KernelTarget, KernelConfig, KernelOptimizationPass,
    create_kernel_compiler, kernel_compilation_context
)

__all__ = [
    # Core compiler infrastructure
    'CompilerCore',
    'CompilationTarget',
    'OptimizationLevel',
    'CompilationResult',
    'create_compiler_core',
    'compilation_context',
    
    # AOT Compilation
    'AOTCompiler',
    'AOTCompilationConfig',
    'AOTOptimizationStrategy',
    'create_aot_compiler',
    'aot_compilation_context',
    
    # JIT Compilation
    'JITCompiler',
    'JITCompilationConfig',
    'JITOptimizationStrategy',
    'create_jit_compiler',
    'jit_compilation_context',
    
    # MLIR Compilation
    'MLIRCompiler',
    'MLIRDialect',
    'MLIROptimizationPass',
    'MLIRCompilationResult',
    'create_mlir_compiler',
    'mlir_compilation_context',
    
    # Plugin System
    'CompilerPlugin',
    'PluginManager',
    'PluginRegistry',
    'PluginInterface',
    'create_plugin_manager',
    'plugin_compilation_context',
    
    # TensorFlow to TensorRT
    'TF2TensorRTCompiler',
    'TensorRTConfig',
    'TensorRTOptimizationLevel',
    'create_tf2tensorrt_compiler',
    'tf2tensorrt_compilation_context',
    
    # TensorFlow to XLA
    'TF2XLACompiler',
    'XLAConfig',
    'XLAOptimizationLevel',
    'create_tf2xla_compiler',
    'tf2xla_compilation_context',
    
    # Compiler Utilities
    'CompilerUtils',
    'CodeGenerator',
    'OptimizationAnalyzer',
    'create_compiler_utils',
    'compiler_utils_context',
    
    # Runtime Compilation
    'RuntimeCompiler',
    'RuntimeCompilationConfig',
    'RuntimeOptimizationStrategy',
    'RuntimeCompilationResult',
    'RuntimeTarget',
    'RuntimeOptimizationLevel',
    'create_runtime_compiler',
    'runtime_compilation_context',
    
    # Kernel Compilation
    'KernelCompiler',
    'KernelOptimizationLevel',
    'KernelCompilationResult',
    'KernelTarget',
    'KernelConfig',
    'KernelOptimizationPass',
    'create_kernel_compiler',
    'kernel_compilation_context'
]

__version__ = "1.0.0"
