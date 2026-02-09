"""
Model Optimization Micro-Modules
Organized by category for maximum modularity
"""

from ..initializers import (
    InitializerBase,
    XavierInitializer,
    KaimingInitializer,
    OrthogonalInitializer,
    UniformInitializer,
    NormalInitializer,
    ZeroInitializer,
    OnesInitializer,
    InitializerFactory
)

from ..compilers import (
    CompilerBase,
    TorchCompileCompiler,
    TorchScriptCompiler,
    TorchScriptScriptCompiler,
    OptimizeForInferenceCompiler,
    CompilerFactory
)

from ..optimizers import (
    OptimizerBase,
    MixedPrecisionOptimizer,
    TorchScriptOptimizer,
    PruningOptimizer,
    FuseOptimizer,
    OptimizerFactory
)

from ..quantizers import (
    QuantizerBase,
    DynamicQuantizer,
    StaticQuantizer,
    QATQuantizer,
    QuantizerFactory
)

# Backward compatibility
from ..model_components import (
    ModelInitializer,
    ModelCompiler,
    ModelOptimizer,
    ModelQuantizer,
)

__all__ = [
    # Initializers
    "InitializerBase",
    "XavierInitializer",
    "KaimingInitializer",
    "OrthogonalInitializer",
    "UniformInitializer",
    "NormalInitializer",
    "ZeroInitializer",
    "OnesInitializer",
    "InitializerFactory",
    # Compilers
    "CompilerBase",
    "TorchCompileCompiler",
    "TorchScriptCompiler",
    "TorchScriptScriptCompiler",
    "OptimizeForInferenceCompiler",
    "CompilerFactory",
    # Optimizers
    "OptimizerBase",
    "MixedPrecisionOptimizer",
    "TorchScriptOptimizer",
    "PruningOptimizer",
    "FuseOptimizer",
    "OptimizerFactory",
    # Quantizers
    "QuantizerBase",
    "DynamicQuantizer",
    "StaticQuantizer",
    "QATQuantizer",
    "QuantizerFactory",
    # Backward Compatibility
    "ModelInitializer",
    "ModelCompiler",
    "ModelOptimizer",
    "ModelQuantizer",
]



