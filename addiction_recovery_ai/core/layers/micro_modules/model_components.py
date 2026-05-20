"""
Model Components - Ultra-Granular Model Management
Re-exports from specialized modules for backward compatibility
"""

# Import from specialized modules
from .initializers import (
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

from .compilers import (
    CompilerBase,
    TorchCompileCompiler,
    TorchScriptCompiler,
    TorchScriptScriptCompiler,
    OptimizeForInferenceCompiler,
    CompilerFactory
)

from .optimizers import (
    OptimizerBase,
    MixedPrecisionOptimizer,
    TorchScriptOptimizer,
    PruningOptimizer,
    FuseOptimizer,
    OptimizerFactory
)

from .quantizers import (
    QuantizerBase,
    DynamicQuantizer,
    StaticQuantizer,
    QATQuantizer,
    QuantizerFactory
)

# Backward compatibility aliases
ModelInitializer = InitializerFactory
ModelCompiler = CompilerFactory
ModelOptimizer = OptimizerFactory
ModelQuantizer = QuantizerFactory

# Convenience methods for backward compatibility
class ModelInitializerCompat:
    """Compatibility wrapper for ModelInitializer"""
    
    @staticmethod
    def initialize(model, method: str = 'xavier', **kwargs):
        """Initialize model with specified method"""
        initializer = InitializerFactory.create(method, **kwargs)
        initializer.initialize(model)


class ModelCompilerCompat:
    """Compatibility wrapper for ModelCompiler"""
    
    @staticmethod
    def compile(model, mode: str = "reduce-overhead", fullgraph: bool = False):
        """Compile model"""
        compiler = CompilerFactory.create('torch_compile', mode=mode, fullgraph=fullgraph)
        return compiler.compile(model)
    
    @staticmethod
    def optimize_for_inference(model):
        """Optimize for inference"""
        compiler = CompilerFactory.create('optimize')
        return compiler.compile(model)


class ModelOptimizerCompat:
    """Compatibility wrapper for ModelOptimizer"""
    
    @staticmethod
    def enable_mixed_precision(model):
        """Enable mixed precision"""
        optimizer = OptimizerFactory.create('mixed_precision')
        return optimizer.optimize(model)
    
    @staticmethod
    def enable_torchscript(model, example_input):
        """Enable TorchScript"""
        optimizer = OptimizerFactory.create('torchscript', example_input=example_input)
        return optimizer.optimize(model)
    
    @staticmethod
    def prune_model(model, pruning_ratio: float = 0.1):
        """Prune model"""
        optimizer = OptimizerFactory.create('pruning', pruning_ratio=pruning_ratio)
        return optimizer.optimize(model)


class ModelQuantizerCompat:
    """Compatibility wrapper for ModelQuantizer"""
    
    @staticmethod
    def quantize_dynamic(model):
        """Dynamic quantization"""
        quantizer = QuantizerFactory.create('dynamic')
        return quantizer.quantize(model)
    
    @staticmethod
    def quantize_static(model, example_input):
        """Static quantization"""
        quantizer = QuantizerFactory.create('static')
        return quantizer.quantize(model)


# Export all components
__all__ = [
    # Specialized components
    "InitializerBase",
    "XavierInitializer",
    "KaimingInitializer",
    "OrthogonalInitializer",
    "UniformInitializer",
    "NormalInitializer",
    "ZeroInitializer",
    "OnesInitializer",
    "InitializerFactory",
    "CompilerBase",
    "TorchCompileCompiler",
    "TorchScriptCompiler",
    "TorchScriptScriptCompiler",
    "OptimizeForInferenceCompiler",
    "CompilerFactory",
    "OptimizerBase",
    "MixedPrecisionOptimizer",
    "TorchScriptOptimizer",
    "PruningOptimizer",
    "FuseOptimizer",
    "OptimizerFactory",
    "QuantizerBase",
    "DynamicQuantizer",
    "StaticQuantizer",
    "QATQuantizer",
    "QuantizerFactory",
    # Backward compatibility
    "ModelInitializer",
    "ModelCompiler",
    "ModelOptimizer",
    "ModelQuantizer",
    "ModelInitializerCompat",
    "ModelCompilerCompat",
    "ModelOptimizerCompat",
    "ModelQuantizerCompat",
]

