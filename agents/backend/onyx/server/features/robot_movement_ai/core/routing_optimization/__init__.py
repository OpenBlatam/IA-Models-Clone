"""
Routing Optimization Package
============================

Módulos para optimización de rendimiento.
"""

from .inference_optimizer import InferenceOptimizer, compile_model, optimize_for_inference
from .quantization import QuantizedModel, quantize_model, dynamic_quantize, static_quantize
from .data_optimization import FastDataLoader, CachedDataset, PrefetchDataLoader, OptimizedDataPipeline
from .model_compilation import ModelCompiler, compile_with_torchscript, compile_with_torch_compile, benchmark_model
from .memory_optimization import MemoryOptimizer, GradientCheckpointing
from .inference_pipeline import BatchInferencePipeline, AsyncInferenceServer
from .model_serving import ModelServer, ServingConfig
from .gpu_optimization import GPUOptimizer
from .ultra_fast_inference import UltraFastInference, StreamInference, PrecompiledModel
from .kernel_fusion import FusedLinear, FusedMLP, fuse_model_layers, optimize_kernels
from .aot_compilation import AOTCompiler, OptimizedModelCache
from .batch_optimizer import DynamicBatching, PinnedMemoryBatchProcessor, BatchQueue
from .extreme_optimization import ExtremeOptimizer, InferenceCache, OptimizedInferenceEngine, VectorizedOperations
from .hardware_optimization import HardwareOptimizer, MemoryPoolOptimizer
from .jit_optimization import JITOptimizer

__all__ = [
    "InferenceOptimizer",
    "compile_model",
    "optimize_for_inference",
    "QuantizedModel",
    "quantize_model",
    "dynamic_quantize",
    "static_quantize",
    "FastDataLoader",
    "CachedDataset",
    "PrefetchDataLoader",
    "OptimizedDataPipeline",
    "ModelCompiler",
    "compile_with_torchscript",
    "compile_with_torch_compile",
    "benchmark_model",
    "MemoryOptimizer",
    "GradientCheckpointing",
    "BatchInferencePipeline",
    "AsyncInferenceServer",
    "ModelServer",
    "ServingConfig",
    "GPUOptimizer",
    "UltraFastInference",
    "StreamInference",
    "PrecompiledModel",
    "FusedLinear",
    "FusedMLP",
    "fuse_model_layers",
    "optimize_kernels",
    "AOTCompiler",
    "OptimizedModelCache",
    "DynamicBatching",
    "PinnedMemoryBatchProcessor",
    "BatchQueue",
    "ExtremeOptimizer",
    "InferenceCache",
    "OptimizedInferenceEngine",
    "VectorizedOperations",
    "HardwareOptimizer",
    "MemoryPoolOptimizer",
    "JITOptimizer"
]

# Imports condicionales
try:
    from .model_serving import create_fastapi_server
    __all__.append("create_fastapi_server")
except ImportError:
    pass

# Imports de evaluación y debugging
try:
    from ..routing_evaluation import (
        ModelEvaluator, EvaluationConfig, EvaluationMetrics,
        KFoldCrossValidator, ModelComparator, compare_models,
        EvaluationVisualizer, plot_training_curves
    )
    __all__.extend([
        "ModelEvaluator", "EvaluationConfig", "EvaluationMetrics",
        "KFoldCrossValidator", "ModelComparator", "compare_models",
        "EvaluationVisualizer", "plot_training_curves"
    ])
except ImportError:
    pass

try:
    from ..routing_debugging import (
        ModelDebugger, DebugConfig,
        GradientAnalyzer, analyze_gradients,
        ActivationAnalyzer, analyze_activations,
        NaNDetector, detect_nans
    )
    __all__.extend([
        "ModelDebugger", "DebugConfig",
        "GradientAnalyzer", "analyze_gradients",
        "ActivationAnalyzer", "analyze_activations",
        "NaNDetector", "detect_nans"
    ])
except ImportError:
    pass

