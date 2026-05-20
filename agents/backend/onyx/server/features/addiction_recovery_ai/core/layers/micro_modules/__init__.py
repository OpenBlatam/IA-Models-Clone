"""
Micro-Modules - Ultra-Granular Modular Components
Each micro-module has a single, focused responsibility
Organized by specialized modules for maximum modularity
"""

# Import from specialized modules
from .normalizers import (
    NormalizerBase,
    StandardNormalizer,
    MinMaxNormalizer,
    RobustNormalizer,
    UnitVectorNormalizer,
    NormalizerFactory
)

from .tokenizers import (
    TokenizerBase,
    SimpleTokenizer,
    CharacterTokenizer,
    HuggingFaceTokenizer,
    BPETokenizer,
    TokenizerFactory
)

from .padders import (
    PadderBase,
    ZeroPadder,
    RepeatPadder,
    ReflectPadder,
    CircularPadder,
    CustomPadder,
    PadderFactory
)

from .augmenters import (
    AugmenterBase,
    NoiseAugmenter,
    DropoutAugmenter,
    ScaleAugmenter,
    ShiftAugmenter,
    FlipAugmenter,
    MixupAugmenter,
    CutoutAugmenter,
    ComposeAugmenter,
    AugmenterFactory
)

from .data_processors import (
    Validator,
    TensorValidator,
    ShapeValidator,
    RangeValidator
)

from .model_components import (
    # Specialized components
    InitializerBase,
    XavierInitializer,
    KaimingInitializer,
    OrthogonalInitializer,
    InitializerFactory,
    CompilerBase,
    TorchCompileCompiler,
    CompilerFactory,
    OptimizerBase,
    MixedPrecisionOptimizer,
    OptimizerFactory,
    QuantizerBase,
    DynamicQuantizer,
    QuantizerFactory,
    # Backward compatibility
    ModelInitializer,
    ModelCompiler,
    ModelOptimizer,
    ModelQuantizer,
)

from .training_components import (
    LossCalculator,
    GradientManager,
    LearningRateManager,
    CheckpointManager
)

from .inference_components import (
    BatchProcessor,
    CacheManager,
    OutputFormatter,
    PostProcessor
)

__all__ = [
    # Normalizers
    "NormalizerBase",
    "StandardNormalizer",
    "MinMaxNormalizer",
    "RobustNormalizer",
    "UnitVectorNormalizer",
    "NormalizerFactory",
    # Tokenizers
    "TokenizerBase",
    "SimpleTokenizer",
    "CharacterTokenizer",
    "HuggingFaceTokenizer",
    "BPETokenizer",
    "TokenizerFactory",
    # Padders
    "PadderBase",
    "ZeroPadder",
    "RepeatPadder",
    "ReflectPadder",
    "CircularPadder",
    "CustomPadder",
    "PadderFactory",
    # Augmenters
    "AugmenterBase",
    "NoiseAugmenter",
    "DropoutAugmenter",
    "ScaleAugmenter",
    "ShiftAugmenter",
    "FlipAugmenter",
    "MixupAugmenter",
    "CutoutAugmenter",
    "ComposeAugmenter",
    "AugmenterFactory",
    # Validators
    "Validator",
    "TensorValidator",
    "ShapeValidator",
    "RangeValidator",
    # Model Components - Specialized Initializers
    "InitializerBase",
    "XavierInitializer",
    "KaimingInitializer",
    "OrthogonalInitializer",
    "UniformInitializer",
    "NormalInitializer",
    "ZeroInitializer",
    "OnesInitializer",
    "InitializerFactory",
    # Model Components - Specialized Compilers
    "CompilerBase",
    "TorchCompileCompiler",
    "TorchScriptCompiler",
    "TorchScriptScriptCompiler",
    "OptimizeForInferenceCompiler",
    "CompilerFactory",
    # Model Components - Specialized Optimizers
    "OptimizerBase",
    "MixedPrecisionOptimizer",
    "TorchScriptOptimizer",
    "PruningOptimizer",
    "FuseOptimizer",
    "OptimizerFactory",
    # Model Components - Specialized Quantizers
    "QuantizerBase",
    "DynamicQuantizer",
    "StaticQuantizer",
    "QATQuantizer",
    "QuantizerFactory",
    # Loss Functions - Specialized
    "LossBase",
    "MSELoss",
    "MAELoss",
    "BCELoss",
    "CrossEntropyLoss",
    "SmoothL1Loss",
    "FocalLoss",
    "LossFactory",
    # Model Components - Backward Compatibility
    "ModelInitializer",
    "ModelCompiler",
    "ModelOptimizer",
    "ModelQuantizer",
    # Training Components
    "LossCalculator",
    "GradientManager",
    "LearningRateManager",
    "CheckpointManager",
    # Inference Components
    "BatchProcessor",
    "CacheManager",
    "OutputFormatter",
    "PostProcessor",
]

