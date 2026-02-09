"""
Layered architecture exports for Addiction Recovery AI
"""

LAYERED_ARCHITECTURE_AVAILABLE = False
MICRO_MODULES_AVAILABLE = False

# Ultra-Modular Layered Architecture
try:
    from .core.layers import (
        # Data Layer
        DataProcessor,
        NormalizationProcessor,
        TokenizationProcessor,
        PaddingProcessor,
        DataValidator,
        DatasetFactory,
        DataPipeline,
        DataLoaderFactory,
        # Model Layer
        ModelConfig,
        ModelRegistry,
        ModelBuilder,
        ModelFactory,
        ModelLoader,
        # Training Layer
        TrainingConfig,
        OptimizerFactory,
        SchedulerFactory,
        TrainingPipeline,
        TrainerFactory,
        # Inference Layer
        InferenceEngine,
        BatchProcessor,
        PredictorFactory,
        InferencePipeline,
        # Service Layer
        ServiceConfig,
        ServiceRegistry,
        ServiceContainer,
        ServiceFactory,
        # Interface Layer
        RequestProcessor,
        ResponseFormatter,
        APIHandler,
        InterfaceFactory,
        # Dependency Injection
        DependencyContainer,
        get_container,
        reset_container,
        inject_dependencies,
        register_service,
    )
    LAYERED_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Layered architecture not available: {e}")

# Micro-Modules - Ultra-Granular Components
if LAYERED_ARCHITECTURE_AVAILABLE:
    try:
        from .core.layers.micro_modules import (
            # Normalizers
            NormalizerBase,
            StandardNormalizer,
            MinMaxNormalizer,
            RobustNormalizer,
            UnitVectorNormalizer,
            NormalizerFactory,
            # Tokenizers
            TokenizerBase,
            SimpleTokenizer,
            CharacterTokenizer,
            HuggingFaceTokenizer,
            BPETokenizer,
            TokenizerFactory,
            # Padders
            PadderBase,
            ZeroPadder,
            RepeatPadder,
            ReflectPadder,
            CircularPadder,
            CustomPadder,
            PadderFactory,
            # Augmenters
            AugmenterBase,
            NoiseAugmenter,
            DropoutAugmenter,
            ScaleAugmenter,
            ShiftAugmenter,
            FlipAugmenter,
            MixupAugmenter,
            CutoutAugmenter,
            ComposeAugmenter,
            AugmenterFactory,
            # Validators
            Validator,
            TensorValidator,
            ShapeValidator,
            RangeValidator,
            # Model Components - Initializers
            InitializerBase,
            XavierInitializer,
            KaimingInitializer,
            OrthogonalInitializer,
            UniformInitializer,
            NormalInitializer,
            ZeroInitializer,
            OnesInitializer,
            InitializerFactory,
            # Model Components - Compilers
            CompilerBase,
            TorchCompileCompiler,
            TorchScriptCompiler,
            TorchScriptScriptCompiler,
            OptimizeForInferenceCompiler,
            CompilerFactory,
            # Model Components - Optimizers
            OptimizerBase,
            MixedPrecisionOptimizer,
            TorchScriptOptimizer,
            PruningOptimizer,
            FuseOptimizer,
            OptimizerFactory,
            # Model Components - Quantizers
            QuantizerBase,
            DynamicQuantizer,
            StaticQuantizer,
            QATQuantizer,
            QuantizerFactory,
            # Loss Functions
            LossBase,
            MSELoss,
            MAELoss,
            BCELoss,
            CrossEntropyLoss,
            SmoothL1Loss,
            FocalLoss,
            LossFactory,
            # Model Components - Backward Compatibility
            ModelInitializer,
            ModelCompiler,
            ModelOptimizer,
            ModelQuantizer,
            # Training Components
            LossCalculator,
            GradientManager,
            LearningRateManager,
            CheckpointManager,
            # Inference Components
            BatchProcessor,
            CacheManager,
            OutputFormatter,
            PostProcessor,
        )
        MICRO_MODULES_AVAILABLE = True
    except ImportError:
        pass

__all__ = [
    "LAYERED_ARCHITECTURE_AVAILABLE",
    "MICRO_MODULES_AVAILABLE",
]

# Conditionally add layer exports
if LAYERED_ARCHITECTURE_AVAILABLE:
    __all__.extend([
        "DataProcessor",
        "NormalizationProcessor",
        "TokenizationProcessor",
        "PaddingProcessor",
        "DataValidator",
        "DatasetFactory",
        "DataPipeline",
        "DataLoaderFactory",
        "ModelConfig",
        "ModelRegistry",
        "ModelBuilder",
        "ModelFactory",
        "ModelLoader",
        "TrainingConfig",
        "OptimizerFactory",
        "SchedulerFactory",
        "TrainingPipeline",
        "TrainerFactory",
        "InferenceEngine",
        "BatchProcessor",
        "PredictorFactory",
        "InferencePipeline",
        "ServiceConfig",
        "ServiceRegistry",
        "ServiceContainer",
        "ServiceFactory",
        "RequestProcessor",
        "ResponseFormatter",
        "APIHandler",
        "InterfaceFactory",
        "DependencyContainer",
        "get_container",
        "reset_container",
        "inject_dependencies",
        "register_service",
    ])

# Conditionally add micro-module exports
if MICRO_MODULES_AVAILABLE:
    __all__.extend([
        "NormalizerBase",
        "StandardNormalizer",
        "MinMaxNormalizer",
        "RobustNormalizer",
        "UnitVectorNormalizer",
        "NormalizerFactory",
        "TokenizerBase",
        "SimpleTokenizer",
        "CharacterTokenizer",
        "HuggingFaceTokenizer",
        "BPETokenizer",
        "TokenizerFactory",
        "PadderBase",
        "ZeroPadder",
        "RepeatPadder",
        "ReflectPadder",
        "CircularPadder",
        "CustomPadder",
        "PadderFactory",
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
        "Validator",
        "TensorValidator",
        "ShapeValidator",
        "RangeValidator",
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
        "LossBase",
        "MSELoss",
        "MAELoss",
        "BCELoss",
        "CrossEntropyLoss",
        "SmoothL1Loss",
        "FocalLoss",
        "LossFactory",
        "ModelInitializer",
        "ModelCompiler",
        "ModelOptimizer",
        "ModelQuantizer",
        "LossCalculator",
        "GradientManager",
        "LearningRateManager",
        "CheckpointManager",
        "CacheManager",
        "OutputFormatter",
        "PostProcessor",
    ])

