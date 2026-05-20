"""
Model exports for Addiction Recovery AI
"""

# Deep Learning Models
from .core.models import (
    RecoverySentimentAnalyzer,
    RecoveryProgressPredictor,
    RelapseRiskPredictor,
    LLMRecoveryCoach,
    T5RecoveryCoach,
    EnhancedAddictionAnalyzer,
    create_sentiment_analyzer,
    create_progress_predictor,
    create_relapse_predictor,
    create_llm_coach,
    create_t5_coach,
    create_enhanced_analyzer
)

# Optional model imports
DIFFUSION_AVAILABLE = False
FAST_INFERENCE_AVAILABLE = False
QUANTIZED_AVAILABLE = False

# Diffusion Models for Visualization
try:
    from .core.models.diffusion_models import (
        RecoveryProgressVisualizer,
        RecoveryChartGenerator,
        create_progress_visualizer,
        create_chart_generator
    )
    DIFFUSION_AVAILABLE = True
except ImportError:
    pass

# Fast Inference Engines
try:
    from .core.models.fast_inference import (
        FastInferenceEngine,
        CachedTransformer,
        OptimizedDataLoader,
        create_fast_engine,
        create_cached_transformer
    )
    FAST_INFERENCE_AVAILABLE = True
except ImportError:
    pass

# Quantized Models
try:
    from .core.models.quantized_models import (
        QuantizedModel,
        OptimizedTransformer,
        create_quantized_model,
        create_optimized_transformer
    )
    QUANTIZED_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "RecoverySentimentAnalyzer",
    "RecoveryProgressPredictor",
    "RelapseRiskPredictor",
    "LLMRecoveryCoach",
    "T5RecoveryCoach",
    "EnhancedAddictionAnalyzer",
    "create_sentiment_analyzer",
    "create_progress_predictor",
    "create_relapse_predictor",
    "create_llm_coach",
    "create_t5_coach",
    "create_enhanced_analyzer",
    "DIFFUSION_AVAILABLE",
    "FAST_INFERENCE_AVAILABLE",
    "QUANTIZED_AVAILABLE",
]

# Conditionally add diffusion exports
if DIFFUSION_AVAILABLE:
    __all__.extend([
        "RecoveryProgressVisualizer",
        "RecoveryChartGenerator",
        "create_progress_visualizer",
        "create_chart_generator",
    ])

# Conditionally add fast inference exports
if FAST_INFERENCE_AVAILABLE:
    __all__.extend([
        "FastInferenceEngine",
        "CachedTransformer",
        "OptimizedDataLoader",
        "create_fast_engine",
        "create_cached_transformer",
    ])

# Conditionally add quantized exports
if QUANTIZED_AVAILABLE:
    __all__.extend([
        "QuantizedModel",
        "OptimizedTransformer",
        "create_quantized_model",
        "create_optimized_transformer",
    ])

