"""
Addition Removal AI - Sistema de IA para Adiciones y Eliminaciones
Enhanced with PyTorch, Transformers, and Diffusion Models
"""

__version__ = "2.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"

from .main import main

# Core components
from .core.editor import ContentEditor

# Enhanced AI Engine
from .core.models.enhanced_ai_engine import EnhancedAIEngine

# Models
from .core.models import (
    TransformerContentAnalyzer,
    SentimentTransformerAnalyzer,
    NERTransformerAnalyzer,
    TextGenerator,
    T5ContentGenerator,
    DiffusionContentGenerator,
    create_transformer_analyzer,
    create_text_generator,
    create_t5_generator,
)

# Gradio Interface
from .utils.gradio_interface import (
    AdditionRemovalGradioInterface,
    create_gradio_app
)

# Fast/Optimized components
from .core.fast_ai_engine import FastAIEngine, create_fast_ai_engine
from .core.fast_editor import FastContentEditor, create_fast_editor
from .core.models.fast_models import create_fast_analyzer, optimize_model_for_inference
from .utils.fast_inference import FastInferenceEngine, BatchProcessor

# Training
from .training import ModelTrainer, LoRATrainer, create_lora_model
from .training.fast_trainer import FastModelTrainer, create_fast_trainer
from .config.training_config import Config, create_default_config_file

# Fast utilities
from .utils.fast_data_loader import (
    create_fast_dataloader,
    FastDataset,
    optimize_dataloader
)
from .core.models.optimized_inference import (
    compile_model,
    optimize_for_inference_fast,
    create_fast_inference_model
)

# Evaluation and utilities
from .training.evaluator import ModelEvaluator, EarlyStopping, create_evaluator
from .training.distributed_trainer import DistributedTrainer, create_distributed_trainer
from .utils.profiler import PerformanceProfiler, profile_model, create_profiler
from .utils.advanced_logging import StructuredLogger, create_logger

# Ultra-fast optimizations
from .core.models.onnx_optimizer import (
    ONNXOptimizer,
    ONNXInference,
    export_model_to_onnx
)
from .utils.async_inference import (
    AsyncInferenceEngine,
    FastAsyncInference,
    create_async_engine
)
from .core.models.quantization import (
    AdvancedQuantization,
    quantize_model_advanced
)

# Enhanced utilities
from .utils.enhanced_gradio import EnhancedGradioInterface, create_enhanced_gradio_app
from .utils.error_handler import ErrorHandler, create_error_handler
from .utils.data_validator import DataValidator, create_validator

# Ultra-fast optimizations
from .core.ultra_fast_engine import UltraFastAIEngine, create_ultra_fast_engine
from .core.models.extreme_optimization import (
    ModelPruner,
    KnowledgeDistillation,
    MemoryOptimizer,
    PipelineOptimizer,
    prune_model,
    optimize_memory_usage
)
from .utils.precomputation import (
    EmbeddingCache,
    PrecomputedFeatures,
    create_embedding_cache
)

__all__ = [
    "main",
    "ContentEditor",
    "EnhancedAIEngine",
    "TransformerContentAnalyzer",
    "SentimentTransformerAnalyzer",
    "NERTransformerAnalyzer",
    "TextGenerator",
    "T5ContentGenerator",
    "DiffusionContentGenerator",
    "create_transformer_analyzer",
    "create_text_generator",
    "create_t5_generator",
    "AdditionRemovalGradioInterface",
    "create_gradio_app",
    # Fast/Optimized
    "FastAIEngine",
    "create_fast_ai_engine",
    "FastContentEditor",
    "create_fast_editor",
    "create_fast_analyzer",
    "optimize_model_for_inference",
    "FastInferenceEngine",
    "BatchProcessor",
    # Training
    "ModelTrainer",
    "FastModelTrainer",
    "create_fast_trainer",
    "LoRATrainer",
    "create_lora_model",
    "Config",
    "create_default_config_file",
    # Fast utilities
    "create_fast_dataloader",
    "FastDataset",
    "optimize_dataloader",
    "compile_model",
    "optimize_for_inference_fast",
    "create_fast_inference_model",
    # Evaluation and utilities
    "ModelEvaluator",
    "EarlyStopping",
    "create_evaluator",
    "DistributedTrainer",
    "create_distributed_trainer",
    "PerformanceProfiler",
    "profile_model",
    "create_profiler",
    "StructuredLogger",
    "create_logger",
    # Ultra-fast optimizations
    "ONNXOptimizer",
    "ONNXInference",
    "export_model_to_onnx",
    "AsyncInferenceEngine",
    "FastAsyncInference",
    "create_async_engine",
    "AdvancedQuantization",
    "quantize_model_advanced",
    # Enhanced utilities
    "EnhancedGradioInterface",
    "create_enhanced_gradio_app",
    "ErrorHandler",
    "create_error_handler",
    "DataValidator",
    "create_validator",
    # Ultra-fast optimizations
    "UltraFastAIEngine",
    "create_ultra_fast_engine",
    "ModelPruner",
    "KnowledgeDistillation",
    "MemoryOptimizer",
    "PipelineOptimizer",
    "prune_model",
    "optimize_memory_usage",
    "EmbeddingCache",
    "PrecomputedFeatures",
    "create_embedding_cache",
]


