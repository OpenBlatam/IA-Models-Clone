"""
Core Module - Research Paper Code Improver
==========================================
"""

# Core utilities and base classes (always available)
from .common_utils import (
    get_device, move_to_device, calculate_model_size,
    count_parameters, estimate_flops, measure_inference_time,
    get_model_output, extract_predictions, calculate_accuracy,
    safe_forward, timing_decorator, validate_model_input,
    create_dummy_input, check_model_health
)
from .constants import (
    DEFAULT_DEVICE, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS, DEFAULT_WEIGHT_DECAY, DEFAULT_PATIENCE,
    DEFAULT_MAX_GRAD_NORM, LATENCY_THRESHOLD_MS, MEMORY_THRESHOLD_MB,
    ACCURACY_THRESHOLD, DEFAULT_SEED
)
from .base_classes import (
    BaseConfig, BaseManager, BaseTrainer, BaseEvaluator, BaseOptimizer
)
from .core_utils import (
    get_logger, ensure_dir, get_paper_storage, safe_file_operation
)
from .optional_imports import (
    optional_import, check_imports,
    get_chromadb, get_sentence_transformers,
    get_openai, get_anthropic,
    get_pymupdf, get_pdfplumber, get_pypdf2,
    get_httpx, get_requests
)
from .llm_factory import LLMFactory, LLMProvider

# Main application modules (core functionality)
from .paper_extractor import PaperExtractor
from .model_trainer import ModelTrainer
from .code_improver import CodeImprover
from .vector_store import VectorStore
from .rag_engine import RAGEngine
from .paper_storage import PaperStorage
from .code_analyzer import CodeAnalyzer
from .cache_manager import CacheManager
from .batch_processor import BatchProcessor

# Export only essential modules by default
# Other modules should be imported explicitly when needed
__all__ = [
    # Utilities
    "get_device",
    "move_to_device",
    "calculate_model_size",
    "count_parameters",
    "estimate_flops",
    "measure_inference_time",
    "get_model_output",
    "extract_predictions",
    "calculate_accuracy",
    "safe_forward",
    "timing_decorator",
    "validate_model_input",
    "create_dummy_input",
    "check_model_health",
    "get_logger",
    "ensure_dir",
    "get_paper_storage",
    "safe_file_operation",
    # Optional imports
    "optional_import",
    "check_imports",
    "get_chromadb",
    "get_sentence_transformers",
    "get_openai",
    "get_anthropic",
    "get_pymupdf",
    "get_pdfplumber",
    "get_pypdf2",
    "get_httpx",
    "get_requests",
    # LLM Factory
    "LLMFactory",
    "LLMProvider",
    # Constants
    "DEFAULT_DEVICE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_NUM_EPOCHS",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_PATIENCE",
    "DEFAULT_MAX_GRAD_NORM",
    "LATENCY_THRESHOLD_MS",
    "MEMORY_THRESHOLD_MB",
    "ACCURACY_THRESHOLD",
    "DEFAULT_SEED",
    # Base classes
    "BaseConfig",
    "BaseManager",
    "BaseTrainer",
    "BaseEvaluator",
    "BaseOptimizer",
    # Core modules
    "PaperExtractor",
    "ModelTrainer",
    "CodeImprover",
    "VectorStore",
    "RAGEngine",
    "PaperStorage",
    "CodeAnalyzer",
    "CacheManager",
    "BatchProcessor",
]

# Lazy imports for optional modules
# Import these explicitly when needed to avoid loading everything at startup
# Example: from core.test_generator import TestGenerator
