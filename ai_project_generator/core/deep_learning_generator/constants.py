"""
Constants Module for Deep Learning Generator

Centralized constants and configuration defaults.
"""

from typing import List, Dict, Any

# Supported frameworks
SUPPORTED_FRAMEWORKS: List[str] = [
    "pytorch",
    "tensorflow",
    "jax",
    "onnx",
    "paddlepaddle"  # Added for completeness
]

# Supported model types
SUPPORTED_MODEL_TYPES: List[str] = [
    "transformer",
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "gan",
    "vae",
    "diffusion",
    "llm",
    "vision_transformer",
    "bert",
    "gpt",
    "resnet",
    "efficientnet",
    "mobilenet"
]

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "framework": "pytorch",
    "use_gpu": True,
    "mixed_precision": True,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "early_stopping": True,
    "gradient_clipping": True,
    "checkpointing": True,
    "experiment_tracking": True
}

# Generator capabilities
CAPABILITIES: List[str] = [
    "model_generation",
    "training_pipelines",
    "evaluation_metrics",
    "optimization",
    "transfer_learning",
    "active_learning",
    "reinforcement_learning",
    "meta_learning",
    "automl",
    "continual_learning",
    "fine_tuning",
    "quantization",
    "pruning",
    "distributed_training"
]

# Version information
VERSION: str = "2.2.0"
MODULE_NAME: str = "deep_learning_generator"















