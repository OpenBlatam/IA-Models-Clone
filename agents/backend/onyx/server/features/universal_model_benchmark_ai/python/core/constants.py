"""
Constants Module - System-wide constants.

This module centralizes all constants used across the system.
"""

"""
Constants Module - System-wide constants.

This module centralizes all constants used across the system.
Consolidated and organized for better maintainability.
"""

# ════════════════════════════════════════════════════════════════════════════════
# VERSION & METADATA
# ════════════════════════════════════════════════════════════════════════════════

VERSION = "1.0.0"
NAME = "Universal Model Benchmark AI"
DESCRIPTION = "Polyglot AI benchmarking system"

# ════════════════════════════════════════════════════════════════════════════════
# DEFAULT VALUES
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_BATCH_SIZE = 1

# ════════════════════════════════════════════════════════════════════════════════
# LIMITS
# ════════════════════════════════════════════════════════════════════════════════

MAX_TOKENS_LIMIT = 32768
MAX_BATCH_SIZE = 128
MIN_BATCH_SIZE = 1

# ════════════════════════════════════════════════════════════════════════════════
# TIMEOUTS (seconds)
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_TIMEOUT = 300
DEFAULT_RETRY_TIMEOUT = 60
DEFAULT_HEALTH_CHECK_TIMEOUT = 5

# ════════════════════════════════════════════════════════════════════════════════
# CACHE
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 3600

# ════════════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_RATE_LIMIT = 100
DEFAULT_RATE_WINDOW = 60

# ════════════════════════════════════════════════════════════════════════════════
# RETRY
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

# ════════════════════════════════════════════════════════════════════════════════
# PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_PROFILING_ENABLED = False
DEFAULT_METRICS_ENABLED = True

# ════════════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_DATA_DIR = "data"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_MODELS_DIR = "models"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_BACKUPS_DIR = "backups"

# ════════════════════════════════════════════════════════════════════════════════
# FILE EXTENSIONS
# ════════════════════════════════════════════════════════════════════════════════

CONFIG_EXTENSIONS = [".yaml", ".yml", ".json"]
MODEL_EXTENSIONS = [".bin", ".safetensors", ".pt", ".pth", ".gguf"]

# ════════════════════════════════════════════════════════════════════════════════
# SUPPORTED FORMATS
# ════════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXPORT_FORMATS = ["json", "csv", "yaml", "markdown", "html"]
SUPPORTED_SERIALIZATION_FORMATS = ["json", "pickle", "yaml", "msgpack"]

# ════════════════════════════════════════════════════════════════════════════════
# BENCHMARK NAMES
# ════════════════════════════════════════════════════════════════════════════════

BENCHMARK_NAMES = [
    "mmlu",
    "hellaswag",
    "gsm8k",
    "truthfulqa",
    "humaneval",
    "arc",
    "winogrande",
    "lambada",
]

# ════════════════════════════════════════════════════════════════════════════════
# MODEL BACKENDS
# ════════════════════════════════════════════════════════════════════════════════

SUPPORTED_BACKENDS = ["vllm", "transformers", "llama_cpp", "tensorrt_llm"]

# ════════════════════════════════════════════════════════════════════════════════
# QUANTIZATION TYPES
# ════════════════════════════════════════════════════════════════════════════════

QUANTIZATION_TYPES = ["fp32", "fp16", "bf16", "int8", "int4", "gptq", "awq"]

# ════════════════════════════════════════════════════════════════════════════════
# DEVICE TYPES
# ════════════════════════════════════════════════════════════════════════════════

DEVICE_TYPES = ["cpu", "cuda", "mps", "metal"]

# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def get_default_inference_config():
    """Get default inference configuration."""
    return {
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
    }

def get_default_benchmark_config():
    """Get default benchmark configuration."""
    return {
        "batch_size": DEFAULT_BATCH_SIZE,
        "max_samples": None,
        "timeout": DEFAULT_TIMEOUT,
    }

def get_benchmark_config(benchmark_name: str):
    """Get benchmark-specific configuration."""
    configs = {
        "mmlu": {"shots": 5, "max_samples": 1000},
        "hellaswag": {"shots": 10, "max_samples": 1000},
        "gsm8k": {"shots": 5, "max_samples": 1000},
        "truthfulqa": {"shots": 0, "max_samples": 1000},
        "humaneval": {"shots": 0, "max_samples": 164},
        "arc": {"shots": 0, "max_samples": 1000},
        "winogrande": {"shots": 0, "max_samples": 1000},
        "lambada": {"shots": 0, "max_samples": 1000},
    }
    return configs.get(benchmark_name, get_default_benchmark_config())

def get_context_length(model_name: str) -> int:
    """Get context length for model."""
    # Common context lengths
    context_lengths = {
        "llama": 4096,
        "llama2": 4096,
        "mistral": 8192,
        "mixtral": 32768,
        "gpt": 4096,
        "gpt2": 1024,
        "gpt-neo": 2048,
    }
    
    for key, length in context_lengths.items():
        if key.lower() in model_name.lower():
            return length
    
    return 4096  # Default
