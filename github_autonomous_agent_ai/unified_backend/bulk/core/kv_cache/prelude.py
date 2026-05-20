"""
Prelude module for KV Cache.

Provides initialization and setup utilities.
"""
from __future__ import annotations

import logging
import os
import warnings

try:
    from kv_cache import __version__ as CACHE_VERSION
except ImportError:
    CACHE_VERSION = "3.0.0"

logger = logging.getLogger(__name__)


def setup_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None
) -> None:
    """
    Setup logging for KV Cache.
    
    Args:
        level: Logging level (int or string)
        format_string: Custom format string (None = use default)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    if format_string is None:
        format_string = (
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        )
    
    logging.basicConfig(
        level=level,
        format=format_string,
        force=True  # Override existing configuration
    )
    
    logger.info(f"KV Cache v{CACHE_VERSION} logging initialized")


def enable_optimizations(
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True,
    enable_cudnn_deterministic: bool = False
) -> None:
    """
    Enable PyTorch optimizations.
    
    Args:
        enable_tf32: Enable TensorFloat-32 (faster on Ampere+ GPUs)
        enable_cudnn_benchmark: Enable cuDNN benchmarking
        enable_cudnn_deterministic: Enable deterministic cuDNN (slower)
    """
    import torch
    
    # TF32 (TensorFloat-32) acceleration for Ampere+ GPUs
    if enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.debug("TF32 acceleration enabled")
    
    # cuDNN benchmarking (faster, non-deterministic)
    if enable_cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.debug("cuDNN benchmarking enabled")
    
    # cuDNN deterministic mode (slower, deterministic)
    if enable_cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("cuDNN deterministic mode enabled")


def check_environment() -> dict[str, bool | str]:
    """
    Check environment and dependencies.
    
    Returns:
        Dictionary with environment information
    """
    info: dict[str, bool | str] = {}
    
    # Check PyTorch
    try:
        import torch
        info["pytorch_available"] = True
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except ImportError:
        info["pytorch_available"] = False
    
    # Check Transformers
    try:
        import transformers
        info["transformers_available"] = True
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_available"] = False
    
    # Check PEFT
    try:
        import peft
        info["peft_available"] = True
    except ImportError:
        info["peft_available"] = False
    
    return info


def print_environment_info() -> None:
    """Print environment information."""
    info = check_environment()
    
    print("KV Cache Environment Info:")
    print(f"  Version: {CACHE_VERSION}")
    print(f"  PyTorch: {info.get('pytorch_available', False)}")
    if info.get("pytorch_available"):
        print(f"    Version: {info.get('pytorch_version', 'unknown')}")
        print(f"    CUDA: {info.get('cuda_available', False)}")
        if info.get("cuda_available"):
            print(f"      CUDA Version: {info.get('cuda_version', 'unknown')}")
            if info.get("cudnn_version"):
                print(f"      cuDNN Version: {info.get('cudnn_version')}")
    print(f"  Transformers: {info.get('transformers_available', False)}")
    if info.get("transformers_available"):
        print(f"    Version: {info.get('transformers_version', 'unknown')}")
    print(f"  PEFT: {info.get('peft_available', False)}")


def suppress_warnings(categories: list[str] | None = None) -> None:
    """
    Suppress specific warning categories.
    
    Args:
        categories: List of warning categories to suppress.
                    If None, suppresses common warnings.
    """
    if categories is None:
        categories = [
            "UserWarning",
            "FutureWarning",
        ]
    
    for category in categories:
        warnings.filterwarnings("ignore", category=category)
    
    logger.debug(f"Suppressed warnings: {categories}")


def get_cache_info() -> dict[str, str | int]:
    """
    Get KV Cache package information.
    
    Returns:
        Dictionary with package information
    """
    return {
        "version": CACHE_VERSION,
        "package_name": "kv_cache",
        "description": "High-performance KV Cache for LLMs",
    }

