"""
Performance Configuration

Optimizations and configurations for maximum performance using best libraries.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def configure_pytorch_performance():
    """
    Configure PyTorch for maximum performance.
    
    Applies best practices for PyTorch optimization.
    """
    try:
        # Set matmul precision for better performance (A100, H100)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')  # or 'medium'
            logger.info("PyTorch float32 matmul precision set to 'high'")
        
        # Enable cuDNN benchmarking for consistent input sizes
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmarking enabled")
        
        # Enable deterministic mode if needed (slower but reproducible)
        # torch.use_deterministic_algorithms(True)
        
        logger.info("PyTorch performance optimizations applied")
    except Exception as e:
        logger.warning(f"Failed to configure PyTorch performance: {e}")


def configure_transformers_performance():
    """
    Configure Transformers library for maximum performance.
    
    Returns:
        dict: Configuration for transformers
    """
    config = {
        "use_fast": True,  # Use Rust-based tokenizers
        "low_cpu_mem_usage": True,  # Reduce memory usage
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    
    logger.info("Transformers performance configuration ready")
    return config


def get_optimal_batch_size(
    model_size: str = "base",
    available_memory: Optional[float] = None
) -> int:
    """
    Get optimal batch size based on model and available memory.
    
    Args:
        model_size: Model size (small, base, large, xl)
        available_memory: Available GPU memory in GB (optional)
        
    Returns:
        Recommended batch size
    """
    # Default batch sizes by model size
    defaults = {
        "small": 32,
        "base": 16,
        "large": 8,
        "xl": 4
    }
    
    if available_memory:
        # Adjust based on available memory
        if available_memory >= 24:  # A100, H100
            return defaults.get(model_size, 16) * 2
        elif available_memory >= 16:  # V100, RTX 3090
            return defaults.get(model_size, 16)
        elif available_memory >= 8:  # RTX 3080
            return max(1, defaults.get(model_size, 16) // 2)
        else:
            return max(1, defaults.get(model_size, 16) // 4)
    
    return defaults.get(model_size, 16)


def configure_mixed_precision():
    """
    Configure mixed precision training for better performance.
    
    Returns:
        GradScaler for mixed precision
    """
    if torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("Mixed precision training configured")
        return scaler
    return None













