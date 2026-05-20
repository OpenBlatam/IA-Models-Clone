"""
CUDA-Specific Optimizations

Optimizations specifically for NVIDIA GPUs using CUDA.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def enable_tensor_cores() -> None:
    """
    Enable Tensor Cores for faster computation on Ampere+ GPUs.
    
    Tensor Cores provide 2-4x speedup for matrix operations.
    """
    if torch.cuda.is_available():
        # Enable TensorFloat-32 (TF32) for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Tensor Cores enabled (TF32)")


def optimize_cuda_settings() -> None:
    """
    Optimize CUDA settings for maximum performance.
    """
    if not torch.cuda.is_available():
        return
    
    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable TensorFloat-32
    enable_tensor_cores()
    
    # Set memory fraction if needed
    # torch.cuda.set_per_process_memory_fraction(0.9)
    
    logger.info("CUDA settings optimized")


def get_optimal_cuda_device() -> torch.device:
    """
    Get optimal CUDA device with best performance.
    
    Returns:
        Optimal CUDA device
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    # Select device with most memory
    device_id = 0
    max_memory = 0
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory = props.total_memory
        if memory > max_memory:
            max_memory = memory
            device_id = i
    
    device = torch.device(f"cuda:{device_id}")
    logger.info(f"Selected optimal CUDA device: {device_id} ({max_memory / 1e9:.2f} GB)")
    
    return device


def pin_memory_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pin memory for faster CPU-GPU transfer.
    
    Args:
        tensor: Tensor to pin
        
    Returns:
        Pinned tensor
    """
    if tensor.is_cuda:
        return tensor
    return tensor.pin_memory()


def async_copy_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    non_blocking: bool = True
) -> torch.Tensor:
    """
    Asynchronously copy tensor to device.
    
    Args:
        tensor: Tensor to copy
        device: Target device
        non_blocking: Non-blocking copy
        
    Returns:
        Tensor on device
    """
    return tensor.to(device, non_blocking=non_blocking)


class CUDAMemoryManager:
    """
    CUDA memory manager for efficient memory usage.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.allocated = 0
        self.reserved = 0
    
    def get_memory_info(self) -> dict:
        """Get current memory information."""
        if self.device.type != "cuda":
            return {}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved
        }
    
    def clear_cache(self) -> None:
        """Clear CUDA cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)


def compile_with_triton(model: torch.nn.Module) -> torch.nn.Module:
    """
    Compile model with Triton for maximum speed (experimental).
    
    Requires PyTorch 2.0+ and Triton.
    
    Args:
        model: PyTorch model
        
    Returns:
        Compiled model
    """
    try:
        if hasattr(torch, 'compile'):
            # Use maximum optimization
            compiled = torch.compile(
                model,
                mode="max-autotune",
                fullgraph=True
            )
            logger.info("Model compiled with Triton (max-autotune)")
            return compiled
        else:
            logger.warning("torch.compile not available")
            return model
    except Exception as e:
        logger.warning(f"Triton compilation failed: {e}")
        return model













