"""
Distributed Training Utilities

Support for multi-GPU and multi-node training with best practices.
"""

import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional
import os

logger = logging.getLogger(__name__)


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None
) -> tuple[int, int, int]:
    """
    Initialize distributed training.
    
    Args:
        backend: Backend to use (nccl for GPU, gloo for CPU)
        init_method: Initialization method (env:// for environment variables)
        
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if not dist.is_available():
        logger.warning("Distributed training not available")
        return 0, 1, 0
    
    # Get environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size == 1:
        logger.info("Single process, distributed training not needed")
        return rank, world_size, local_rank
    
    # Initialize process group
    if init_method is None:
        init_method = "env://"
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    logger.info(
        f"Distributed training initialized: rank={rank}, "
        f"world_size={world_size}, local_rank={local_rank}"
    )
    
    return rank, world_size, local_rank


def wrap_model_ddp(
    model: nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True
) -> nn.Module:
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: PyTorch model
        device: Device to use
        find_unused_parameters: Find unused parameters (slower but more flexible)
        gradient_as_bucket_view: Use gradient as bucket view (memory efficient)
        
    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, returning original model")
        return model.to(device)
    
    model = model.to(device)
    
    ddp_model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        broadcast_buffers=True
    )
    
    logger.info("Model wrapped with DistributedDataParallel")
    
    return ddp_model


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce tensor and compute mean across all processes.
    
    Args:
        tensor: Tensor to reduce
        
    Returns:
        Mean across all processes
    """
    if not dist.is_initialized():
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    
    return tensor


def gather_tensors(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    return gathered


def is_main_process() -> bool:
    """Check if current process is main process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")













