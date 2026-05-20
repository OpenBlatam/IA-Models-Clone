"""DistributedDataParallel utilities for multi-GPU training."""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional
import logging
import os

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Setup distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend (nccl for GPU, gloo for CPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    logger.info(f"✅ DDP initialized: rank={rank}, world_size={world_size}")


def cleanup_ddp() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("✅ DDP cleaned up")


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_initialized()


def wrap_model_ddp(model: nn.Module, device_id: int) -> nn.Module:
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: Model to wrap
        device_id: Device ID for this process
    
    Returns:
        Wrapped model
    """
    if not dist.is_initialized():
        logger.warning("DDP not initialized, returning original model")
        return model
    
    model = model.to(device_id)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=False
    )
    
    logger.info(f"✅ Model wrapped with DDP on device {device_id}")
    
    return model



