"""
Distributed Training Utilities
================================

Utilities for setting up and managing distributed training.

Author: BUL System
Date: 2024
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def setup_distributed_training(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> Dict[str, Any]:
    """
    Setup distributed training environment.
    
    Args:
        backend: Distributed backend (nccl, gloo, mpi)
        init_method: Initialization method (env://, file://, tcp://)
        world_size: Number of processes (auto-detect if None)
        rank: Rank of this process (auto-detect if None)
        
    Returns:
        Dictionary with distributed training configuration
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        logger.warning("PyTorch not available, distributed training not supported")
        return {"distributed": False}
    
    # Check if already initialized
    if dist.is_initialized():
        logger.info("Distributed training already initialized")
        return {
            "distributed": True,
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
            "backend": dist.get_backend(),
        }
    
    # Auto-detect from environment variables
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    
    if init_method is None:
        init_method = os.environ.get("INIT_METHOD", "env://")
    
    # Only initialize if multi-GPU
    if world_size > 1:
        try:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )
            logger.info(
                f"Distributed training initialized: "
                f"world_size={dist.get_world_size()}, "
                f"rank={dist.get_rank()}, "
                f"backend={dist.get_backend()}"
            )
            return {
                "distributed": True,
                "world_size": dist.get_world_size(),
                "rank": dist.get_rank(),
                "backend": dist.get_backend(),
            }
        except Exception as e:
            logger.warning(f"Failed to initialize distributed training: {e}")
            return {"distributed": False, "error": str(e)}
    else:
        logger.info("Single GPU/CPU detected, distributed training not needed")
        return {"distributed": False}


def is_distributed() -> bool:
    """
    Check if distributed training is active.
    
    Returns:
        True if distributed training is initialized
    """
    try:
        import torch.distributed as dist
        return dist.is_initialized()
    except ImportError:
        return False


def get_world_size() -> int:
    """
    Get world size (number of processes).
    
    Returns:
        World size or 1 if not distributed
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_world_size()
    except ImportError:
        pass
    return 1


def get_rank() -> int:
    """
    Get current process rank.
    
    Returns:
        Rank or 0 if not distributed
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except ImportError:
        pass
    return 0

