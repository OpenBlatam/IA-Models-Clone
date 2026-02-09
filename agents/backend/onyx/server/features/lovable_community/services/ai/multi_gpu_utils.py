"""
Multi-GPU Training Utilities

Provides utilities for multi-GPU training:
- DataParallel
- DistributedDataParallel
- Gradient synchronization
"""

import logging
import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class MultiGPUTrainer:
    """
    Multi-GPU training utilities
    
    Supports both DataParallel and DistributedDataParallel.
    """
    
    @staticmethod
    def setup_data_parallel(
        model: nn.Module,
        device_ids: Optional[list] = None,
        output_device: Optional[int] = None
    ) -> nn.Module:
        """
        Setup DataParallel for multi-GPU training
        
        Args:
            model: Model to parallelize
            device_ids: List of GPU IDs (None for all available)
            output_device: Output device (None for device_ids[0])
            
        Returns:
            Parallelized model
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, returning original model")
            return model
        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        if len(device_ids) == 1:
            logger.info("Only one GPU available, using single GPU")
            return model.to(f"cuda:{device_ids[0]}")
        
        logger.info(f"Setting up DataParallel on GPUs: {device_ids}")
        model = DataParallel(model, device_ids=device_ids, output_device=output_device)
        
        return model
    
    @staticmethod
    def setup_distributed(
        model: nn.Module,
        local_rank: int = 0,
        find_unused_parameters: bool = False
    ) -> nn.Module:
        """
        Setup DistributedDataParallel for distributed training
        
        Args:
            model: Model to parallelize
            local_rank: Local rank of the process
            find_unused_parameters: Whether to find unused parameters
            
        Returns:
            Distributed model
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, returning original model")
            return model
        
        # Set device
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        
        # Setup DDP
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters
        )
        
        logger.info(f"Setting up DistributedDataParallel on rank {local_rank}")
        
        return model
    
    @staticmethod
    def get_distributed_sampler(
        dataset: torch.utils.data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True
    ) -> DistributedSampler:
        """
        Get distributed sampler for data loading
        
        Args:
            dataset: Dataset to sample from
            num_replicas: Number of processes
            rank: Rank of current process
            shuffle: Whether to shuffle
            
        Returns:
            Distributed sampler
        """
        if num_replicas is None:
            num_replicas = int(os.environ.get("WORLD_SIZE", 1))
        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle
        )
        
        return sampler


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None
) -> Dict[str, Any]:
    """
    Initialize distributed training
    
    Args:
        backend: Backend to use (nccl for GPU, gloo for CPU)
        init_method: Initialization method (None for env://)
        
    Returns:
        Dictionary with rank, world_size, etc.
    """
    if init_method is None:
        init_method = "env://"
    
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method
    )
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    logger.info(
        f"Initialized distributed training - "
        f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}"
    )
    
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_main_process": rank == 0
    }


def cleanup_distributed() -> None:
    """Cleanup distributed training"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Distributed training cleaned up")















