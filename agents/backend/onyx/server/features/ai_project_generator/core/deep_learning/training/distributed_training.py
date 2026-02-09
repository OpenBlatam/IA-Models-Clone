"""
Distributed Training - Multi-GPU and Distributed Training Support
=================================================================

Implements distributed training using:
- DataParallel (single node, multiple GPUs)
- DistributedDataParallel (multiple nodes, multiple GPUs)
- Proper initialization and synchronization
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist
import os

logger = logging.getLogger(__name__)


def setup_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: str = 'nccl'
) -> Dict[str, Any]:
    """
    Setup distributed training environment.
    
    Args:
        rank: Process rank (auto-detected if None)
        world_size: Total number of processes (auto-detected if None)
        backend: Backend ('nccl' for GPU, 'gloo' for CPU)
        
    Returns:
        Dictionary with rank, world_size, device
    """
    # Auto-detect from environment variables
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size
            )
        
        # Set device
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        
        logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Single GPU/CPU training")
    
    return {
        'rank': rank,
        'world_size': world_size,
        'device': device,
        'is_distributed': world_size > 1
    }


def wrap_model_for_distributed(
    model: nn.Module,
    use_ddp: bool = True,
    find_unused_parameters: bool = False
) -> nn.Module:
    """
    Wrap model for distributed training.
    
    Args:
        model: PyTorch model
        use_ddp: Use DistributedDataParallel (True) or DataParallel (False)
        find_unused_parameters: Find unused parameters for DDP
        
    Returns:
        Wrapped model
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping distributed wrapping")
        return model
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        logger.info("Single GPU, no wrapping needed")
        return model
    
    if use_ddp and dist.is_initialized():
        # Use DistributedDataParallel
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=find_unused_parameters
        )
        logger.info("Model wrapped with DistributedDataParallel")
    else:
        # Use DataParallel (single node, multiple GPUs)
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel ({num_gpus} GPUs)")
    
    return model


def cleanup_distributed() -> None:
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


class DistributedSamplerWrapper:
    """Wrapper for distributed data sampling."""
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True
    ):
        """
        Initialize distributed sampler.
        
        Args:
            dataset: Dataset to sample from
            num_replicas: Number of replicas
            rank: Current rank
            shuffle: Whether to shuffle
        """
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle
        )
    
    def __iter__(self):
        """Iterate over samples."""
        return iter(self.sampler)
    
    def __len__(self):
        """Get length."""
        return len(self.sampler)
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""
        self.sampler.set_epoch(epoch)



