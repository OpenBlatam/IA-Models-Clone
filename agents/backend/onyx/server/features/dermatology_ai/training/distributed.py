"""
Distributed Training Support
Implements DataParallel and DistributedDataParallel for multi-GPU training
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from typing import Optional, List
import logging
import os

logger = logging.getLogger(__name__)


def setup_distributed(
    rank: int = -1,
    world_size: int = -1,
    backend: str = "nccl"
) -> tuple:
    """
    Setup distributed training environment
    
    Args:
        rank: Process rank (-1 for single process)
        world_size: Total number of processes
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
        
    Returns:
        Tuple of (rank, world_size, device)
    """
    if rank == -1:
        # Single process mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device
    
    # Initialize process group
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    
    logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}")
    
    return rank, world_size, device


def wrap_model_for_distributed(
    model: nn.Module,
    device: torch.device,
    use_ddp: bool = True,
    find_unused_parameters: bool = False
) -> nn.Module:
    """
    Wrap model for distributed training
    
    Args:
        model: PyTorch model
        device: Device to use
        use_ddp: Use DistributedDataParallel (True) or DataParallel (False)
        find_unused_parameters: For DDP, find unused parameters
        
    Returns:
        Wrapped model
    """
    model = model.to(device)
    
    if use_ddp and torch.distributed.is_initialized():
        # Use DistributedDataParallel for multi-node/multi-GPU
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == 'cuda' else None,
            find_unused_parameters=find_unused_parameters
        )
        logger.info("Model wrapped with DistributedDataParallel")
    elif torch.cuda.device_count() > 1:
        # Use DataParallel for single-node multi-GPU
        model = DataParallel(model)
        logger.info(f"Model wrapped with DataParallel on {torch.cuda.device_count()} GPUs")
    
    return model


def get_world_size() -> int:
    """Get world size (number of processes)"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def get_rank() -> int:
    """Get process rank"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process"""
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce tensor across all processes
    
    Args:
        tensor: Tensor to reduce
        average: Whether to average (True) or sum (False)
        
    Returns:
        Reduced tensor
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    
    if average:
        rt /= get_world_size()
    
    return rt


def synchronize():
    """Synchronize all processes"""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


class DistributedSampler:
    """
    Distributed sampler wrapper
    Ensures each process gets a different subset of data
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if not torch.distributed.is_initialized():
            self.num_replicas = 1
            self.rank = 0
        else:
            self.num_replicas = num_replicas or torch.distributed.get_world_size()
            self.rank = rank or torch.distributed.get_rank()
        
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch = 0
    
    def __iter__(self):
        if self.shuffle:
            # Deterministic shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Subsample for this rank
        indices = indices[self.rank::self.num_replicas]
        
        return iter(indices)
    
    def __len__(self):
        return (len(self.dataset) + self.num_replicas - 1) // self.num_replicas
    
    def set_epoch(self, epoch):
        """Set epoch for shuffling"""
        self.epoch = epoch













