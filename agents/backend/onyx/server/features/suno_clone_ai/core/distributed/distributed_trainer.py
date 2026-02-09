"""
Distributed Training

Utilities for multi-GPU and distributed training.
"""

import logging
import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None
) -> Dict[str, Any]:
    """
    Setup distributed training.
    
    Args:
        backend: Distributed backend ('nccl', 'gloo', 'mpi')
        init_method: Initialization method
        
    Returns:
        Distributed configuration
    """
    # Get environment variables
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        # Initialize process group
        if init_method is None:
            init_method = f"tcp://{os.environ.get('MASTER_ADDR', 'localhost')}:{os.environ.get('MASTER_PORT', '29500')}"
        
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        logger.info(
            f"Distributed training initialized: "
            f"rank={rank}, world_size={world_size}, device={device}"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Single GPU training, distributed not initialized")
    
    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'device': device,
        'is_distributed': world_size > 1
    }


def get_distributed_config() -> Dict[str, Any]:
    """
    Get current distributed configuration.
    
    Returns:
        Distributed configuration
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'is_distributed': world_size > 1,
        'is_main_process': rank == 0
    }


class DistributedTrainer:
    """Trainer for distributed training."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        find_unused_parameters: bool = False
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Model to train
            device: Device to use
            find_unused_parameters: Find unused parameters for DDP
        """
        self.config = get_distributed_config()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.config['is_distributed']:
            # Move model to device
            model = model.to(self.device)
            
            # Wrap with DDP
            self.model = DDP(
                model,
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank'],
                find_unused_parameters=find_unused_parameters
            )
            
            logger.info("Model wrapped with DistributedDataParallel")
        else:
            self.model = model.to(self.device)
            logger.info("Single GPU training, DDP not used")
    
    def get_model(self) -> nn.Module:
        """Get model (unwrap DDP if needed)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.config['is_main_process']
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.config['is_distributed']:
            dist.barrier()



