"""
Distributed Training
Support for multi-GPU and distributed training.
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Distributed training support with DataParallel and DistributedDataParallel.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device_ids: Optional[list] = None,
        use_ddp: bool = True,
        find_unused_parameters: bool = False,
    ):
        self.model = model
        self.device_ids = device_ids
        self.use_ddp = use_ddp
        self.find_unused_parameters = find_unused_parameters
        self.is_distributed = False
        
        # Check if distributed training is available
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = 0
        
        # Setup distributed training
        if self.num_gpus > 1:
            if self.use_ddp and self._is_distributed_available():
                self._setup_ddp()
            else:
                self._setup_data_parallel()
        else:
            logger.info("Single GPU/CPU detected, distributed training not used")
    
    def _is_distributed_available(self) -> bool:
        """Check if distributed training is available."""
        return (
            "RANK" in os.environ and
            "WORLD_SIZE" in os.environ and
            int(os.environ.get("WORLD_SIZE", 0)) > 1
        )
    
    def _setup_ddp(self):
        """Setup DistributedDataParallel."""
        if not self._is_distributed_available():
            logger.warning("DDP environment not detected, falling back to DataParallel")
            self._setup_data_parallel()
            return
        
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        self.model = self.model.to(device)
        
        # Initialize process group
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        
        # Wrap model
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=self.find_unused_parameters,
        )
        
        self.is_distributed = True
        logger.info(f"DDP initialized: rank={rank}, world_size={world_size}, device={local_rank}")
    
    def _setup_data_parallel(self):
        """Setup DataParallel for multi-GPU."""
        if self.num_gpus <= 1:
            logger.info("Single GPU, DataParallel not used")
            return
        
        if self.device_ids is None:
            self.device_ids = list(range(self.num_gpus))
        
        self.model = DataParallel(self.model, device_ids=self.device_ids)
        logger.info(f"DataParallel initialized on devices: {self.device_ids}")
    
    def get_model(self) -> nn.Module:
        """Get the wrapped model."""
        return self.model
    
    def get_sampler(self, dataset, shuffle: bool = True):
        """
        Get distributed sampler if using DDP.
        
        Args:
            dataset: Dataset to sample from
            shuffle: Whether to shuffle
            
        Returns:
            Sampler or None
        """
        if self.is_distributed:
            return DistributedSampler(
                dataset,
                num_replicas=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0)),
                shuffle=shuffle,
            )
        return None
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_distributed:
            torch.distributed.destroy_process_group()
            logger.info("Distributed process group destroyed")



