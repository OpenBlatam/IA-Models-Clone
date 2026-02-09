"""
Distributed Training with Multi-GPU Support
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict
import logging
import os

from .trainer import ModelTrainer

logger = logging.getLogger(__name__)


class DistributedTrainer(ModelTrainer):
    """Distributed trainer for multi-GPU training"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        rank: int = 0,
        world_size: int = 1,
        backend: str = "nccl",
        **kwargs
    ):
        """
        Initialize distributed trainer
        
        Args:
            rank: Process rank
            world_size: Number of processes
            backend: Distributed backend
        """
        # Initialize distributed process group
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Wrap with DDP
        self.model = DDP(model, device_ids=[rank], output_device=rank)
        
        # Update DataLoader with DistributedSampler
        if hasattr(train_loader, 'sampler'):
            train_loader.sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=world_size,
                rank=rank
            )
        
        # Initialize parent (without model wrapping)
        super().__init__(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            **kwargs
        )
        
        logger.info(f"DistributedTrainer initialized on rank {rank}/{world_size}")
    
    def train_epoch(self, optimizer, criterion, epoch, clip_grad_norm=None):
        """Distributed training epoch"""
        # Set epoch for sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        # Call parent training
        metrics = super().train_epoch(optimizer, criterion, epoch, clip_grad_norm)
        
        # Synchronize metrics across processes
        if dist.is_initialized():
            # Average loss across processes
            avg_loss = torch.tensor(metrics['loss']).to(self.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            metrics['loss'] = (avg_loss / self.world_size).item()
        
        return metrics
    
    def cleanup(self):
        """Cleanup distributed process group"""
        if dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def create_distributed_trainer(
    model: nn.Module,
    train_loader,
    val_loader=None,
    rank: int = 0,
    world_size: int = 1,
    **kwargs
) -> DistributedTrainer:
    """Factory function for distributed trainer"""
    return DistributedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        rank=rank,
        world_size=world_size,
        **kwargs
    )

