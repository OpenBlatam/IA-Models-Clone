"""
Distributed Training for Recovery Models
Multi-GPU training with DistributedDataParallel
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict
import logging
import os

logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """
    Setup distributed training
    
    Args:
        rank: Process rank
        world_size: Number of processes
        backend: Backend (nccl for GPU, gloo for CPU)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    logger.info(f"Distributed setup: rank={rank}, world_size={world_size}")


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


class DistributedTrainer:
    """Distributed trainer for multi-GPU training"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize distributed trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            rank: Process rank
            world_size: Number of processes
            device: Device to use
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device(f"cuda:{rank}")
        
        # Setup distributed if needed
        if world_size > 1:
            setup_distributed(rank, world_size)
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Wrap with DDP
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False
            )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        logger.info(f"DistributedTrainer initialized on rank {rank}")
    
    def train_epoch(self, optimizer, criterion, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Average across all processes
        if self.world_size > 1:
            dist.all_reduce(torch.tensor(total_loss), op=dist.ReduceOp.SUM)
            total_loss = total_loss / self.world_size
        
        return {"loss": total_loss / num_batches}
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.world_size > 1:
            cleanup_distributed()

