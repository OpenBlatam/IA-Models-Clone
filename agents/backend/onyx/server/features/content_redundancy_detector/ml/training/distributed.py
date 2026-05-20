"""
Distributed Training Support
Multi-GPU and distributed training utilities
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class DistributedTrainingManager:
    """
    Manages distributed training setup and utilities
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
    ):
        """
        Initialize distributed training manager
        
        Args:
            backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
            init_method: Initialization method
        """
        self.backend = backend
        self.init_method = init_method
        self.initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
    
    def initialize(self) -> None:
        """Initialize distributed training"""
        if self.initialized:
            return
        
        # Check if distributed training is enabled
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Set device
            torch.cuda.set_device(self.local_rank)
            
            # Initialize process group
            if self.init_method is None:
                self.init_method = f"tcp://{os.environ.get('MASTER_ADDR', 'localhost')}:{os.environ.get('MASTER_PORT', '12355')}"
            
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                world_size=self.world_size,
                rank=self.rank
            )
            
            self.initialized = True
            logger.info(f"Initialized distributed training: rank={self.rank}, world_size={self.world_size}")
        else:
            logger.info("Distributed training not enabled, using single GPU/CPU")
    
    def wrap_model(
        self,
        model: nn.Module,
        device: torch.device,
        use_ddp: bool = True,
    ) -> nn.Module:
        """
        Wrap model for distributed training
        
        Args:
            model: Model to wrap
            device: Target device
            use_ddp: Use DistributedDataParallel (True) or DataParallel (False)
            
        Returns:
            Wrapped model
        """
        if not self.initialized:
            self.initialize()
        
        if self.world_size > 1 and use_ddp:
            model = model.to(device)
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            logger.info(f"Wrapped model with DDP on rank {self.rank}")
        elif torch.cuda.device_count() > 1 and not use_ddp:
            model = model.to(device)
            model = DP(model)
            logger.info(f"Wrapped model with DataParallel on {torch.cuda.device_count()} GPUs")
        else:
            model = model.to(device)
        
        return model
    
    def get_rank(self) -> int:
        """Get current process rank"""
        return self.rank
    
    def get_world_size(self) -> int:
        """Get world size"""
        return self.world_size
    
    def get_local_rank(self) -> int:
        """Get local rank"""
        return self.local_rank
    
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.rank == 0
    
    def barrier(self) -> None:
        """Synchronize all processes"""
        if self.initialized:
            dist.barrier()
    
    def cleanup(self) -> None:
        """Cleanup distributed training"""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False
            logger.info("Cleaned up distributed training")


class GradientAccumulator:
    """
    Gradient accumulation for large batch sizes
    """
    
    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulator
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_update(self) -> bool:
        """Check if gradients should be updated"""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def step(self) -> None:
        """Increment step counter"""
        self.current_step += 1
    
    def reset(self) -> None:
        """Reset step counter"""
        self.current_step = 0
    
    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any] = None,
    ) -> None:
        """
        Perform backward step with gradient accumulation
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            scaler: GradScaler for mixed precision (optional)
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Update if accumulation complete
        if self.should_update():
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        self.step()

