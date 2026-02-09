"""
Distributed Training Strategy
Multi-GPU training with DistributedDataParallel
"""

from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DISTRIBUTED_AVAILABLE = False
    logger.warning("PyTorch or distributed training not available")

from .mixed_precision_strategy import MixedPrecisionStrategy


class DistributedTrainingStrategy(MixedPrecisionStrategy):
    """
    Training strategy for distributed multi-GPU training
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        find_unused_parameters: bool = False
    ):
        if not DISTRIBUTED_AVAILABLE:
            raise ImportError("Distributed training not available")
        
        # Initialize distributed process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        # Get local rank
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = f"cuda:{self.local_rank}"
        
        super().__init__(
            model, optimizer, loss_fn, device,
            gradient_accumulation_steps, max_grad_norm
        )
        
        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=find_unused_parameters
        )
        
        logger.info(f"Initialized distributed training on rank {self.local_rank}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch with distributed training"""
        # Set epoch for distributed sampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        # Call parent training
        metrics = super().train_epoch(dataloader, epoch)
        
        # Synchronize metrics across processes
        if dist.is_initialized():
            # Average metrics across all processes
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tensor = torch.tensor(value, device=self.device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    metrics[key] = (tensor.item() / dist.get_world_size())
        
        return metrics

