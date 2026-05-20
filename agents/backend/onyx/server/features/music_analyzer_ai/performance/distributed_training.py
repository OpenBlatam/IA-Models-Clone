"""
Distributed Training Support
Multi-GPU training with DataParallel and DistributedDataParallel
"""

from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DistributedTrainer:
    """
    Distributed training wrapper for multi-GPU training
    Supports both DataParallel and DistributedDataParallel
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_ddp: bool = True,
        device_ids: Optional[list] = None,
        output_device: Optional[int] = None,
        find_unused_parameters: bool = False
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for distributed training")
        
        self.model = model
        self.use_ddp = use_ddp
        self.device_ids = device_ids
        self.output_device = output_device
        self.find_unused_parameters = find_unused_parameters
        
        # Detect available GPUs
        self.num_gpus = torch.cuda.device_count()
        
        if self.num_gpus == 0:
            logger.warning("No GPUs available, using CPU")
            self.device = torch.device("cpu")
            self.parallel_model = model
        elif self.num_gpus == 1:
            logger.info("Single GPU available")
            self.device = torch.device("cuda:0")
            self.parallel_model = model.to(self.device)
        else:
            logger.info(f"{self.num_gpus} GPUs available")
            self.device = torch.device("cuda:0")
            self.parallel_model = self._setup_parallel_model()
    
    def _setup_parallel_model(self) -> nn.Module:
        """Setup parallel model based on configuration"""
        if self.use_ddp and self.num_gpus > 1:
            # DistributedDataParallel (recommended for multi-node)
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                
                torch.distributed.init_process_group(
                    backend="nccl",
                    rank=rank,
                    world_size=world_size
                )
                
                model = self.model.to(self.device)
                model = DistributedDataParallel(
                    model,
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=self.find_unused_parameters
                )
                
                logger.info(f"Initialized DDP on rank {rank}/{world_size}")
                return model
            else:
                # Fallback to DataParallel for single-node multi-GPU
                logger.info("Using DataParallel for single-node multi-GPU")
                return self._setup_data_parallel()
        else:
            # DataParallel for single-node multi-GPU
            return self._setup_data_parallel()
    
    def _setup_data_parallel(self) -> nn.Module:
        """Setup DataParallel for single-node multi-GPU"""
        model = self.model.to(self.device)
        
        if self.device_ids is None:
            self.device_ids = list(range(self.num_gpus))
        
        if self.output_device is None:
            self.output_device = self.device_ids[0]
        
        model = DataParallel(
            model,
            device_ids=self.device_ids,
            output_device=self.output_device
        )
        
        logger.info(f"Using DataParallel on devices {self.device_ids}")
        return model
    
    def get_model(self) -> nn.Module:
        """Get the parallel model"""
        return self.parallel_model
    
    def get_device(self) -> torch.device:
        """Get the primary device"""
        return self.device
    
    def get_sampler(self, dataset, shuffle: bool = True):
        """Get DistributedSampler if using DDP"""
        if isinstance(self.parallel_model, DistributedDataParallel):
            return DistributedSampler(
                dataset,
                num_replicas=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0)),
                shuffle=shuffle
            )
        return None


def setup_distributed_training(
    model: nn.Module,
    use_ddp: bool = True,
    device_ids: Optional[list] = None
) -> DistributedTrainer:
    """Setup distributed training"""
    return DistributedTrainer(model, use_ddp=use_ddp, device_ids=device_ids)

