"""
Distributed Training System - Sistema de entrenamiento distribuido
===================================================================
Soporte para DataParallel y DistributedDataParallel
"""

import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Sistema de entrenamiento distribuido"""
    
    def __init__(
        self,
        use_distributed: bool = False,
        backend: str = "nccl",
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        self.use_distributed = use_distributed and torch.cuda.is_available()
        self.backend = backend
        self.world_size = world_size or int(os.getenv("WORLD_SIZE", 1))
        self.rank = rank or int(os.getenv("RANK", 0))
        self.device = None
        
        if self.use_distributed:
            self._init_distributed()
    
    def _init_distributed(self):
        """Inicializa proceso distribuido"""
        if not dist.is_available():
            logger.warning("Distributed training not available")
            self.use_distributed = False
            return
        
        try:
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank
            )
            self.device = torch.device(f"cuda:{self.rank}")
            torch.cuda.set_device(self.rank)
            logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            self.use_distributed = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Envuelve modelo para entrenamiento distribuido"""
        if not torch.cuda.is_available():
            return model
        
        if self.use_distributed:
            model = model.to(self.device)
            model = DistributedDataParallel(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False
            )
            logger.info("Model wrapped with DistributedDataParallel")
        elif torch.cuda.device_count() > 1:
            model = model.cuda()
            model = DataParallel(model)
            logger.info(f"Model wrapped with DataParallel on {torch.cuda.device_count()} GPUs")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        
        return model
    
    def get_sampler(self, dataset):
        """Obtiene sampler distribuido para dataset"""
        if self.use_distributed:
            from torch.utils.data.distributed import DistributedSampler
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        return None
    
    def cleanup(self):
        """Limpia recursos distribuidos"""
        if self.use_distributed and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed training cleaned up")




