"""
Distributed Training - Entrenamiento distribuido con DDP
=========================================================
"""

import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuración de entrenamiento distribuido"""
    backend: str = "nccl"  # nccl para GPU, gloo para CPU
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0


class DistributedTrainer:
    """Entrenador distribuido con DDP"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_distributed = config.world_size > 1
        self.device = None
        self._setup_distributed()
    
    def _setup_distributed(self):
        """Configura el entorno distribuido"""
        if not self.is_distributed:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return
        
        # Inicializar proceso distribuido
        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        # Configurar device local
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config.local_rank}")
            torch.cuda.set_device(self.config.local_rank)
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Proceso distribuido inicializado: rank={self.config.rank}, device={self.device}")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Envuelve el modelo con DDP"""
        if not self.is_distributed:
            return model.to(self.device)
        
        model = model.to(self.device)
        model = DDP(model, device_ids=[self.config.local_rank])
        logger.info("Modelo envuelto con DDP")
        return model
    
    def get_distributed_sampler(self, dataset, shuffle: bool = True):
        """Obtiene un sampler distribuido"""
        if not self.is_distributed:
            return None
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle
        )
        return sampler
    
    def cleanup(self):
        """Limpia el proceso distribuido"""
        if self.is_distributed:
            dist.destroy_process_group()
            logger.info("Proceso distribuido finalizado")




