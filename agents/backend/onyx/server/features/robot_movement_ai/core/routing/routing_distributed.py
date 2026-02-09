"""
Routing Distributed Training
============================

Soporte para entrenamiento distribuido multi-GPU usando DistributedDataParallel.
"""

import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DISTRIBUTED_AVAILABLE = False
    logger.warning("PyTorch distributed not available. Distributed training will be disabled.")


class DistributedTrainer:
    """Trainer para entrenamiento distribuido."""
    
    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        backend: str = "nccl"
    ):
        """
        Inicializar trainer distribuido.
        
        Args:
            model: Modelo PyTorch
            rank: Rango del proceso actual
            world_size: Número total de procesos
            backend: Backend de comunicación (nccl para GPU, gloo para CPU)
        """
        if not DISTRIBUTED_AVAILABLE:
            raise ImportError("PyTorch distributed is required")
        
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        
        # Inicializar proceso distribuido
        self._init_process_group()
        
        # Mover modelo a dispositivo
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.device = device
        model = model.to(device)
        
        # Envolver con DDP
        self.model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    def _init_process_group(self):
        """Inicializar grupo de procesos distribuido."""
        if dist.is_initialized():
            return
        
        dist.init_process_group(
            backend=self.backend,
            init_method='env://',
            rank=self.rank,
            world_size=self.world_size
        )
        logger.info(f"Process group initialized: rank={self.rank}, world_size={self.world_size}")
    
    def cleanup(self):
        """Limpiar proceso distribuido."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info(f"Process group destroyed: rank={self.rank}")


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """
    Configurar entorno distribuido.
    
    Args:
        rank: Rango del proceso
        world_size: Número total de procesos
        backend: Backend de comunicación
    """
    if not DISTRIBUTED_AVAILABLE:
        raise ImportError("PyTorch distributed is required")
    
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_distributed():
    """Limpiar proceso distribuido."""
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()

