"""
Distributed Utils - Utilidades de Entrenamiento Distribuido
============================================================

Utilidades para entrenamiento distribuido multi-GPU y multi-nodo.
"""

import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Trainer para entrenamiento distribuido.
    """
    
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
            model: Modelo
            rank: Rango del proceso
            world_size: Tamaño del mundo
            backend: Backend de comunicación
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        
        # Inicializar proceso grupo
        self._init_process_group()
        
        # Mover modelo a dispositivo
        device = torch.device(f"cuda:{rank}")
        self.model = model.to(device)
        
        # Envolver con DDP
        self.model = DDP(self.model, device_ids=[rank])
    
    def _init_process_group(self):
        """Inicializar proceso grupo."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                rank=self.rank,
                world_size=self.world_size
            )
    
    def cleanup(self):
        """Limpiar proceso grupo."""
        if dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl"
):
    """
    Configurar entorno distribuido.
    
    Args:
        rank: Rango del proceso
        world_size: Tamaño del mundo
        backend: Backend
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )


def cleanup_distributed():
    """Limpiar entorno distribuido."""
    if dist.is_initialized():
        dist.destroy_process_group()


class GradientSynchronizer:
    """
    Sincronizador de gradientes.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar sincronizador.
        
        Args:
            model: Modelo
        """
        self.model = model
    
    def synchronize_gradients(self):
        """Sincronizar gradientes entre procesos."""
        if isinstance(self.model, DDP):
            # DDP sincroniza automáticamente
            pass
        else:
            # Sincronización manual
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= dist.get_world_size()


class DistributedDataLoader:
    """
    DataLoader distribuido.
    """
    
    @staticmethod
    def create(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        rank: int,
        world_size: int,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """
        Crear DataLoader distribuido.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch
            rank: Rango del proceso
            world_size: Tamaño del mundo
            **kwargs: Argumentos adicionales
            
        Returns:
            DataLoader distribuido
        """
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            **kwargs
        )


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce y promedio de tensor.
    
    Args:
        tensor: Tensor
        
    Returns:
        Tensor promediado
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor desde src.
    
    Args:
        tensor: Tensor
        src: Rango fuente
        
    Returns:
        Tensor broadcasted
    """
    if dist.is_initialized():
        dist.broadcast(tensor, src)
    return tensor




