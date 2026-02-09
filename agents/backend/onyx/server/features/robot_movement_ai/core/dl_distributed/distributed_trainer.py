"""
Distributed Training - Modular Distributed Training
==================================================

Entrenamiento distribuido modular usando DDP y Horovod.
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Trainer distribuido modular.
    
    Soporta entrenamiento distribuido con DDP
    y múltiples GPUs/nodos.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        backend: str = 'nccl',
        find_unused_parameters: bool = False
    ):
        """
        Inicializar trainer distribuido.
        
        Args:
            model: Modelo a entrenar
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            backend: Backend distribuido ('nccl', 'gloo')
            find_unused_parameters: Encontrar parámetros no usados
        """
        if not torch.distributed.is_available():
            raise RuntimeError("Distributed training not available")
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.backend = backend
        self.find_unused_parameters = find_unused_parameters
        
        # Inicializar proceso distribuido
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(f'cuda:{self.rank}')
        
        # Mover modelo a dispositivo
        self.model = self.model.to(self.device)
        
        # Crear DDP model
        self.model = DDP(
            self.model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=find_unused_parameters
        )
        
        logger.info(f"Distributed trainer initialized: rank={self.rank}, world_size={self.world_size}")
    
    def setup_distributed_sampler(self, dataset, shuffle: bool = True):
        """
        Configurar sampler distribuido.
        
        Args:
            dataset: Dataset
            shuffle: Mezclar datos
            
        Returns:
            DistributedSampler
        """
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
    
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Reducir métricas entre procesos.
        
        Args:
            metrics: Métricas locales
            
        Returns:
            Métricas promediadas
        """
        import torch.distributed as dist
        
        averaged_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            averaged_metrics[key] = (tensor / self.world_size).item()
        
        return averaged_metrics
    
    def save_checkpoint(self, path: str, **kwargs):
        """
        Guardar checkpoint (solo en rank 0).
        
        Args:
            path: Ruta del checkpoint
            **kwargs: Datos adicionales
        """
        if self.rank == 0:
            checkpoint = {
                'model_state_dict': self.model.module.state_dict(),
                **kwargs
            }
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")


def setup_distributed(
    backend: str = 'nccl',
    init_method: Optional[str] = None
):
    """
    Configurar entorno distribuido.
    
    Args:
        backend: Backend ('nccl', 'gloo')
        init_method: Método de inicialización
    """
    if not torch.distributed.is_available():
        raise RuntimeError("Distributed training not available")
    
    if torch.distributed.is_initialized():
        return
    
    if init_method is None:
        # Usar variables de entorno
        init_method = 'env://'
    
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method
    )
    
    logger.info("Distributed environment initialized")


def cleanup_distributed():
    """Limpiar entorno distribuido."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Distributed environment cleaned up")








