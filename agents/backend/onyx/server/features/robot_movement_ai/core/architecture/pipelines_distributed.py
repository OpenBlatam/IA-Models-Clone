"""
Distributed Training Module
============================

Soporte profesional para entrenamiento distribuido con PyTorch.
Incluye DataParallel, DistributedDataParallel y optimizaciones avanzadas.
"""

import logging
import os
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    dist = None
    DataParallel = None
    DistributedDataParallel = None
    logging.warning("PyTorch not available. Distributed training disabled.")

logger = logging.getLogger(__name__)


class DistributedTrainingSetup:
    """
    Configuración profesional para entrenamiento distribuido.
    
    Soporta:
    - Multi-GPU con DataParallel
    - Multi-node con DistributedDataParallel
    - Optimizaciones de comunicación
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        """
        Inicializar configuración distribuida.
        
        Args:
            backend: Backend de comunicación ("nccl", "gloo", "mpi")
            init_method: Método de inicialización
            world_size: Tamaño del mundo (número de procesos)
            rank: Rango del proceso actual
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for distributed training")
        
        self.backend = backend
        self.initialized = False
        self.world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))
        self.rank = rank or int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Inicializar si no está inicializado
        if not dist.is_initialized():
            if init_method is None:
                init_method = os.environ.get('MASTER_ADDR', 'env://')
            
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    world_size=self.world_size,
                    rank=self.rank
                )
                self.initialized = True
                logger.info(f"Distributed training initialized: rank={self.rank}, world_size={self.world_size}")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed training: {e}")
                self.initialized = False
        else:
            self.initialized = True
    
    def setup_model(self, model: nn.Module, use_ddp: bool = True) -> nn.Module:
        """
        Configurar modelo para entrenamiento distribuido.
        
        Args:
            model: Modelo PyTorch
            use_ddp: Usar DistributedDataParallel (True) o DataParallel (False)
            
        Returns:
            Modelo envuelto
        """
        if not self.initialized:
            logger.warning("Distributed training not initialized, using single GPU")
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                return DataParallel(model)
            return model
        
        if use_ddp:
            # DistributedDataParallel es más eficiente
            model = model.to(self.local_rank)
            model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            logger.info("Model wrapped with DistributedDataParallel")
        else:
            # DataParallel para single-node multi-GPU
            if torch.cuda.device_count() > 1:
                model = DataParallel(model)
                logger.info(f"Model wrapped with DataParallel on {torch.cuda.device_count()} GPUs")
        
        return model
    
    def setup_dataloader(self, dataset, batch_size: int, **kwargs):
        """
        Configurar DataLoader para entrenamiento distribuido.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch por GPU
            **kwargs: Argumentos adicionales para DataLoader
            
        Returns:
            DataLoader configurado
        """
        from torch.utils.data import DataLoader
        
        if self.initialized:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=kwargs.get('shuffle', True)
            )
            kwargs['sampler'] = sampler
            kwargs['shuffle'] = False
        else:
            kwargs.setdefault('shuffle', True)
        
        return DataLoader(dataset, batch_size=batch_size, **kwargs)
    
    def cleanup(self):
        """Limpiar proceso distribuido."""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")
    
    def is_main_process(self) -> bool:
        """Verificar si es el proceso principal."""
        return self.rank == 0
    
    def synchronize(self):
        """Sincronizar todos los procesos."""
        if self.initialized:
            dist.barrier()


def setup_distributed_training(
    backend: str = "nccl",
    use_ddp: bool = True
) -> Optional[DistributedTrainingSetup]:
    """
    Helper function para configurar entrenamiento distribuido.
    
    Args:
        backend: Backend de comunicación
        use_ddp: Usar DistributedDataParallel
        
    Returns:
        DistributedTrainingSetup o None si no está disponible
    """
    if not TORCH_AVAILABLE:
        return None
    
    try:
        setup = DistributedTrainingSetup(backend=backend)
        return setup if setup.initialized else None
    except Exception as e:
        logger.warning(f"Could not setup distributed training: {e}")
        return None

