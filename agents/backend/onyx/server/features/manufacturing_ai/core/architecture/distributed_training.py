"""
Distributed Training Support
============================

Soporte para entrenamiento distribuido y multi-GPU.
"""

import logging
from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    dist = None
    DataParallel = None
    DistributedDataParallel = None

logger = logging.getLogger(__name__)


class DistributedTrainingManager:
    """
    Gestor de entrenamiento distribuido.
    
    Maneja DataParallel y DistributedDataParallel.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        self.num_gpus = torch.cuda.device_count() if TORCH_AVAILABLE and torch.cuda.is_available() else 0
    
    def wrap_model(
        self,
        model: nn.Module,
        use_distributed: bool = False,
        device_ids: Optional[list] = None
    ) -> nn.Module:
        """
        Envolver modelo para entrenamiento distribuido.
        
        Args:
            model: Modelo a envolver
            use_distributed: Usar DistributedDataParallel (True) o DataParallel (False)
            device_ids: IDs de dispositivos (opcional)
            
        Returns:
            Modelo envuelto
        """
        if not TORCH_AVAILABLE:
            return model
        
        if self.num_gpus == 0:
            logger.warning("No GPUs available, using CPU")
            return model.to(self.device)
        
        model = model.to(self.device)
        
        if use_distributed:
            # DistributedDataParallel
            if dist.is_available() and dist.is_initialized():
                model = DistributedDataParallel(
                    model,
                    device_ids=device_ids or [torch.cuda.current_device()],
                    find_unused_parameters=False
                )
                logger.info("Wrapped model with DistributedDataParallel")
            else:
                logger.warning("Distributed not initialized, using DataParallel")
                use_distributed = False
        
        if not use_distributed:
            # DataParallel
            if device_ids is None:
                device_ids = list(range(self.num_gpus))
            
            model = DataParallel(model, device_ids=device_ids)
            logger.info(f"Wrapped model with DataParallel on {len(device_ids)} GPUs")
        
        return model
    
    def get_batch_size_per_gpu(self, total_batch_size: int) -> int:
        """
        Calcular batch size por GPU.
        
        Args:
            total_batch_size: Batch size total
            
        Returns:
            Batch size por GPU
        """
        if self.num_gpus == 0:
            return total_batch_size
        
        return total_batch_size // self.num_gpus
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "num_gpus": self.num_gpus,
            "device": str(self.device) if self.device else "N/A",
            "distributed_available": dist.is_available() if TORCH_AVAILABLE else False
        }


# Instancia global
_distributed_training_manager = None


def get_distributed_training_manager() -> DistributedTrainingManager:
    """Obtener instancia global."""
    global _distributed_training_manager
    if _distributed_training_manager is None:
        _distributed_training_manager = DistributedTrainingManager()
    return _distributed_training_manager

