"""
Distributed Training Service - Entrenamiento distribuido
=========================================================

Sistema para entrenamiento distribuido multi-GPU y multi-node.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DistributedStrategy(str, Enum):
    """Estrategias de entrenamiento distribuido"""
    DATA_PARALLEL = "data_parallel"  # Single node, multi-GPU
    DISTRIBUTED_DATA_PARALLEL = "ddp"  # Multi-node, multi-GPU
    MODEL_PARALLEL = "model_parallel"  # Split model across GPUs
    PIPELINE_PARALLEL = "pipeline_parallel"  # Pipeline parallelism


@dataclass
class DistributedConfig:
    """Configuración de entrenamiento distribuido"""
    strategy: DistributedStrategy
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    backend: str = "nccl"  # nccl, gloo
    master_addr: str = "localhost"
    master_port: int = 29500
    world_size: Optional[int] = None
    rank: Optional[int] = None


class DistributedTrainingService:
    """Servicio de entrenamiento distribuido"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.distributed_jobs: Dict[str, Any] = {}
        logger.info("DistributedTrainingService initialized")
    
    def setup_distributed(
        self,
        job_id: str,
        config: DistributedConfig
    ) -> bool:
        """Configurar entorno distribuido"""
        if not TORCH_AVAILABLE:
            return False
        
        try:
            if config.strategy == DistributedStrategy.DISTRIBUTED_DATA_PARALLEL:
                # Initialize process group
                if config.world_size is not None and config.rank is not None:
                    dist.init_process_group(
                        backend=config.backend,
                        init_method=f"tcp://{config.master_addr}:{config.master_port}",
                        world_size=config.world_size,
                        rank=config.rank,
                    )
                    logger.info(f"Distributed training initialized: rank={config.rank}, world_size={config.world_size}")
            
            self.distributed_jobs[job_id] = {
                "config": config,
                "initialized": True,
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up distributed training: {e}")
            return False
    
    def wrap_model_for_distributed(
        self,
        model: nn.Module,
        config: DistributedConfig,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True
    ) -> nn.Module:
        """
        Envolver modelo para entrenamiento distribuido.
        
        Args:
            model: Modelo a envolver
            config: Configuración distribuida
            find_unused_parameters: Para DDP, encontrar parámetros no usados
            gradient_as_bucket_view: Optimización de memoria para DDP
        
        Returns:
            Modelo envuelto
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            if config.strategy == DistributedStrategy.DATA_PARALLEL:
                if torch.cuda.device_count() > 1:
                    # DataParallel para single node, multi-GPU
                    model = nn.DataParallel(model)
                    logger.info(
                        f"Model wrapped with DataParallel on {torch.cuda.device_count()} GPUs"
                    )
                else:
                    logger.warning("DataParallel requested but only 1 GPU available")
            
            elif config.strategy == DistributedStrategy.DISTRIBUTED_DATA_PARALLEL:
                if not dist.is_initialized():
                    logger.error("Distributed not initialized. Call setup_distributed first.")
                    return model
                
                # DistributedDataParallel para multi-node, multi-GPU
                device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
                
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[device_id] if device_id is not None else None,
                    output_device=device_id,
                    find_unused_parameters=find_unused_parameters,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
                logger.info(
                    f"Model wrapped with DistributedDataParallel "
                    f"(rank={dist.get_rank()}, world_size={dist.get_world_size()})"
                )
            
            return model
            
        except Exception as e:
            logger.error(f"Error wrapping model for distributed training: {e}", exc_info=True)
            return model
    
    def get_world_info(self) -> Dict[str, Any]:
        """Obtener información del mundo distribuido"""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if dist.is_initialized():
            info.update({
                "distributed": True,
                "world_size": dist.get_world_size(),
                "rank": dist.get_rank(),
                "backend": dist.get_backend(),
            })
        else:
            info["distributed"] = False
        
        return info

