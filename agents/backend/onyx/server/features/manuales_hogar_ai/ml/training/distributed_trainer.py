"""
Distributed Trainer
===================

Entrenamiento distribuido multi-GPU.
"""

import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict, Any
from .trainer import ManualTrainer

logger = logging.getLogger(__name__)


class DistributedTrainer(ManualTrainer):
    """Trainer con soporte multi-GPU distribuido."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        use_lora: bool = True,
        device: Optional[str] = None,
        use_wandb: bool = False,
        project_name: str = "manuales-hogar-ai",
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        """
        Inicializar trainer distribuido.
        
        Args:
            model_name: Nombre del modelo base
            use_lora: Usar LoRA
            device: Dispositivo
            use_wandb: Usar Weights & Biases
            project_name: Nombre del proyecto
            world_size: Número de procesos
            rank: Rango del proceso actual
        """
        # Inicializar proceso distribuido
        if world_size is not None and rank is not None:
            self._init_distributed(rank, world_size)
        
        super().__init__(
            model_name=model_name,
            use_lora=use_lora,
            device=device,
            use_wandb=use_wandb,
            project_name=project_name
        )
        
        # Envolver modelo con DDP
        if dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[rank] if rank is not None else None,
                find_unused_parameters=False
            )
            logger.info(f"Modelo envuelto con DDP (rank {rank})")
    
    def _init_distributed(self, rank: int, world_size: int):
        """Inicializar proceso distribuido."""
        try:
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                rank=rank,
                world_size=world_size
            )
            logger.info(f"Proceso distribuido inicializado: rank {rank}/{world_size}")
        except Exception as e:
            logger.warning(f"No se pudo inicializar proceso distribuido: {str(e)}")
    
    def train(
        self,
        train_dataset,
        val_dataset=None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        output_dir: str = "./models/finetuned",
        **kwargs
    ):
        """
        Entrenar con soporte distribuido.
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            num_epochs: Número de épocas
            batch_size: Tamaño de batch (por GPU)
            learning_rate: Learning rate
            gradient_accumulation_steps: Pasos de acumulación
            output_dir: Directorio de salida
            **kwargs: Otros parámetros
        """
        # Ajustar batch size para múltiples GPUs
        if dist.is_initialized():
            world_size = dist.get_world_size()
            effective_batch_size = batch_size * world_size
            logger.info(f"Batch size efectivo: {effective_batch_size} ({batch_size} x {world_size} GPUs)")
        
        # Usar DistributedSampler
        if dist.is_initialized():
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True
            )
            kwargs['train_sampler'] = train_sampler
        
        # Llamar al método padre
        super().train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            output_dir=output_dir,
            **kwargs
        )
        
        # Limpiar proceso distribuido
        if dist.is_initialized():
            dist.destroy_process_group()




