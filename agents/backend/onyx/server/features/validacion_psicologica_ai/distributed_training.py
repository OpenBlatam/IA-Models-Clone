"""
Entrenamiento Distribuido
==========================
Multi-GPU y distributed training
"""

from typing import Dict, Any, List, Optional
import structlog
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import os

logger = structlog.get_logger()


class DistributedTrainer:
    """Trainer para entrenamiento distribuido"""
    
    def __init__(
        self,
        model: nn.Module,
        use_ddp: bool = True,
        backend: str = "nccl"
    ):
        """
        Inicializar trainer distribuido
        
        Args:
            model: Modelo a entrenar
            use_ddp: Usar DistributedDataParallel
            backend: Backend para comunicación (nccl, gloo)
        """
        self.model = model
        self.use_ddp = use_ddp
        self.backend = backend
        self.device = None
        self.world_size = 1
        self.rank = 0
        
        self._setup_distributed()
    
    def _setup_distributed(self) -> None:
        """Configurar entorno distribuido"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = torch.device("cpu")
            return
        
        # Verificar si estamos en entorno distribuido
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            
            if self.use_ddp:
                torch.distributed.init_process_group(
                    backend=self.backend,
                    init_method="env://"
                )
                
                self.device = torch.device(f"cuda:{self.rank}")
                self.model = self.model.to(self.device)
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.rank]
                )
                
                logger.info(
                    "Distributed training initialized",
                    rank=self.rank,
                    world_size=self.world_size
                )
            else:
                # DataParallel (más simple, menos eficiente)
                self.device = torch.device("cuda:0")
                self.model = self.model.to(self.device)
                self.model = DataParallel(self.model)
                
                logger.info("DataParallel initialized")
        else:
            # Single GPU o CPU
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            logger.info("Single device training", device=str(self.device))
    
    def get_distributed_sampler(
        self,
        dataset,
        shuffle: bool = True
    ) -> Optional[DistributedSampler]:
        """
        Obtener sampler distribuido
        
        Args:
            dataset: Dataset
            shuffle: Mezclar datos
            
        Returns:
            Sampler distribuido o None
        """
        if self.world_size > 1:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
        return None
    
    def cleanup(self) -> None:
        """Limpiar recursos distribuidos"""
        if self.world_size > 1 and self.use_ddp:
            torch.distributed.destroy_process_group()
            logger.info("Distributed training cleaned up")


class GradientAccumulator:
    """Acumulador de gradientes para batches grandes"""
    
    def __init__(self, accumulation_steps: int = 4):
        """
        Inicializar acumulador
        
        Args:
            accumulation_steps: Pasos de acumulación
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_update(self) -> bool:
        """Verificar si se debe actualizar"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self) -> None:
        """Resetear contador"""
        self.current_step = 0


class MixedPrecisionTrainer:
    """Trainer con mixed precision"""
    
    def __init__(self, model: nn.Module, optimizer, scaler=None):
        """
        Inicializar trainer con mixed precision
        
        Args:
            model: Modelo
            optimizer: Optimizador
            scaler: GradScaler (opcional)
        """
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler or torch.cuda.amp.GradScaler()
        self.use_amp = torch.cuda.is_available()
    
    def train_step(self, loss_fn, inputs, targets):
        """
        Paso de entrenamiento con mixed precision
        
        Args:
            loss_fn: Función de pérdida
            inputs: Entradas
            targets: Objetivos
            
        Returns:
            Pérdida
        """
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()


# Instancias globales
distributed_trainer = None  # Se inicializa cuando se necesita




