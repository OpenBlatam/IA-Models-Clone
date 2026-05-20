"""
Distributed Trainer - Entrenador Distribuido
=============================================

Entrenamiento distribuido multi-GPU con DDP.
"""

import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Entrenador distribuido para multi-GPU"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None
    ):
        """
        Inicializar entrenador distribuido
        
        Args:
            model: Modelo a entrenar
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            rank: Rango del proceso (auto-detecta si es None)
            world_size: Tamaño del mundo (auto-detecta si es None)
        """
        self.rank = rank or int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        
        # Inicializar proceso distribuido
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=self.rank,
                world_size=self.world_size
            )
        
        # Mover modelo a GPU
        self.device = torch.device(f"cuda:{self.rank}")
        model = model.to(self.device)
        
        # Envolver con DDP
        self.model = DDP(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False
        )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        logger.info(f"Distributed Trainer inicializado en rank {self.rank}/{self.world_size}")
    
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Entrenar una época distribuida
        
        Args:
            optimizer: Optimizador
            criterion: Función de pérdida
            scheduler: Scheduler de learning rate
            use_amp: Usar mixed precision
            gradient_accumulation_steps: Pasos de acumulación
            
        Returns:
            Dict con métricas
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Mover batch a dispositivo
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast() if use_amp else torch.no_grad():
                outputs = self.model(**batch)
                loss = criterion(outputs.logits, batch["labels"])
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if use_amp and scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Actualizar pesos
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp and scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
        
        # Promediar métricas entre procesos
        avg_loss = total_loss / num_batches
        tensor_loss = torch.tensor(avg_loss).to(self.device)
        dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
        avg_loss = (tensor_loss / self.world_size).item()
        
        return {"train_loss": avg_loss}
    
    def cleanup(self):
        """Limpiar proceso distribuido"""
        if dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed(rank: int, world_size: int, master_port: str = "12355"):
    """
    Configurar entorno distribuido
    
    Args:
        rank: Rango del proceso
        world_size: Tamaño del mundo
        master_port: Puerto maestro
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size
    )




