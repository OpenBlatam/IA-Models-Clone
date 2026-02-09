"""
Distributed training avanzado con DataParallel y DistributedDataParallel

Mejoras:
- Gradient synchronization optimizado
- Mixed precision support
- Gradient accumulation
- Better error handling
- Performance monitoring
"""

import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Trainer para entrenamiento distribuido"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_ddp: bool = False,
        ddp_backend: str = "nccl"
    ):
        self.device = device
        self.use_ddp = use_ddp
        self.model = model
        
        # Setup distributed
        if use_ddp:
            self._setup_ddp(ddp_backend)
            self.model = DistributedDataParallel(
                model.to(device),
                device_ids=[int(device.split(":")[-1])] if ":" in device else [0]
            )
        elif torch.cuda.device_count() > 1:
            logger.info(f"Usando DataParallel con {torch.cuda.device_count()} GPUs")
            self.model = DataParallel(model.to(device))
        else:
            self.model = model.to(device)
    
    def _setup_ddp(self, backend: str = "nccl"):
        """
        Configura DistributedDataParallel
        
        Args:
            backend: Backend para DDP (nccl para GPU, gloo para CPU)
        """
        if not dist.is_available():
            raise RuntimeError("Distributed training no disponible")
        
        if not dist.is_initialized():
            # Intentar inicializar desde variables de entorno
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            
            if world_size > 1:
                # Configurar master address y port
                master_addr = os.environ.get("MASTER_ADDR", "localhost")
                master_port = os.environ.get("MASTER_PORT", "12355")
                
                os.environ["MASTER_ADDR"] = master_addr
                os.environ["MASTER_PORT"] = master_port
                
                dist.init_process_group(
                    backend=backend,
                    rank=rank,
                    world_size=world_size
                )
                
                # Set device
                torch.cuda.set_device(local_rank)
                self.device = f"cuda:{local_rank}"
                
                logger.info(
                    f"DDP inicializado: rank={rank}, world_size={world_size}, "
                    f"local_rank={local_rank}"
                )
            else:
                logger.warning("WORLD_SIZE=1, DDP no inicializado")
    
    def train_step(
        self,
        inputs: Any,
        targets: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int = 0
    ) -> Dict[str, float]:
        """
        Ejecuta un paso de entrenamiento con soporte distribuido
        
        Args:
            inputs: Inputs del modelo
            targets: Targets para loss
            criterion: Función de pérdida
            optimizer: Optimizador
            step: Número de paso (para gradient accumulation)
            
        Returns:
            Diccionario con métricas
        """
        # Forward pass con mixed precision
        if self.use_mixed_precision:
            with autocast():
                outputs = self.model(inputs)
                loss = criterion(outputs, targets) / self.gradient_accumulation_steps
        else:
            outputs = self.model(inputs)
            loss = criterion(outputs, targets) / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_mixed_precision:
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Sincronizar métricas en DDP
        if self.use_ddp and dist.is_initialized():
            # All-reduce para loss
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()
        
        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "step": step
        }
    
    def get_world_size(self) -> int:
        """Obtiene world size"""
        if self.use_ddp and dist.is_initialized():
            return dist.get_world_size()
        elif torch.cuda.device_count() > 1:
            return torch.cuda.device_count()
        return 1
    
    def get_rank(self) -> int:
        """Obtiene rank actual"""
        if self.use_ddp and dist.is_initialized():
            return dist.get_rank()
        return 0
    
    def get_model(self) -> nn.Module:
        """Obtiene el modelo (unwrap si es DDP)"""
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            return self.model.module
        return self.model


def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = "nccl"
):
    """Configura proceso distribuido"""
    import os
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )


def cleanup_ddp():
    """Limpia proceso distribuido"""
    if dist.is_initialized():
        dist.destroy_process_group()

