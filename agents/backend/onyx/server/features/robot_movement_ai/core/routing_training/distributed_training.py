"""
Distributed Training Support
============================

Soporte para entrenamiento distribuido con DataParallel y DistributedDataParallel.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Wrapper para entrenamiento distribuido.
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_ddp: bool = True,
        use_dp: bool = False,
        device_ids: Optional[list] = None
    ):
        """
        Inicializar entrenador distribuido.
        
        Args:
            model: Modelo a entrenar
            use_ddp: Usar DistributedDataParallel (recomendado)
            use_dp: Usar DataParallel (más simple pero menos eficiente)
            device_ids: IDs de dispositivos para DataParallel
        """
        self.model = model
        self.use_ddp = use_ddp
        self.use_dp = use_dp
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        
        if use_ddp and torch.cuda.device_count() > 1:
            self._setup_ddp()
        elif use_dp and torch.cuda.device_count() > 1:
            self._setup_dp()
        else:
            logger.info("Entrenamiento en un solo dispositivo")
    
    def _setup_ddp(self):
        """Configurar DistributedDataParallel."""
        if not torch.distributed.is_available():
            logger.warning("PyTorch distributed no disponible, usando modelo normal")
            return
        
        try:
            # Inicializar proceso grupal
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend='nccl')
            
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f'cuda:{local_rank}')
            self.model = self.model.to(device)
            
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
            
            logger.info(f"DistributedDataParallel configurado en dispositivo {local_rank}")
        except Exception as e:
            logger.warning(f"Error configurando DDP: {e}, usando modelo normal")
    
    def _setup_dp(self):
        """Configurar DataParallel."""
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
            logger.info(f"DataParallel configurado en {len(self.device_ids)} dispositivos")
        else:
            logger.warning("Solo hay 1 GPU disponible, DataParallel no es necesario")
    
    def get_model(self) -> nn.Module:
        """
        Obtener modelo (desempaquetar si es necesario).
        
        Returns:
            Modelo
        """
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.model.module
        return self.model
    
    def save_checkpoint(self, path: str, **kwargs):
        """Guardar checkpoint (maneja modelos distribuidos)."""
        model_to_save = self.get_model()
        if hasattr(model_to_save, 'save_checkpoint'):
            model_to_save.save_checkpoint(path, **kwargs)
        else:
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                **kwargs
            }, path)


class GradientAccumulator:
    """
    Acumulador de gradientes para batches grandes.
    """
    
    def __init__(self, accumulation_steps: int = 4):
        """
        Inicializar acumulador.
        
        Args:
            accumulation_steps: Número de pasos a acumular
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_update(self) -> bool:
        """Verificar si se debe actualizar."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def step(self):
        """Incrementar contador."""
        self.current_step += 1
    
    def reset(self):
        """Resetear contador."""
        self.current_step = 0


