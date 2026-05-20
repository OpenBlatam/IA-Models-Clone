"""
Optimization Utils - Utilidades de Optimización
===============================================

Utilidades para optimización avanzada de modelos y entrenamiento.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MixedPrecisionManager:
    """
    Gestor de mixed precision training.
    """
    
    def __init__(self, enabled: bool = True, device: str = "cuda"):
        """
        Inicializar gestor.
        
        Args:
            enabled: Si está habilitado
            device: Dispositivo
        """
        self.enabled = enabled and device == "cuda" and torch.cuda.is_available()
        self.device = device
        
        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    @contextmanager
    def autocast(self):
        """
        Context manager para autocast.
        
        Yields:
            Contexto de autocast
        """
        if self.enabled:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Escalar pérdida.
        
        Args:
            loss: Pérdida
            
        Returns:
            Pérdida escalada
        """
        if self.scaler:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Step del optimizador.
        
        Args:
            optimizer: Optimizador
        """
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


class GradientAccumulator:
    """
    Acumulador de gradientes para batches grandes.
    """
    
    def __init__(self, accumulation_steps: int = 1):
        """
        Inicializar acumulador.
        
        Args:
            accumulation_steps: Pasos de acumulación
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        """
        Verificar si debe hacer step.
        
        Returns:
            True si debe hacer step
        """
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self) -> None:
        """Resetear contador"""
        self.current_step = 0


class ModelOptimizer:
    """
    Optimizador de modelos para inferencia.
    """
    
    @staticmethod
    def compile_model(model: nn.Module) -> nn.Module:
        """
        Compilar modelo para optimización (PyTorch 2.0+).
        
        Args:
            model: Modelo
            
        Returns:
            Modelo compilado
        """
        try:
            if hasattr(torch, 'compile'):
                return torch.compile(model)
            else:
                logger.warning("torch.compile not available, returning original model")
                return model
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}, returning original model")
            return model
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Optimizar modelo para inferencia.
        
        Args:
            model: Modelo
            
        Returns:
            Modelo optimizado
        """
        model.eval()
        
        # Fusionar operaciones
        try:
            if hasattr(torch.jit, 'optimize_for_inference'):
                model = torch.jit.optimize_for_inference(torch.jit.script(model))
        except Exception:
            pass
        
        return model
    
    @staticmethod
    def quantize_model(
        model: nn.Module,
        quantization_type: str = "dynamic"
    ) -> nn.Module:
        """
        Cuantizar modelo.
        
        Args:
            model: Modelo
            quantization_type: Tipo (dynamic, static, qat)
            
        Returns:
            Modelo cuantizado
        """
        if quantization_type == "dynamic":
            return torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
        elif quantization_type == "static":
            # Requiere calibración
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Nota: Requiere calibración antes de convertir
            return model
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")


class MultiGPUTrainer:
    """
    Trainer para entrenamiento multi-GPU.
    """
    
    @staticmethod
    def setup_ddp(
        model: nn.Module,
        device_ids: Optional[List[int]] = None
    ) -> nn.Module:
        """
        Configurar DistributedDataParallel.
        
        Args:
            model: Modelo
            device_ids: IDs de dispositivos
            
        Returns:
            Modelo envuelto en DDP
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, returning original model")
            return model
        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
            logger.info(f"Model wrapped in DataParallel for devices: {device_ids}")
        else:
            model = model.to(f"cuda:{device_ids[0]}")
        
        return model
    
    @staticmethod
    def setup_ddp_distributed(
        model: nn.Module,
        device_id: int
    ) -> nn.Module:
        """
        Configurar DistributedDataParallel para entrenamiento distribuido.
        
        Args:
            model: Modelo
            device_id: ID de dispositivo
            
        Returns:
            Modelo envuelto en DDP
        """
        model = model.to(device_id)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device_id]
        )
        logger.info(f"Model wrapped in DistributedDataParallel on device: {device_id}")
        return model


class MemoryOptimizer:
    """
    Optimizador de memoria para entrenamiento.
    """
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> None:
        """
        Habilitar gradient checkpointing.
        
        Args:
            model: Modelo
        """
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    @staticmethod
    def clear_cache() -> None:
        """Limpiar caché de CUDA"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """
        Obtener estadísticas de memoria.
        
        Returns:
            Diccionario con estadísticas
        """
        if not torch.cuda.is_available():
            return {}
        
        stats = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            stats[f"gpu_{i}_allocated_gb"] = allocated
            stats[f"gpu_{i}_reserved_gb"] = reserved
        
        return stats




