"""
Memory Optimizer - Optimizaciones agresivas de memoria
=======================================================

Utilidades para optimizar uso de memoria y permitir batches más grandes:
- Gradient checkpointing
- Activation offloading
- Memory-efficient attention
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Optimizador de memoria para modelos grandes"""
    
    def __init__(self):
        """Inicializa el optimizador de memoria"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de optimización de memoria.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        perf_dir = utils_dir / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_memory_optimizer(perf_dir, keywords, project_info)
    
    def _generate_memory_optimizer(
        self,
        perf_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera optimizador de memoria"""
        
        optimizer_content = '''"""
Memory Optimizer - Optimizaciones agresivas de memoria
=======================================================

Utilidades para reducir uso de memoria y permitir batches más grandes.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


def enable_gradient_checkpointing(
    model: nn.Module,
    checkpoint_every: int = 1,
) -> nn.Module:
    """
    Habilita gradient checkpointing para ahorrar memoria.
    
    Args:
        model: Modelo a optimizar
        checkpoint_every: Cada cuántas capas hacer checkpoint
    
    Returns:
        Modelo con gradient checkpointing habilitado
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing habilitado")
    else:
        # Para modelos personalizados, aplicar manualmente
        try:
            from torch.utils.checkpoint import checkpoint
            
            def checkpoint_forward(module, *args, **kwargs):
                def custom_forward(*inputs):
                    return module(*inputs)
                return checkpoint(custom_forward, *args, use_reentrant=False)
            
            # Aplicar a capas específicas
            for name, module in model.named_modules():
                if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                    module.forward = lambda *args, **kwargs: checkpoint_forward(module, *args, **kwargs)
            
            logger.info("Gradient checkpointing aplicado manualmente")
        except Exception as e:
            logger.warning(f"No se pudo aplicar gradient checkpointing: {e}")
    
    return model


def enable_activation_offloading(
    model: nn.Module,
) -> nn.Module:
    """
    Habilita activation offloading para ahorrar memoria.
    
    Args:
        model: Modelo a optimizar
    
    Returns:
        Modelo con activation offloading
    """
    try:
        # Para transformers
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        
        # Usar CPU offload para activaciones
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Guardar activaciones en CPU
                original_forward = module.forward
                
                def offload_forward(self, x):
                    x = x.cpu()
                    result = original_forward(x)
                    return result.to(next(self.parameters()).device)
                
                module.forward = offload_forward.__get__(module, type(module))
        
        logger.info("Activation offloading habilitado")
    except Exception as e:
        logger.warning(f"No se pudo aplicar activation offloading: {e}")
    
    return model


def optimize_memory_for_training(
    model: nn.Module,
    use_gradient_checkpointing: bool = True,
    use_activation_offloading: bool = False,
    use_8bit_optimizer: bool = False,
) -> nn.Module:
    """
    Aplica todas las optimizaciones de memoria para entrenamiento.
    
    Args:
        model: Modelo a optimizar
        use_gradient_checkpointing: Si usar gradient checkpointing
        use_activation_offloading: Si usar activation offloading
        use_8bit_optimizer: Si usar optimizador 8-bit (bitsandbytes)
    
    Returns:
        Modelo optimizado
    """
    if use_gradient_checkpointing:
        model = enable_gradient_checkpointing(model)
    
    if use_activation_offloading:
        model = enable_activation_offloading(model)
    
    # Optimizaciones adicionales
    if hasattr(model, "config"):
        # Para modelos de transformers
        if hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = use_gradient_checkpointing
    
    logger.info("Optimizaciones de memoria aplicadas")
    return model


def get_memory_usage(device: str = "cuda") -> Dict[str, float]:
    """
    Obtiene uso actual de memoria.
    
    Args:
        device: Dispositivo a consultar
    
    Returns:
        Diccionario con estadísticas de memoria
    """
    if device == "cuda" and torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        }
    return {}


def clear_memory_cache(device: str = "cuda") -> None:
    """
    Limpia caché de memoria.
    
    Args:
        device: Dispositivo a limpiar
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Caché de memoria limpiado")
'''
        
        (perf_dir / "memory_optimizer.py").write_text(optimizer_content, encoding="utf-8")

