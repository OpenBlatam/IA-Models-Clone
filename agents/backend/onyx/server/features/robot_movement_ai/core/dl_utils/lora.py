"""
LoRA (Low-Rank Adaptation)
===========================

Implementación de LoRA para fine-tuning eficiente.
"""

import logging
from typing import Optional
import math

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Capa LoRA.
    
    Aplica adaptación de bajo rango a una capa lineal.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        Inicializar capa LoRA.
        
        Args:
            in_features: Features de entrada
            out_features: Features de salida
            rank: Rango de la descomposición
            alpha: Factor de escalado
            dropout: Tasa de dropout
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LoRA")
        
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrices de bajo rango
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Inicialización
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [..., in_features]
            
        Returns:
            Tensor de salida [..., out_features]
        """
        # LoRA: x @ A^T @ B^T * scaling
        x = self.dropout(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return lora_output


def apply_lora(
    module: nn.Module,
    target_modules: list,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> nn.Module:
    """
    Aplicar LoRA a módulos específicos.
    
    Args:
        module: Módulo base
        target_modules: Lista de nombres de módulos a adaptar
        rank: Rango de LoRA
        alpha: Factor de escalado
        dropout: Tasa de dropout
        
    Returns:
        Módulo con LoRA aplicado
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for LoRA")
    
    for name, child in module.named_modules():
        if name in target_modules and isinstance(child, nn.Linear):
            # Reemplazar con LoRA
            lora_layer = LoRALayer(
                child.in_features,
                child.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            
            # Guardar peso original
            lora_layer.register_buffer('weight', child.weight.data.clone())
            
            # Reemplazar módulo
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = module
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, lora_layer)
            else:
                setattr(module, child_name, lora_layer)
            
            logger.info(f"Applied LoRA to {name}")
    
    return module


def remove_lora(module: nn.Module):
    """
    Remover LoRA y restaurar pesos originales.
    
    Args:
        module: Módulo con LoRA
    """
    # Implementación simplificada
    # En producción, se necesitaría guardar los pesos originales
    logger.warning("LoRA removal not fully implemented")

