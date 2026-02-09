"""
LoRA (Low-Rank Adaptation) for Efficient Fine-tuning
=====================================================

Implementación de LoRA para fine-tuning eficiente de modelos.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Capa LoRA para adaptación de bajo rango.
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
            rank: Rango de la descomposición (r)
            alpha: Factor de escalado (alpha)
            dropout: Tasa de dropout
        """
        super(LoRALayer, self).__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrices de bajo rango
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor transformado
        """
        # x @ A^T @ B^T * scaling
        x = self.dropout(x)
        x = x @ self.lora_A.T
        x = x @ self.lora_B.T
        return x * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer con LoRA adaptado.
    """
    
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        Inicializar LoRA Linear.
        
        Args:
            linear_layer: Capa Linear original
            rank: Rango de LoRA
            alpha: Factor de escalado
            dropout: Tasa de dropout
        """
        super(LoRALinear, self).__init__()
        
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Congelar pesos originales
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        # Salida original + adaptación LoRA
        return self.linear(x) + self.lora(x)
    
    def merge_weights(self):
        """Fusionar pesos LoRA con pesos originales."""
        with torch.no_grad():
            # W_new = W_original + B @ A * scaling
            lora_weight = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            self.linear.weight.data += lora_weight.T


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> nn.Module:
    """
    Aplicar LoRA a un modelo.
    
    Args:
        model: Modelo a adaptar
        target_modules: Lista de nombres de módulos a adaptar (None = todos los Linear)
        rank: Rango de LoRA
        alpha: Factor de escalado
        dropout: Tasa de dropout
        
    Returns:
        Modelo con LoRA aplicado
    """
    if target_modules is None:
        target_modules = []
    
    def _apply_lora_recursive(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                if not target_modules or any(tm in full_name for tm in target_modules):
                    # Reemplazar con LoRALinear
                    lora_linear = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, name, lora_linear)
                    logger.info(f"LoRA aplicado a: {full_name}")
            else:
                _apply_lora_recursive(child, full_name)
    
    _apply_lora_recursive(model)
    return model


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Contar parámetros LoRA vs totales.
    
    Args:
        model: Modelo
        
    Returns:
        Diccionario con conteos
    """
    total_params = 0
    trainable_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            if 'lora' in name.lower():
                lora_params += num_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "lora_parameters": lora_params,
        "frozen_parameters": total_params - trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0
    }


