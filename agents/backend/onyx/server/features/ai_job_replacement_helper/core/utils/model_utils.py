"""
Model Utilities - Utilidades para modelos
==========================================

Funciones de utilidad para trabajar con modelos de PyTorch.
Sigue mejores prácticas de deep learning.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


def initialize_weights(
    model: nn.Module,
    method: str = "xavier_uniform",
    gain: float = 1.0
) -> None:
    """
    Inicializar pesos del modelo usando diferentes métodos.
    
    Args:
        model: Modelo PyTorch
        method: Método de inicialización ('xavier_uniform', 'xavier_normal',
                'kaiming_uniform', 'kaiming_normal', 'orthogonal', 'normal')
        gain: Gain para inicialización (usado en algunos métodos)
    """
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if method == "xavier_uniform":
                init.xavier_uniform_(m.weight, gain=gain)
            elif method == "xavier_normal":
                init.xavier_normal_(m.weight, gain=gain)
            elif method == "kaiming_uniform":
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif method == "kaiming_normal":
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif method == "orthogonal":
                init.orthogonal_(m.weight, gain=gain)
            elif method == "normal":
                init.normal_(m.weight, mean=0.0, std=0.02)
            else:
                logger.warning(f"Unknown initialization method: {method}")
                return
            
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if method == "kaiming_uniform":
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif method == "kaiming_normal":
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif method == "xavier_uniform":
                init.xavier_uniform_(m.weight, gain=gain)
            elif method == "xavier_normal":
                init.xavier_normal_(m.weight, gain=gain)
            
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            init.constant_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)
        
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    init.constant_(param.data, 0.0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)
    
    model.apply(init_weights)
    logger.info(f"Weights initialized using {method} method")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Contar parámetros del modelo.
    
    Args:
        model: Modelo PyTorch
        trainable_only: Si True, solo contar parámetros entrenables
    
    Returns:
        Número de parámetros
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module, unit: str = "MB") -> float:
    """
    Obtener tamaño del modelo en memoria.
    
    Args:
        model: Modelo PyTorch
        unit: Unidad ('B', 'KB', 'MB', 'GB')
    
    Returns:
        Tamaño en la unidad especificada
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    divisor = units.get(unit.upper(), units["MB"])
    
    return total_size / divisor


def freeze_model(model: nn.Module, freeze: bool = True) -> None:
    """
    Congelar o descongelar todos los parámetros del modelo.
    
    Args:
        model: Modelo PyTorch
        freeze: Si True, congelar parámetros; si False, descongelar
    """
    for param in model.parameters():
        param.requires_grad = not freeze
    
    logger.info(f"Model parameters {'frozen' if freeze else 'unfrozen'}")


def freeze_layers(
    model: nn.Module,
    layer_names: List[str],
    freeze: bool = True
) -> None:
    """
    Congelar o descongelar capas específicas.
    
    Args:
        model: Modelo PyTorch
        layer_names: Lista de nombres de capas a congelar/descongelar
        freeze: Si True, congelar; si False, descongelar
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = not freeze
    
    logger.info(f"Layers {layer_names} {'frozen' if freeze else 'unfrozen'}")


def get_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Obtener norma de los gradientes del modelo.
    
    Args:
        model: Modelo PyTorch
        norm_type: Tipo de norma (2.0 para L2)
    
    Returns:
        Norma de los gradientes
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> float:
    """
    Recortar gradientes usando norma.
    
    Args:
        model: Modelo PyTorch
        max_norm: Norma máxima permitida
        norm_type: Tipo de norma
    
    Returns:
        Norma total antes del clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type=norm_type
    )
    return total_norm.item()


def set_dropout(model: nn.Module, dropout_rate: float) -> None:
    """
    Establecer tasa de dropout para todas las capas de dropout.
    
    Args:
        model: Modelo PyTorch
        dropout_rate: Nueva tasa de dropout
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.p = dropout_rate
    
    logger.info(f"Dropout rate set to {dropout_rate}")


def set_batch_norm_momentum(model: nn.Module, momentum: float) -> None:
    """
    Establecer momentum para todas las capas de batch normalization.
    
    Args:
        model: Modelo PyTorch
        momentum: Nuevo valor de momentum
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = momentum
    
    logger.info(f"BatchNorm momentum set to {momentum}")


def get_layer_output_shape(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    layer_name: Optional[str] = None
) -> Tuple[int, ...]:
    """
    Obtener forma de salida de una capa específica.
    
    Args:
        model: Modelo PyTorch
        input_shape: Forma de entrada (sin batch dimension)
        layer_name: Nombre de la capa (None para última capa)
    
    Returns:
        Forma de salida
    """
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        
        if layer_name:
            # Hook para capturar salida de capa específica
            output_shape = None
            
            def hook(module, input, output):
                nonlocal output_shape
                output_shape = output.shape[1:]  # Remove batch dimension
            
            for name, module in model.named_modules():
                if name == layer_name:
                    handle = module.register_forward_hook(hook)
                    break
            
            _ = model(dummy_input)
            handle.remove()
            return output_shape if output_shape else input_shape
        else:
            output = model(dummy_input)
            return output.shape[1:]


def check_for_nan_inf(model: nn.Module) -> Dict[str, bool]:
    """
    Verificar si hay valores NaN o Inf en los parámetros del modelo.
    
    Args:
        model: Modelo PyTorch
    
    Returns:
        Diccionario con resultados de la verificación
    """
    has_nan = False
    has_inf = False
    nan_params = []
    inf_params = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            has_nan = True
            nan_params.append(name)
        if torch.isinf(param).any():
            has_inf = True
            inf_params.append(name)
    
    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nan_params": nan_params,
        "inf_params": inf_params,
    }




