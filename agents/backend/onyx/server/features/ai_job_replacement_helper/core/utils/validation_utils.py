"""
Validation Utilities - Utilidades de validación
================================================

Funciones para validar modelos, datos y configuraciones.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def validate_model_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validar configuración de modelo.
    
    Args:
        config: Diccionario de configuración
    
    Returns:
        Tupla con (es_válido, lista_de_errores)
    """
    errors = []
    
    # Validar campos requeridos
    required_fields = ["input_size", "output_size"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validar tipos
    if "input_size" in config and not isinstance(config["input_size"], int):
        errors.append("input_size must be an integer")
    
    if "output_size" in config and not isinstance(config["output_size"], int):
        errors.append("output_size must be an integer")
    
    # Validar valores
    if "input_size" in config and config["input_size"] <= 0:
        errors.append("input_size must be positive")
    
    if "output_size" in config and config["output_size"] <= 0:
        errors.append("output_size must be positive")
    
    if "dropout" in config:
        dropout = config["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout <= 1):
            errors.append("dropout must be between 0 and 1")
    
    return len(errors) == 0, errors


def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validar configuración de entrenamiento.
    
    Args:
        config: Diccionario de configuración
    
    Returns:
        Tupla con (es_válido, lista_de_errores)
    """
    errors = []
    
    # Validar epochs
    if "num_epochs" in config:
        if not isinstance(config["num_epochs"], int) or config["num_epochs"] <= 0:
            errors.append("num_epochs must be a positive integer")
    
    # Validar batch_size
    if "batch_size" in config:
        if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
            errors.append("batch_size must be a positive integer")
    
    # Validar learning_rate
    if "learning_rate" in config:
        lr = config["learning_rate"]
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors.append("learning_rate must be a positive number")
        if lr > 1.0:
            errors.append("learning_rate seems too high (typically < 1.0)")
    
    # Validar weight_decay
    if "weight_decay" in config:
        wd = config["weight_decay"]
        if not isinstance(wd, (int, float)) or wd < 0:
            errors.append("weight_decay must be non-negative")
    
    return len(errors) == 0, errors


def validate_data_shape(
    data: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str = "data"
) -> Tuple[bool, Optional[str]]:
    """
    Validar forma de datos.
    
    Args:
        data: Tensor de datos
        expected_shape: Forma esperada
        name: Nombre del tensor (para mensajes de error)
    
    Returns:
        Tupla con (es_válido, mensaje_de_error)
    """
    if data.shape != expected_shape:
        return False, f"{name} shape {data.shape} doesn't match expected {expected_shape}"
    return True, None


def validate_model_output(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    expected_output_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[bool, Optional[str], Optional[torch.Tensor]]:
    """
    Validar que el modelo produce la salida esperada.
    
    Args:
        model: Modelo a validar
        input_shape: Forma de entrada (sin batch dimension)
        expected_output_shape: Forma esperada de salida (opcional)
    
    Returns:
        Tupla con (es_válido, mensaje_de_error, output)
    """
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            output = model(dummy_input)
            
            if expected_output_shape:
                expected = (1,) + expected_output_shape
                if output.shape != expected:
                    return False, \
                        f"Output shape {output.shape} doesn't match expected {expected}", \
                        None
            
            # Verificar que no hay NaN o Inf
            if torch.isnan(output).any():
                return False, "Model output contains NaN", output
            
            if torch.isinf(output).any():
                return False, "Model output contains Inf", output
            
            return True, None, output
            
    except Exception as e:
        return False, f"Error validating model: {str(e)}", None


def validate_gradients(model: nn.Module) -> Tuple[bool, Dict[str, Any]]:
    """
    Validar gradientes del modelo.
    
    Args:
        model: Modelo con gradientes calculados
    
    Returns:
        Tupla con (es_válido, información_de_gradientes)
    """
    info = {
        "has_gradients": False,
        "gradient_norms": {},
        "has_nan": False,
        "has_inf": False,
        "max_norm": 0.0,
    }
    
    has_gradients = False
    max_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            info["gradient_norms"][name] = grad_norm
            max_norm = max(max_norm, grad_norm)
            
            if torch.isnan(param.grad).any():
                info["has_nan"] = True
            if torch.isinf(param.grad).any():
                info["has_inf"] = True
    
    info["has_gradients"] = has_gradients
    info["max_norm"] = max_norm
    
    is_valid = has_gradients and not info["has_nan"] and not info["has_inf"]
    
    return is_valid, info


def check_device_compatibility(
    model: nn.Module,
    data: torch.Tensor
) -> Tuple[bool, Optional[str]]:
    """
    Verificar que modelo y datos están en el mismo dispositivo.
    
    Args:
        model: Modelo
        data: Datos
    
    Returns:
        Tupla con (es_compatible, mensaje_de_error)
    """
    model_device = next(model.parameters()).device
    data_device = data.device
    
    if model_device != data_device:
        return False, \
            f"Model on {model_device} but data on {data_device}"
    
    return True, None




