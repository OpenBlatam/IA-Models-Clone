"""
Debugging Utilities - Utilidades de debugging
===============================================

Funciones para debugging y diagnóstico de modelos.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def check_model_health(model: nn.Module) -> Dict[str, Any]:
    """
    Verificar salud general del modelo.
    
    Args:
        model: Modelo a verificar
    
    Returns:
        Diccionario con información de salud
    """
    health = {
        "has_parameters": False,
        "has_gradients": False,
        "has_nan": False,
        "has_inf": False,
        "frozen_params": 0,
        "trainable_params": 0,
        "total_params": 0,
        "issues": [],
    }
    
    try:
        # Contar parámetros
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
                # Verificar gradientes
                if param.grad is not None:
                    health["has_gradients"] = True
                    if torch.isnan(param.grad).any():
                        health["has_nan"] = True
                        health["issues"].append("NaN in gradients")
                    if torch.isinf(param.grad).any():
                        health["has_inf"] = True
                        health["issues"].append("Inf in gradients")
            else:
                frozen_params += param.numel()
        
        health["total_params"] = total_params
        health["trainable_params"] = trainable_params
        health["frozen_params"] = frozen_params
        health["has_parameters"] = total_params > 0
        
        # Verificar parámetros del modelo
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                health["has_nan"] = True
                health["issues"].append(f"NaN in parameters: {name}")
            if torch.isinf(param).any():
                health["has_inf"] = True
                health["issues"].append(f"Inf in parameters: {name}")
        
        health["is_healthy"] = len(health["issues"]) == 0
        
    except Exception as e:
        health["error"] = str(e)
        health["is_healthy"] = False
    
    return health


def diagnose_training_issue(
    model: nn.Module,
    loss: float,
    gradients: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, Any]:
    """
    Diagnosticar problemas comunes en entrenamiento.
    
    Args:
        model: Modelo
        loss: Pérdida actual
        gradients: Gradientes (opcional)
    
    Returns:
        Diccionario con diagnóstico
    """
    diagnosis = {
        "loss_value": loss,
        "loss_is_nan": np.isnan(loss),
        "loss_is_inf": np.isinf(loss),
        "loss_too_high": loss > 1000,
        "loss_too_low": loss < 1e-10,
        "gradient_issues": [],
        "recommendations": [],
    }
    
    # Verificar pérdida
    if diagnosis["loss_is_nan"]:
        diagnosis["recommendations"].append("Loss is NaN - check learning rate, data, or model")
    if diagnosis["loss_is_inf"]:
        diagnosis["recommendations"].append("Loss is Inf - check for numerical instability")
    if diagnosis["loss_too_high"]:
        diagnosis["recommendations"].append("Loss is very high - consider reducing learning rate")
    if diagnosis["loss_too_low"]:
        diagnosis["recommendations"].append("Loss is very low - model might be overfitting")
    
    # Verificar gradientes
    if gradients:
        for name, grad in gradients.items():
            if grad is not None:
                grad_norm = grad.norm().item()
                if grad_norm > 100:
                    diagnosis["gradient_issues"].append(f"Large gradient in {name}: {grad_norm:.2f}")
                    diagnosis["recommendations"].append("Consider gradient clipping")
                if torch.isnan(grad).any():
                    diagnosis["gradient_issues"].append(f"NaN gradient in {name}")
                if torch.isinf(grad).any():
                    diagnosis["gradient_issues"].append(f"Inf gradient in {name}")
    
    return diagnosis


def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    input_shape: Tuple[int, ...]
) -> Dict[str, Any]:
    """
    Comparar dos modelos.
    
    Args:
        model1: Primer modelo
        model2: Segundo modelo
        input_shape: Forma de entrada (sin batch dimension)
    
    Returns:
        Diccionario con comparación
    """
    comparison = {
        "same_architecture": False,
        "same_parameters": False,
        "same_output": False,
        "parameter_diff": {},
    }
    
    # Comparar arquitectura
    comparison["same_architecture"] = str(model1) == str(model2)
    
    # Comparar parámetros
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    if set(params1.keys()) == set(params2.keys()):
        comparison["same_parameters"] = True
        for name in params1.keys():
            diff = (params1[name] - params2[name]).abs().mean().item()
            comparison["parameter_diff"][name] = diff
        comparison["same_parameters"] = all(
            diff < 1e-6 for diff in comparison["parameter_diff"].values()
        )
    
    # Comparar salidas
    try:
        model1.eval()
        model2.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            output1 = model1(dummy_input)
            output2 = model2(dummy_input)
            
            output_diff = (output1 - output2).abs().mean().item()
            comparison["output_diff"] = output_diff
            comparison["same_output"] = output_diff < 1e-5
    except Exception as e:
        comparison["output_error"] = str(e)
    
    return comparison


def trace_model_forward(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Trazar el forward pass del modelo.
    
    Args:
        model: Modelo
        input_shape: Forma de entrada (sin batch dimension)
        verbose: Si True, mostrar información detallada
    
    Returns:
        Diccionario con información del forward pass
    """
    trace_info = {
        "layers": [],
        "shapes": [],
        "activations": [],
    }
    
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            trace_info["layers"].append(name)
            trace_info["shapes"].append(str(output.shape))
            if verbose:
                logger.info(f"{name}: {output.shape}")
        return hook
    
    # Registrar hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            handle = module.register_forward_hook(hook_fn(name))
            hooks.append(handle)
    
    # Forward pass
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            _ = model(dummy_input)
    finally:
        # Remover hooks
        for handle in hooks:
            handle.remove()
    
    return trace_info




