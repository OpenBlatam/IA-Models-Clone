"""
Common Utilities - Utilidades compartidas para módulos de deep learning
========================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import wraps
import time

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """Obtiene dispositivo (CUDA si está disponible, sino CPU)"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def move_to_device(data: Any, device: torch.device) -> Any:
    """Mueve datos a dispositivo"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    return data


def calculate_model_size(model: nn.Module, dtype_size: int = 4) -> float:
    """Calcula tamaño del modelo en MB"""
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params * dtype_size
    return size_bytes / (1024 ** 2)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Cuenta parámetros del modelo"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[str] = None
) -> int:
    """Estima FLOPs del modelo"""
    device = get_device(device)
    model = model.to(device)
    model.eval()
    
    flops = 0
    
    def count_flops_hook(module, input, output):
        nonlocal flops
        if isinstance(module, nn.Linear):
            if len(input) > 0 and len(input[0].shape) > 1:
                flops += input[0].shape[1] * output.shape[1]
        elif isinstance(module, nn.Conv2d):
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            output_elements = output.numel()
            flops += kernel_flops * output_elements
    
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(count_flops_hook))
    
    try:
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        logger.warning(f"Error estimando FLOPs: {e}")
    finally:
        for hook in hooks:
            hook.remove()
    
    return flops


def measure_inference_time(
    model: nn.Module,
    example_input: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10,
    device: Optional[str] = None
) -> float:
    """Mide tiempo de inferencia en ms"""
    device = get_device(device)
    model = model.to(device)
    model.eval()
    
    example_input = move_to_device(example_input, device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(example_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(example_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times.append((time.time() - start) * 1000)  # Convert to ms
    
    return sum(times) / len(times) if times else 0.0


def get_model_output(
    model: nn.Module,
    inputs: Any,
    device: Optional[str] = None
) -> torch.Tensor:
    """Obtiene output del modelo, manejando diferentes formatos"""
    device = get_device(device)
    model = model.to(device)
    model.eval()
    
    inputs = move_to_device(inputs, device)
    
    with torch.no_grad():
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)
        
        if hasattr(outputs, 'logits'):
            return outputs.logits
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            return outputs


def extract_predictions(outputs: torch.Tensor) -> torch.Tensor:
    """Extrae predicciones de outputs"""
    if outputs.dim() > 1:
        return torch.argmax(outputs, dim=-1)
    return outputs


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """Calcula accuracy"""
    if predictions.shape != targets.shape:
        predictions = extract_predictions(predictions)
    
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def safe_forward(
    model: nn.Module,
    inputs: Any,
    device: Optional[str] = None
) -> Optional[torch.Tensor]:
    """Forward pass seguro con manejo de errores"""
    try:
        return get_model_output(model, inputs, device)
    except Exception as e:
        logger.error(f"Error en forward pass: {e}")
        return None


def timing_decorator(func):
    """Decorador para medir tiempo de ejecución"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} ejecutado en {elapsed:.4f}s")
        return result
    return wrapper


def validate_model_input(
    inputs: Any,
    expected_type: type = torch.Tensor
) -> bool:
    """Valida tipo de input"""
    if isinstance(inputs, expected_type):
        return True
    elif isinstance(inputs, dict):
        return all(isinstance(v, expected_type) for v in inputs.values() if isinstance(v, torch.Tensor))
    return False


def create_dummy_input(
    input_shape: Tuple[int, ...],
    device: Optional[str] = None
) -> torch.Tensor:
    """Crea input dummy para testing"""
    device = get_device(device)
    return torch.randn(input_shape).to(device)


def check_model_health(
    model: nn.Module,
    device: Optional[str] = None
) -> Dict[str, bool]:
    """Verifica salud básica del modelo"""
    device = get_device(device)
    health = {
        "has_parameters": len(list(model.parameters())) > 0,
        "has_gradients": any(p.grad is not None for p in model.parameters() if p.requires_grad),
        "device_compatible": True
    }
    
    try:
        model = model.to(device)
        health["device_compatible"] = True
    except Exception:
        health["device_compatible"] = False
    
    return health




