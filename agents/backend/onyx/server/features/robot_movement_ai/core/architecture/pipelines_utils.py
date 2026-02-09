"""
Pipeline Utilities Module
==========================

Utilidades profesionales para pipelines de deep learning.
Incluye validación, debugging, profiling y helpers.
"""

import logging
import time
import warnings
from typing import Dict, Any, Optional, List, Tuple, Callable
from contextlib import contextmanager
from functools import wraps
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.profiler import profile, record_function, ProfilerActivity
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    profile = None
    logging.warning("PyTorch not available. Some utilities will be disabled.")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

logger = logging.getLogger(__name__)


def validate_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    check_nan: bool = True,
    check_inf: bool = True,
    check_shape: Optional[Tuple] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> bool:
    """
    Validar tensor de PyTorch.
    
    Args:
        tensor: Tensor a validar
        name: Nombre del tensor para mensajes de error
        check_nan: Verificar NaN
        check_inf: Verificar Inf
        check_shape: Tupla con forma esperada (None para ignorar dimensión)
        min_val: Valor mínimo permitido
        max_val: Valor máximo permitido
        
    Returns:
        True si es válido
        
    Raises:
        ValueError: Si el tensor no es válido
    """
    if not TORCH_AVAILABLE:
        return True
    
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if check_nan and torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    
    if check_inf and torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")
    
    if check_shape:
        if len(tensor.shape) != len(check_shape):
            raise ValueError(
                f"{name} has {len(tensor.shape)} dimensions, expected {len(check_shape)}"
            )
        for i, (actual, expected) in enumerate(zip(tensor.shape, check_shape)):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} dimension {i} is {actual}, expected {expected}"
                )
    
    if min_val is not None and (tensor < min_val).any():
        raise ValueError(f"{name} contains values below {min_val}")
    
    if max_val is not None and (tensor > max_val).any():
        raise ValueError(f"{name} contains values above {max_val}")
    
    return True


def check_gradients(model: nn.Module, max_norm: float = 10.0) -> Dict[str, float]:
    """
    Verificar gradientes del modelo.
    
    Args:
        model: Modelo PyTorch
        max_norm: Norma máxima esperada
        
    Returns:
        Dict con estadísticas de gradientes
    """
    if not TORCH_AVAILABLE:
        return {}
    
    stats = {
        "total_norm": 0.0,
        "max_grad": float('-inf'),
        "min_grad": float('inf'),
        "num_params": 0,
        "num_zero_grads": 0
    }
    
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            stats["max_grad"] = max(stats["max_grad"], param.grad.data.abs().max().item())
            stats["min_grad"] = min(stats["min_grad"], param.grad.data.abs().min().item())
            stats["num_params"] += 1
            
            if (param.grad.data.abs() < 1e-8).all():
                stats["num_zero_grads"] += 1
        else:
            logger.warning(f"Parameter {name} has no gradient")
    
    stats["total_norm"] = total_norm ** 0.5
    
    if stats["total_norm"] > max_norm:
        warnings.warn(
            f"Gradient norm ({stats['total_norm']:.2f}) exceeds max_norm ({max_norm})"
        )
    
    return stats


@contextmanager
def detect_anomaly():
    """Context manager para detectar anomalías en autograd."""
    if not TORCH_AVAILABLE:
        yield
        return
    
    torch.autograd.set_detect_anomaly(True)
    try:
        yield
    finally:
        torch.autograd.set_detect_anomaly(False)


@contextmanager
def profile_training(
    activities: Optional[List] = None,
    record_shapes: bool = True,
    profile_memory: bool = True
):
    """
    Context manager para profiling de entrenamiento.
    
    Args:
        activities: Lista de actividades a perfilar
        record_shapes: Registrar formas de tensores
        profile_memory: Perfilar uso de memoria
    """
    if not TORCH_AVAILABLE or profile is None:
        yield None
        return
    
    if activities is None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory
    ) as prof:
        yield prof
    
    # Exportar resultados
    prof.export_chrome_trace("trace.json")
    logger.info("Profiling trace exported to trace.json")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """
    Contar parámetros del modelo.
    
    Args:
        model: Modelo PyTorch
        trainable_only: Solo contar parámetros entrenables
        
    Returns:
        Dict con estadísticas de parámetros
    """
    if not TORCH_AVAILABLE:
        return {}
    
    total = 0
    trainable = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params
    
    return {
        "total": total,
        "trainable": trainable if trainable_only else total,
        "non_trainable": total - trainable,
        "total_mb": total * 4 / (1024 ** 2),  # Asumiendo float32
        "trainable_mb": trainable * 4 / (1024 ** 2) if trainable_only else 0
    }


def get_model_summary(model: nn.Module, input_size: Tuple) -> str:
    """
    Obtener resumen del modelo.
    
    Args:
        model: Modelo PyTorch
        input_size: Tamaño de entrada (sin batch dimension)
        
    Returns:
        String con resumen del modelo
    """
    if not TORCH_AVAILABLE:
        return "PyTorch not available"
    
    try:
        from torchsummary import summary
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            summary(model, input_size)
        return f.getvalue()
    except ImportError:
        # Fallback simple
        param_stats = count_parameters(model)
        return (
            f"Model: {model.__class__.__name__}\n"
            f"Total parameters: {param_stats['total']:,}\n"
            f"Trainable parameters: {param_stats['trainable']:,}\n"
            f"Model size: {param_stats['total_mb']:.2f} MB"
        )


def set_seed(seed: int = 42):
    """
    Establecer semilla para reproducibilidad.
    
    Args:
        seed: Semilla
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    logger.info(f"Seed set to {seed}")


def freeze_model(model: nn.Module, freeze: bool = True):
    """
    Congelar o descongelar parámetros del modelo.
    
    Args:
        model: Modelo PyTorch
        freeze: True para congelar, False para descongelar
    """
    if not TORCH_AVAILABLE:
        return
    
    for param in model.parameters():
        param.requires_grad = not freeze
    
    logger.info(f"Model parameters {'frozen' if freeze else 'unfrozen'}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Cargar checkpoint del modelo.
    
    Args:
        checkpoint_path: Ruta al checkpoint
        model: Modelo a cargar
        optimizer: Optimizador (opcional)
        scheduler: Scheduler (opcional)
        device: Dispositivo
        
    Returns:
        Dict con información del checkpoint
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Cargar estado del modelo
    if hasattr(model, 'module'):  # DataParallel
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Cargar optimizador
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Cargar scheduler
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float('inf')),
        "config": checkpoint.get("config", {})
    }


def get_device_info() -> Dict[str, Any]:
    """
    Obtener información del dispositivo.
    
    Returns:
        Dict con información de GPU/CPU
    """
    info = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "cuda_device_count": 0,
        "current_device": None,
        "device_name": None
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
    
    return info


def time_function(func: Callable) -> Callable:
    """
    Decorador para medir tiempo de ejecución.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.debug(f"{func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper


class GradientMonitor:
    """Monitor para gradientes durante entrenamiento."""
    
    def __init__(self, log_interval: int = 100):
        """
        Inicializar monitor de gradientes.
        
        Args:
            log_interval: Intervalo de logging
        """
        self.log_interval = log_interval
        self.step_count = 0
        self.gradient_history: List[Dict[str, float]] = []
    
    def check(self, model: nn.Module) -> Optional[Dict[str, float]]:
        """
        Verificar gradientes.
        
        Args:
            model: Modelo a verificar
            
        Returns:
            Estadísticas de gradientes o None
        """
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            stats = check_gradients(model)
            self.gradient_history.append(stats)
            logger.info(
                f"Gradient stats (step {self.step_count}): "
                f"norm={stats['total_norm']:.4f}, "
                f"max={stats['max_grad']:.4f}, "
                f"min={stats['min_grad']:.4f}"
            )
            return stats
        
        return None
    
    def get_history(self) -> List[Dict[str, float]]:
        """Obtener historial de gradientes."""
        return self.gradient_history
