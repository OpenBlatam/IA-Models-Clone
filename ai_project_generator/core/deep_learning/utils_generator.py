"""
Utils Generator - Generador de utilidades adicionales
======================================================

Genera utilidades adicionales para proyectos de Deep Learning:
- Debugging utilities
- Visualization utilities
- Data preprocessing
- Model inspection
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class UtilsGenerator:
    """Generador de utilidades adicionales"""
    
    def __init__(self):
        """Inicializa el generador de utilidades"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades adicionales.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar utilidades
        self._generate_debugging_utils(utils_dir, keywords, project_info)
        self._generate_visualization_utils(utils_dir, keywords, project_info)
        self._generate_preprocessing_utils(utils_dir, keywords, project_info)
        self._generate_inspection_utils(utils_dir, keywords, project_info)
        self._generate_logging_utils(utils_dir, keywords, project_info)
        self._generate_utils_init(utils_dir, keywords)
    
    def _generate_utils_init(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de utilidades"""
        
        init_content = '''"""
Additional Utilities Module
============================

Utilidades adicionales para debugging, visualización y procesamiento.
"""

from .debugging_utils import (
    enable_debug_mode,
    check_model_health,
    validate_model_outputs,
)
from .visualization_utils import (
    plot_training_curves,
    visualize_attention,
    plot_confusion_matrix,
)
from .preprocessing_utils import (
    normalize_data,
    augment_data,
    prepare_dataset,
)
from .inspection_utils import (
    inspect_model_architecture,
    count_parameters,
    analyze_model_complexity,
)
from .logging_utils import (
    setup_logging,
    get_logger,
    log_training_metrics,
)

__all__ = [
    "enable_debug_mode",
    "check_model_health",
    "validate_model_outputs",
    "plot_training_curves",
    "visualize_attention",
    "plot_confusion_matrix",
    "normalize_data",
    "augment_data",
    "prepare_dataset",
    "inspect_model_architecture",
    "count_parameters",
    "analyze_model_complexity",
    "setup_logging",
    "get_logger",
    "log_training_metrics",
]
'''
        
        utils_subdir = utils_dir / "utils"
        utils_subdir.mkdir(parents=True, exist_ok=True)
        (utils_subdir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_debugging_utils(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de debugging"""
        
        debugging_content = '''"""
Debugging Utilities - Utilidades para debugging
================================================

Herramientas para debugging y diagnóstico de modelos.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def enable_debug_mode(verbose: bool = True) -> None:
    """
    Habilita modo debug con detección de anomalías.
    
    Args:
        verbose: Si mostrar información detallada
    """
    torch.autograd.set_detect_anomaly(True)
    if verbose:
        logger.warning("Modo debug habilitado - puede ser lento")
        logger.info("Detección de anomalías en autograd activada")


def check_model_health(
    model: nn.Module,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Verifica la salud del modelo.
    
    Args:
        model: Modelo a verificar
        device: Dispositivo a usar
    
    Returns:
        Diccionario con información de salud
    """
    health_status = {
        "has_nan": False,
        "has_inf": False,
        "gradient_flow": True,
        "parameter_stats": {},
    }
    
    # Verificar parámetros
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).any():
                health_status["has_nan"] = True
                logger.error(f"NaN encontrado en parámetro: {name}")
            
            if torch.isinf(param).any():
                health_status["has_inf"] = True
                logger.error(f"Inf encontrado en parámetro: {name}")
            
            # Estadísticas de parámetros
            health_status["parameter_stats"][name] = {
                "mean": float(param.data.mean().item()),
                "std": float(param.data.std().item()),
                "min": float(param.data.min().item()),
                "max": float(param.data.max().item()),
            }
    
    # Verificar gradientes
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            if torch.isnan(param.grad).any():
                health_status["has_nan"] = True
                logger.error(f"NaN en gradiente: {name}")
            if torch.isinf(param.grad).any():
                health_status["has_inf"] = True
                logger.error(f"Inf en gradiente: {name}")
    
    health_status["gradient_flow"] = has_grad
    
    return health_status


def validate_model_outputs(
    outputs: torch.Tensor,
    expected_shape: Optional[tuple] = None,
    check_finite: bool = True,
) -> Dict[str, Any]:
    """
    Valida outputs del modelo.
    
    Args:
        outputs: Outputs del modelo
        expected_shape: Shape esperado (opcional)
        check_finite: Si verificar valores finitos
    
    Returns:
        Diccionario con resultados de validación
    """
    validation = {
        "is_valid": True,
        "has_nan": False,
        "has_inf": False,
        "shape_match": True,
        "is_finite": True,
    }
    
    # Verificar NaN/Inf
    if check_finite:
        if torch.isnan(outputs).any():
            validation["has_nan"] = True
            validation["is_valid"] = False
        
        if torch.isinf(outputs).any():
            validation["has_inf"] = True
            validation["is_valid"] = False
        
        if not torch.isfinite(outputs).all():
            validation["is_finite"] = False
            validation["is_valid"] = False
    
    # Verificar shape
    if expected_shape is not None:
        if outputs.shape != expected_shape:
            validation["shape_match"] = False
            validation["is_valid"] = False
            logger.warning(
                f"Shape mismatch: expected {expected_shape}, got {outputs.shape}"
            )
    
    return validation
'''
        
        utils_subdir = utils_dir / "utils"
        (utils_subdir / "debugging_utils.py").write_text(debugging_content, encoding="utf-8")
    
    def _generate_visualization_utils(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de visualización"""
        
        viz_content = '''"""
Visualization Utilities - Utilidades de visualización
======================================================

Herramientas para visualizar resultados, métricas y modelos.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly no disponible")

logger = logging.getLogger(__name__)


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plotea curvas de entrenamiento.
    
    Args:
        train_losses: Lista de pérdidas de entrenamiento
        val_losses: Lista de pérdidas de validación (opcional)
        train_metrics: Métricas de entrenamiento (opcional)
        val_metrics: Métricas de validación (opcional)
        save_path: Ruta donde guardar (opcional)
        show: Si mostrar el plot
    """
    num_plots = 1
    if train_metrics:
        num_plots += len(train_metrics)
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]
    
    # Plot losses
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot metrics
    if train_metrics:
        for idx, (metric_name, values) in enumerate(train_metrics.items(), 1):
            if idx < len(axes):
                ax = axes[idx]
                ax.plot(epochs, values, 'b-', label=f'Train {metric_name}', linewidth=2)
                if val_metrics and metric_name in val_metrics:
                    ax.plot(epochs, val_metrics[metric_name], 'r-', 
                           label=f'Val {metric_name}', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'Training and Validation {metric_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot guardado en {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plotea matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        class_names: Nombres de clases (opcional)
        save_path: Ruta donde guardar (opcional)
        show: Si mostrar el plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Matriz de confusión guardada en {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_attention(
    attention_weights: np.ndarray,
    tokens: List[str],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Visualiza pesos de atención.
    
    Args:
        attention_weights: Matriz de pesos de atención
        tokens: Lista de tokens
        save_path: Ruta donde guardar (opcional)
        show: Si mostrar el plot
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar=True,
    )
    plt.title('Attention Weights')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Attention weights guardados en {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
'''
        
        utils_subdir = utils_dir / "utils"
        (utils_subdir / "visualization_utils.py").write_text(viz_content, encoding="utf-8")
    
    def _generate_preprocessing_utils(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de preprocesamiento"""
        
        preprocessing_content = '''"""
Preprocessing Utilities - Utilidades de preprocesamiento
=========================================================

Herramientas para preprocesamiento y transformación de datos.
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def normalize_data(
    data: Union[torch.Tensor, np.ndarray],
    method: str = "standard",
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normaliza datos.
    
    Args:
        data: Datos a normalizar
        method: Método de normalización (standard, minmax, robust)
        mean: Media para normalización estándar (opcional)
        std: Desviación estándar (opcional)
    
    Returns:
        Datos normalizados
    """
    is_tensor = isinstance(data, torch.Tensor)
    
    if not is_tensor:
        data = torch.from_numpy(data)
    
    if method == "standard":
        if mean is None:
            mean = data.mean()
        if std is None:
            std = data.std()
        
        normalized = (data - mean) / (std + 1e-8)
    
    elif method == "minmax":
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "robust":
        median = data.median()
        q75, q25 = torch.quantile(data, 0.75), torch.quantile(data, 0.25)
        iqr = q75 - q25
        normalized = (data - median) / (iqr + 1e-8)
    
    else:
        raise ValueError(f"Método de normalización {method} no soportado")
    
    if not is_tensor:
        normalized = normalized.numpy()
    
    return normalized


def augment_data(
    data: torch.Tensor,
    augmentation_type: str = "noise",
    **kwargs,
) -> torch.Tensor:
    """
    Aumenta datos con transformaciones.
    
    Args:
        data: Datos a aumentar
        augmentation_type: Tipo de aumentación (noise, flip, rotate)
        **kwargs: Argumentos adicionales
    
    Returns:
        Datos aumentados
    """
    if augmentation_type == "noise":
        noise_std = kwargs.get("noise_std", 0.1)
        noise = torch.randn_like(data) * noise_std
        return data + noise
    
    elif augmentation_type == "flip":
        return torch.flip(data, dims=kwargs.get("dims", [-1]))
    
    elif augmentation_type == "rotate":
        # Rotación simple para tensores 2D
        if data.dim() == 2:
            k = kwargs.get("k", 1)
            return torch.rot90(data, k=k)
        return data
    
    else:
        logger.warning(f"Tipo de aumentación {augmentation_type} no implementado")
        return data


def prepare_dataset(
    data: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    normalize: bool = True,
    augment: bool = False,
) -> Dict[str, Any]:
    """
    Prepara dataset para entrenamiento.
    
    Args:
        data: Datos
        labels: Etiquetas (opcional)
        normalize: Si normalizar datos
        augment: Si aumentar datos
    
    Returns:
        Diccionario con datos preparados
    """
    prepared = {}
    
    # Convertir a tensor si es necesario
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    if labels is not None and isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    # Normalizar
    if normalize:
        data = normalize_data(data)
    
    # Aumentar
    if augment:
        data = augment_data(data)
    
    prepared["data"] = data
    if labels is not None:
        prepared["labels"] = labels
    
    return prepared
'''
        
        utils_subdir = utils_dir / "utils"
        (utils_subdir / "preprocessing_utils.py").write_text(preprocessing_content, encoding="utf-8")
    
    def _generate_inspection_utils(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de inspección de modelos"""
        
        inspection_content = '''"""
Model Inspection Utilities - Utilidades de inspección de modelos
=================================================================

Herramientas para inspeccionar y analizar arquitecturas de modelos.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


def inspect_model_architecture(
    model: nn.Module,
    input_shape: Optional[tuple] = None,
) -> Dict[str, Any]:
    """
    Inspecciona la arquitectura del modelo.
    
    Args:
        model: Modelo a inspeccionar
        input_shape: Shape de input para forward pass (opcional)
    
    Returns:
        Diccionario con información de la arquitectura
    """
    architecture_info = {
        "model_class": model.__class__.__name__,
        "total_layers": len(list(model.modules())),
        "trainable_layers": sum(1 for m in model.modules() if hasattr(m, 'weight') and m.weight.requires_grad),
        "layers": [],
    }
    
    # Analizar capas
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layer_info = {
                "name": name,
                "type": module.__class__.__name__,
            }
            
            # Información de parámetros
            if hasattr(module, "weight") and module.weight is not None:
                layer_info["weight_shape"] = list(module.weight.shape)
                layer_info["weight_requires_grad"] = module.weight.requires_grad
            
            if hasattr(module, "bias") and module.bias is not None:
                layer_info["bias_shape"] = list(module.bias.shape)
            
            architecture_info["layers"].append(layer_info)
    
    # Forward pass si se proporciona input_shape
    if input_shape is not None:
        try:
            dummy_input = torch.randn(1, *input_shape)
            with torch.no_grad():
                output = model(dummy_input)
            architecture_info["input_shape"] = input_shape
            architecture_info["output_shape"] = list(output.shape)
        except Exception as e:
            logger.warning(f"No se pudo hacer forward pass: {e}")
    
    return architecture_info


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> Dict[str, int]:
    """
    Cuenta parámetros del modelo.
    
    Args:
        model: Modelo a analizar
        trainable_only: Si contar solo parámetros entrenables
    
    Returns:
        Diccionario con conteos
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params if trainable_only else total_params,
        "non_trainable_parameters": total_params - trainable_params,
        "total_mb": total_params * 4 / (1024**2),  # Asumiendo float32
    }


def analyze_model_complexity(
    model: nn.Module,
    input_shape: tuple,
) -> Dict[str, Any]:
    """
    Analiza complejidad del modelo.
    
    Args:
        model: Modelo a analizar
        input_shape: Shape de input
    
    Returns:
        Diccionario con análisis de complejidad
    """
    param_counts = count_parameters(model)
    
    # Estimar FLOPs (aproximado)
    dummy_input = torch.randn(1, *input_shape)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        # Estimación simple de FLOPs
        # Esto es una aproximación, para cálculo preciso usar herramientas especializadas
        estimated_flops = param_counts["total_parameters"] * 2  # Aproximación
        
        complexity = {
            **param_counts,
            "input_shape": input_shape,
            "output_shape": list(output.shape),
            "estimated_flops": estimated_flops,
            "estimated_gflops": estimated_flops / 1e9,
        }
        
        return complexity
    
    except Exception as e:
        logger.error(f"Error analizando complejidad: {e}")
        return param_counts
'''
        
        utils_subdir = utils_dir / "utils"
        (utils_subdir / "inspection_utils.py").write_text(inspection_content, encoding="utf-8")

